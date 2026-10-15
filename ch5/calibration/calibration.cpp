#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <dirent.h>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

// 张正友标定法: 用多张棋盘格照片标定相机内参和畸变系数
// 用法: ./calibration <图片目录> <棋盘格内角点列数> <内角点行数> [方格边长,单位米,默认0.025]
// 例:  ./calibration images 9 6 0.025
int main(int argc, char **argv) {
  if (argc < 4) {
    cout << "用法: " << argv[0] << " <图片目录> <内角点列数> <内角点行数> [方格边长m]" << endl;
    cout << "注意: 内角点数 = 棋盘格格子数 - 1 (如 10x7 的棋盘格应输入 9 6)" << endl;
    return 1;
  }
  string image_dir = argv[1];
  int board_cols = atoi(argv[2]);  // 内角点列数
  int board_rows = atoi(argv[3]);  // 内角点行数
  double square_size = (argc >= 5) ? atof(argv[4]) : 0.025;
  cv::Size board_size(board_cols, board_rows);

  // 1. 读取目录下的所有图片
  DIR *dir = opendir(image_dir.c_str());
  if (dir == nullptr) {
    cerr << "无法打开目录: " << image_dir << endl;
    return 1;
  }
  vector<string> filenames;
  struct dirent *ent;
  while ((ent = readdir(dir)) != nullptr) {
    string name = ent->d_name;
    if (name.size() > 4) {
      string ext = name.substr(name.rfind('.') + 1);
      transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp") {
        filenames.push_back(name);
      }
    }
  }
  closedir(dir);
  sort(filenames.begin(), filenames.end());
  if (filenames.size() < 5) {
    cerr << "图片太少(找到" << filenames.size() << "张), 标定至少需要5张不同角度的棋盘格照片." << endl;
    return 1;
  }
  cout << "共找到 " << filenames.size() << " 张图片." << endl;

  // 2. 提取每张图的角点
  vector<vector<cv::Point3f>> points_3d;  // 世界坐标(棋盘格平面 Z=0)
  vector<vector<cv::Point2f>> points_2d;  // 像素坐标
  cv::Mat first_image;
  vector<cv::Point3f> obj_points;  // 一张图的3D点, 所有图相同
  for (int v = 0; v < board_rows; v++) {
    for (int u = 0; u < board_cols; u++) {
      obj_points.push_back(cv::Point3f(u * square_size, v * square_size, 0));
    }
  }

  for (const string &name : filenames) {
    cv::Mat image = cv::imread(image_dir + "/" + name);
    if (image.empty()) {
      cerr << "无法读取: " << name << ", 跳过" << endl;
      continue;
    }
    // 保存第一张图的尺寸用于初始化相机矩阵
    if (first_image.empty()) first_image = image.clone();

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(
        gray, board_size, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    if (!found) {
      cout << "未在 " << name << " 中检测到角点, 跳过" << endl;
      continue;
    }
    // 亚像素角点精化, 提高标定精度
    cv::cornerSubPix(
        gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

    points_3d.push_back(obj_points);
    points_2d.push_back(corners);
    cout << "  " << name << ": 角点检测成功" << endl;
  }

  if (points_2d.size() < 5) {
    cerr << "有效图片不足5张, 无法标定." << endl;
    return 1;
  }

  // 3. 标定: 解出内参K与畸变系数
  cv::Mat K, dist_coeffs;                 // 内参矩阵, 畸变系数(k1,k2,p1,p2,k3)
  vector<cv::Mat> rvecs, tvecs;           // 每张图的外参(旋转向量/平移)
  cout << "\n开始标定..." << endl;
  double rms = cv::calibrateCamera(
      points_3d, points_2d, first_image.size(), K, dist_coeffs, rvecs, tvecs,
      cv::CALIB_FIX_K3);

  // 4. 输出结果
  cout << "\n=============== 标定结果 ===============" << endl;
  cout << "总体重投影误差 RMS = " << rms << " 像素 (越小越好, 一般应 < 0.5)" << endl;
  cout << "\n内参矩阵 K = " << endl << K << endl;
  cout << "\n畸变系数 [k1, k2, p1, p2, k3] = " << dist_coeffs.t() << endl;
  cout << "\n图像尺寸: " << first_image.size() << endl;

  // 5. 检查每张图的重投影误差, 找出拍得不好的图
  cout << "\n各图片重投影误差:" << endl;
  for (size_t i = 0; i < points_2d.size(); i++) {
    vector<cv::Point2f> projected;
    cv::projectPoints(points_3d[i], rvecs[i], tvecs[i], K, dist_coeffs, projected);
    double err = norm(cv::Mat(points_2d[i]), cv::Mat(projected), cv::NORM_L2)
                 / sqrt((double)projected.size());
    cout << "  " << filenames[i] << ": " << err << " 像素" << endl;
  }

  // 6. 保存结果到yaml, 供后续程序读取
  cv::FileStorage fs("calibration.yaml", cv::FileStorage::WRITE);
  fs << "camera_matrix" << K;
  fs << "dist_coeffs" << dist_coeffs;
  fs << "image_width" << first_image.cols;
  fs << "image_height" << first_image.rows;
  fs << "rms_error" << rms;
  fs.release();
  cout << "\n结果已保存到 calibration.yaml" << endl;

  // 展示一张角点检测效果图
  cv::Mat demo = cv::imread(image_dir + "/" + filenames[0]);
  drawChessboardCorners(demo, board_size, points_2d[0], true);
  cv::imshow("corners: " + filenames[0], demo);
  cv::waitKey(0);
  return 0;
}
