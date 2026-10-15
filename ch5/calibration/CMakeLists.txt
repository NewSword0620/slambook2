project(calibration)

add_executable(calibration calibration.cpp)
# 链接OpenCV库
target_link_libraries(calibration ${OpenCV_LIBS})
