cmake_minimum_required(VERSION 2.8.3)
project(EKF_SLAM)
add_compile_options(-std=c++14)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)
#############################
#Dependencies
#############################

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


##############################
#SLAM
##############################
add_executable(ekf_slam src/ekf_slam.cpp)
target_link_libraries(ekf_slam ${OpenCV_LIBS} )