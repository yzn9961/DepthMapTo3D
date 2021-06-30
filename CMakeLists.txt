cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE "Debug")
# 添加c++ 11标准支持
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")
# Pangolin
find_package(Pangolin REQUIRED)
# Eigen
include_directories("/usr/include/eigen3")
# opencv
set(OpenCV_DIR "/home/telemoro/openCV/opencv-4.4.0/build/")
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})
# 打印消息
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
#message(STATUS "    libraries: ${OpenCV_LIBS}")
#message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
add_executable(DepthMap DepthMap.cpp)
target_link_libraries(DepthMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
