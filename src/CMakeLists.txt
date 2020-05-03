cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(segmentation)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

add_executable (main main.cpp)
target_link_libraries (main ${OpenCV_LIBS} stdc++fs)
