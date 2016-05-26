cmake_minimum_required(VERSION 2.8.12)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(extract_gabor_filters extract_gabor_filters.cpp)
target_link_libraries(extract_gabor_filters ${OpenCV_LIBS})