cmake_minimum_required(VERSION 3.5)
project(hand_pose_detection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    src/gabor_filters/build/CMakeFiles/2.8.12.2/CompilerIdC/CMakeCCompilerId.c
    src/gabor_filters/build/CMakeFiles/2.8.12.2/CompilerIdCXX/CMakeCXXCompilerId.cpp
    src/gabor_filters/EF/extract_features.cpp
    src/gabor_filters/EF/extract_features.h
    src/gabor_filters/EGF/extract_gabor_filters.cpp
    src/gabor_filters/EGF/extract_gabor_filters.h
    src/gabor_filters/sift_models/build/CMakeFiles/2.8.12.2/CompilerIdC/CMakeCCompilerId.c
    src/gabor_filters/sift_models/build/CMakeFiles/2.8.12.2/CompilerIdCXX/CMakeCXXCompilerId.cpp
    src/gabor_filters/sift_models/create_models.cpp
    src/gabor_filters/extract_gabor_filters.cpp
    src/gabor_filters/extract_gabor_filters1.cpp
    src/gabor_filters/extract_gabor_filters1.h
    src/gabor_filters/main.cpp
    src/hough_transform/make_template.cpp)

add_executable(hand_pose_detection ${SOURCE_FILES})