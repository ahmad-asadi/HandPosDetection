cmake_minimum_required(VERSION 2.8.12)

project(SIFT_Models)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/../EGF")
include_directories("${PROJECT_SOURCE_DIR}/../EF")


file( GLOB EGF_SOURCES ../EGF/*.cpp )
file( GLOB EGF_HEADERS ../EGF/*.h )

file( GLOB EF_SOURCES ../EF/*.cpp )
file( GLOB EF_HEADERS ../EF/*.h )

# add_library( EGF ${EGF_SOURCES} ${EGF_HEADERS} )
# add_library( EF ${EF_SOURCES} ${EF_HEADERS} )

add_executable(create_models create_models.cpp)
target_link_libraries(create_models ${OpenCV_LIBS} ${EGF_SOURCES} ${EGF_HEADERS} ${EF_SOURCES} ${EF_HEADERS})