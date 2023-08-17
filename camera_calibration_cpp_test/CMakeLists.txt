cmake_minimum_required(VERSION 3.17)
project(camera_calibration)
find_package(OpenCV REQUIRED)

##下面是输出信息
message(STATUS "Opnecv ;ibrary status: ")
message(STATUS "> version: ${OpenCV_VERSION} ")
message(STATUS "libraries: ${OpenCV_LIBS} ")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS}  ")

set(CMAKE_CXX_STANDARD 14)
# set(SOURCES camera_calibration.cpp cppmatrix.cpp) 
file(GLOB sources *.cpp *.h)
set(SOURCES ${sources}) 
message(${SOURCES})

include_directories(${OpenCV_INCLUDE_DIRS}) 
add_executable(camera_calibration ${SOURCES})

target_link_libraries(camera_calibration ${OpenCV_LIBS})