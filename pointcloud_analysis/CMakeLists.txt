cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(pytorch_cpp_example)

set( CMAKE_CXX_FLAGS "-std=c++11" )

include_directories(include src)
file(GLOB SOURCES
    include/*.h
    src/*.cpp
)

message(STATUS "Sources: ${SOURCES}")

set(CMAKE_PREFIX_PATH /home/albert/coding/libraries/libtorch_latest; /usr)
set (CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS} -fpermissive")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
include_directories(

  ${OpenCV_INCLUDE_DIRS}
  usr/s
  )
#${TORCH_INCLUDE_DIRS}
  message(STATUS "OpenCV library status:")
  message(STATUS "    config: ${OpenCV_DIR}")
  message(STATUS "    version: ${OpenCV_VERSION}")
  message(STATUS "    libraries: ${OpenCV_LIBS}")
  message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(program main.cpp ${SOURCES})

target_link_libraries(program
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
      #  glog::glog
        )

#set_target_properties(program PROPERTIES COMPILE_FLAGS)
set_property(TARGET program PROPERTY CXX_STANDARD 14)

#/home/albert/coding/libraries/opencv4_cpp/opencv/build/lib/libopencv_highgui.so.4.0.0
