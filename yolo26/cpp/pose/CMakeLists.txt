cmake_minimum_required(VERSION 3.10)

project(trt_yolo26_pose)

add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE release)

find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("Embed_platform on")
  include_directories(/usr/include/aarch64-linux-gnu)
  link_directories(/usr/lib/aarch64-linux-gnu)
else()
  message("Embed_platform off")
  include_directories(/usr/include/x86_64-linux-gnu)
  link_directories(/usr/lib/x86_64-linux-gnu)
endif()

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_library(trt_yolo26_pose SHARED ${SRCS})
target_link_libraries(trt_yolo26_pose cudart nvinfer nvonnxparser ${OpenCV_LIBS})

add_executable(pose ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(pose trt_yolo26_pose)
