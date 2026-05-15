cmake_minimum_required(VERSION 3.10)

project(trt_yolo11_track)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE release)

include_directories("/usr/include/eigen3")

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# ============= cuda ============
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# ============= tensorrt ============
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("Embed_platform on")
  include_directories(/usr/include/aarch64-linux-gnu)
  link_directories(/usr/lib/aarch64-linux-gnu)
else()
  message("Embed_platform off")
  include_directories(/usr/include/x86_64-linux-gnu)
  link_directories(/usr/lib/x86_64-linux-gnu)
endif()

# ============ opencv ============
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# =========== bytetrack lib ===========
include_directories(${PROJECT_SOURCE_DIR}/../../../shared/cpp/bytetrack/include)
file(GLOB_RECURSE SRCS01 ${PROJECT_SOURCE_DIR}/../../../shared/cpp/bytetrack/src/*.cpp)
add_library(trt_shared_bytetrack SHARED ${SRCS01})
target_link_libraries(trt_shared_bytetrack cudart nvinfer ${OpenCV_LIBS})

# ============= yolo11 lib =============
include_directories(${PROJECT_SOURCE_DIR}/../detect/include)
file(GLOB_RECURSE SRCS02 ${PROJECT_SOURCE_DIR}/../detect/src/*.cpp ${PROJECT_SOURCE_DIR}/../detect/src/*.cu)
cuda_add_library(trt_yolo11_detect SHARED ${SRCS02})
target_link_libraries(trt_yolo11_detect nvinfer cudart nvonnxparser ${OpenCV_LIBS})

# ======== main execute file ========
add_executable(trt_yolo11_track_cli ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(trt_yolo11_track_cli trt_shared_bytetrack trt_yolo11_detect)
