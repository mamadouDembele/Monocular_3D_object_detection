cmake_minimum_required(VERSION 3.8)
project(camera_detection_single_stage)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# -------------- OpenCV  ------------------#
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- Compile CUDA for FP16 inference if needed  ------------------#
option(USE_CUDA "Enable CUDA support" ON)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
add_definitions(-DUSE_CUDA)

# -------------- ONNXRUNTIME  ------------------#
set(ONNXRUNTIME_ROOT "~/onnx/onnxruntime")
include_directories(${ONNXRUNTIME_ROOT}/include)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

ament_auto_add_library(${PROJECT_NAME}_component SHARED 
                      src/camera_detection_single_stage_node.cpp
                      src/monoflex.cpp
                      src/postprocess.cpp)

target_link_libraries(${PROJECT_NAME}_component
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)

rclcpp_components_register_node(${PROJECT_NAME}_component
  PLUGIN "${PROJECT_NAME}::CameraDetectionSingleStageNode"
  EXECUTABLE ${PROJECT_NAME}_node
)

ament_auto_package(INSTALL_TO_SHARE
    launch
    config
    model
)