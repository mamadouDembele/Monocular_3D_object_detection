# Monocular 3D detection

## requirements

* cuda 12.5
* Tensorrt 10.4
* ROS2 humble or later

## Installation
```
mkdir -p ~/mono3d_ws/src
mkdir ~/onnx/
cd ~/onnx/
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz
tar zxvf onnxruntime-linux-x64-1.19.2.tgz
mv onnxruntime-linux-x64-1.19.2 onnxruntime
cd ~/mono3d_ws/src
git clone https://github.com/mamadouDembele/Monocular_3D_object_detection.git
cd ..
colcon build --symlink-install --packages-select camera_detection_single_stage
```
## Run
```
source install/setup.bash
ros2 launch camera_detection_single_stage camera_detection_single_stage.launch.xml
```

## Acknowledgement
https://github.com/Owen-Liuyuxuan/ros2_vision_inference/tree/onnx
https://github.com/ApolloAuto/apollo/tree/master/modules/perception
