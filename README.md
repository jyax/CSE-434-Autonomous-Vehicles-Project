# Project 08: Detection of crosswalks and pedestrians in autonomous vehicles
## Setup
Clone this repo:
```
git clone https://gitlab.msu.edu/yaxjacob/av_project_08.git
```
Clone the yolov5 repo and install its requirements:
```
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```
Follow this folder hierarchy:
```
ros_ws/
├─ src/
│  ├─ av_project_08/
│  ├─ yolov5/
```

## Running
In the ros_ws workspace, build the `project` package:
```
colcon build --symlink-install packages-select project
```

Launch the gazebo world:
```
cd project/launch
ros2 launch crosswalk.launch.py
```

In a seperate terminal run the detection code:
```
ros2 run project detect
```
