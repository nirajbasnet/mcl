# Particle Filter implementation 
This code consists of implementation of Monte carlo localization. Turtlebot in a Gazebo environment is used for the demo, however it can be used for other robots in a real world setting as well provided that laserscan and odometry updates are passed.
The sensor model used is the naive Breshenham's ray casting, so the speed isn't that great. However, it can be replaced with lookup table or other methods like CDDT,RM(Ray Marching),etc. to get huge boost in localization update rate. 

## Usage 
__1. Run the turtlebot demo world. __
```sh
roslaunch mcl turtlebot_in_stage.launch
```
__2. Run mcl node.
```sh
roslaunch mcl mcl.launch
```
You can change the parameters the mcl in the config file mcl.yaml located inside config folder.

