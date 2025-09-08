
# ROS2 
# source ros2_custom_msgs/install/local_setup.bash 
# or better add to ~/.bashrc
# echo 'source {INSERT YOUR FOLDER PATH HERE}/ros2_custom_msgs/install/local_setup.bash' >> ~/.bashrc


# MAKE SURE THE terminal where you run rosbridge has sourced ros2_custom_msgs
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
ros2 launch foxglove_bridge foxglove_bridge_launch.xml


useful commands:
 1. check if sr msgs are sourced
 ros2 interface list -m 
 2. check if topics are available
 ros2 topic list
 3. check topic update rate
    ros2 topic hz /topic_name

