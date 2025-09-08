#!/bin/bash
# filepath: quad/run.sh

# Path to ROS2 custom messages setup script
ROS2_SETUP="ros2_custom_msgs/install/local_setup.bash"

# Source ROS2 setup
source $ROS2_SETUP

# Start both nodes in the background
python3 quad/sim_node.py &
PID1=$!
python3 quad/control_node.py &
PID2=$!

# Function to kill both on Ctrl+C
cleanup() {
    echo "Killing both nodes..."
    kill $PID1 $PID2
    wait $PID1 $PID2 2>/dev/null
    exit
}

trap cleanup SIGINT

# Wait for both to finish
wait $PID1 $PID2