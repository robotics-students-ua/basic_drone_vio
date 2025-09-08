#!/bin/bash

# Script to clean and rebuild sr_msgs package
# This script deletes install/sr_msgs and build/sr_msgs directories
# and rebuilds the package using colcon build --packages-select sr_msgs

set -e  # Exit on any error

# Get the workspace root directory (assuming script is in src/sr_msgs/)
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Workspace root: $WORKSPACE_ROOT"

# Change to workspace root
cd "$WORKSPACE_ROOT"

# Delete install/sr_msgs directory if it exists
if [ -d "install/sr_msgs" ]; then
    echo "Removing install/sr_msgs..."
    rm -rf install/sr_msgs
else
    echo "install/sr_msgs directory not found, skipping..."
fi

# Delete build/sr_msgs directory if it exists
if [ -d "build/sr_msgs" ]; then
    echo "Removing build/sr_msgs..."
    rm -rf build/sr_msgs
else
    echo "build/sr_msgs directory not found, skipping..."
fi

echo "Building sr_msgs package..."
colcon build --packages-select sr_msgs

echo "Rebuild completed successfully!"