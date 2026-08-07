#!/bin/bash

# Script to record navigation data into a rosbag file
# Usage: ./record_navigation_data.sh [output_directory] [duration_in_seconds]

# Default values
OUTPUT_DIR="${1:-navigation_data}"
DURATION="${2:-0}"  # 0 means record until Ctrl+C is pressed

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get current timestamp for the bag name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BAG_NAME="${OUTPUT_DIR}/navigation_data_${TIMESTAMP}"

# Topics to record
TOPICS=(
    "/lio_sam_ros2/mapping/odometry"    # LIO-SAM odometry (primary pose source)
    "/gnss"                             # GPS data (for validation/fusion)
    "/robot0/camera/image_raw"          # Camera images
    "/robot0/camera/camera_info"        # Camera calibration info
    "/tf"                               # Transform frames
    "/tf_static"                        # Static transform frames
    "/robot0/imu"                       # IMU data (for potential fusion)
    "/utlidar/robot_pose"               # Alternative pose source
    "/robot0/odom"                      # Robot odometry (for comparison)
)

echo "====================================================="
echo "Starting to record navigation data to: $BAG_NAME"
echo "Recording the following topics:"
for topic in "${TOPICS[@]}"; do
    echo " - $topic"
done
echo "====================================================="
echo "Press Ctrl+C to stop recording (if no duration specified)"
echo

# Join topics with a space for the ros2 bag record command
TOPIC_LIST=$(IFS=' '; echo "${TOPICS[*]}")

if [ "$DURATION" -gt 0 ]; then
    echo "Recording for $DURATION seconds..."
    timeout "$DURATION" ros2 bag record -o "$BAG_NAME" $TOPIC_LIST
else
    echo "Recording until interrupted..."
    ros2 bag record -o "$BAG_NAME" $TOPIC_LIST
fi

echo
echo "====================================================="
echo "Recording finished. Bag saved to: $BAG_NAME"
echo "====================================================="