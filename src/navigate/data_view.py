#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import cv2
from cv_bridge import CvBridge
import csv
import yaml
import json
from datetime import datetime
import tf2_ros
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, NavSatFix

class MessageSynchronizer:
    def __init__(self, bag_path, output_dir, max_time_diff=0.05):
        """
        Initialize the synchronizer for extracting time-synchronized data from a rosbag.
        
        Args:
            bag_path: Path to the rosbag
            output_dir: Directory to save the extracted data
            max_time_diff: Maximum time difference in seconds to consider messages synchronized
        """
        self.bag_path = bag_path
        self.output_dir = output_dir
        self.max_time_diff = max_time_diff
        
        # Create output directories
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize data structures
        self.odom_msgs = []
        self.gnss_msgs = []
        self.image_msgs = []
        self.bridge = CvBridge()
        
        # Initialize CSV file for pose data
        self.csv_path = os.path.join(output_dir, "poses.csv")
        self.csv_file = open(self.csv_path, 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'image_filename',
            'odom_x', 'odom_y', 'odom_z',
            'odom_qx', 'odom_qy', 'odom_qz', 'odom_qw',
            'gnss_latitude', 'gnss_longitude', 'gnss_altitude'
        ])
        
        # Metadata
        self.metadata = {
            'bag_path': bag_path,
            'extraction_time': datetime.now().isoformat(),
            'max_time_diff': max_time_diff,
            'synchronized_samples': 0
        }

    def process_bag(self):
        """Process the rosbag and extract synchronized data."""
        print(f"Processing bag: {self.bag_path}")
        
        # Set up the reader
        storage_options = StorageOptions(uri=self.bag_path)
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        # Create a reader
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Get topic types
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic_info.name: topic_info.type for topic_info in topic_types}
        
        # Process all messages
        topic_count = {topic: 0 for topic in type_map.keys()}
        
        while reader.has_next():
            topic_name, data, t = reader.read_next()
            topic_count[topic_name] += 1
            
            # Get the message type
            msg_type = get_message(type_map[topic_name])
            msg = deserialize_message(data, msg_type)
            
            # Store messages by type
            if topic_name == "/lio_sam_ros2/mapping/odometry":
                self.odom_msgs.append((t, msg))
            elif topic_name == "/gnss":
                self.gnss_msgs.append((t, msg))
            elif topic_name == "/robot0/camera/image_raw":
                self.image_msgs.append((t, msg))
        
        # Print statistics
        print("\nMessage counts by topic:")
        for topic, count in topic_count.items():
            if count > 0:
                print(f"  {topic}: {count}")
        
        print(f"\nFound {len(self.odom_msgs)} odometry messages")
        print(f"Found {len(self.gnss_msgs)} GNSS messages")
        print(f"Found {len(self.image_msgs)} image messages")
        
        # Sort messages by timestamp
        self.odom_msgs.sort(key=lambda x: x[0])
        self.gnss_msgs.sort(key=lambda x: x[0])
        self.image_msgs.sort(key=lambda x: x[0])
        
        # Synchronize and extract data
        self._synchronize_and_extract()
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        self.csv_file.close()
        print(f"\nExtracted {self.metadata['synchronized_samples']} synchronized samples")
        print(f"Data saved to {self.output_dir}")

    def _synchronize_and_extract(self):
        """Find synchronized messages and extract the data."""
        synchronized_count = 0
        
        # Use images as the reference
        for img_idx, (img_time, img_msg) in enumerate(self.image_msgs):
            # Find the closest odometry message
            closest_odom = self._find_closest_message(img_time, self.odom_msgs)
            if closest_odom is None:
                continue
            
            # Find the closest GNSS message
            closest_gnss = self._find_closest_message(img_time, self.gnss_msgs)
            if closest_gnss is None:
                # We can proceed without GNSS if needed
                closest_gnss = (img_time, None)
            
            # Check if they are within the time threshold
            odom_time_diff = abs(closest_odom[0] - img_time) * 1e-9  # convert ns to seconds
            if odom_time_diff > self.max_time_diff:
                continue
            
            # Extract the data
            odom_msg = closest_odom[1]
            gnss_msg = closest_gnss[1]
            
            # Save the image
            timestamp_str = str(img_time)
            img_filename = f"{timestamp_str}.jpg"
            img_path = os.path.join(self.images_dir, img_filename)
            
            try:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
                cv2.imwrite(img_path, cv_image)
                
                # Write to CSV
                row = [timestamp_str, img_filename]
                
                # Add odometry data
                row.extend([
                    odom_msg.pose.pose.position.x,
                    odom_msg.pose.pose.position.y,
                    odom_msg.pose.pose.position.z,
                    odom_msg.pose.pose.orientation.x,
                    odom_msg.pose.pose.orientation.y,
                    odom_msg.pose.pose.orientation.z,
                    odom_msg.pose.pose.orientation.w
                ])
                
                # Add GNSS data (if available)
                if gnss_msg:
                    row.extend([
                        gnss_msg.latitude,
                        gnss_msg.longitude,
                        gnss_msg.altitude
                    ])
                else:
                    row.extend([None, None, None])
                
                self.csv_writer.writerow(row)
                synchronized_count += 1
                
                # Print progress
                if synchronized_count % 100 == 0:
                    print(f"Extracted {synchronized_count} synchronized samples...")
            
            except Exception as e:
                print(f"Error processing image at time {img_time}: {e}")
        
        self.metadata['synchronized_samples'] = synchronized_count

    def _find_closest_message(self, target_time, messages):
        """Find the message closest in time to the target time."""
        if not messages:
            return None
        
        # Binary search to find the closest message
        left, right = 0, len(messages) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if messages[mid][0] < target_time:
                left = mid + 1
            else:
                right = mid - 1
        
        # Check boundaries
        if left >= len(messages):
            closest_idx = right
        elif right < 0:
            closest_idx = left
        else:
            # Check which neighbor is closer
            left_diff = abs(target_time - messages[left][0])
            right_diff = abs(target_time - messages[right][0])
            closest_idx = left if left_diff < right_diff else right
        
        return messages[closest_idx]

def main():
    parser = argparse.ArgumentParser(description="Extract synchronized data from a ROS2 bag file")
    parser.add_argument("bag_path", help="Path to the ROS2 bag directory")
    parser.add_argument("--output", "-o", default="synchronized_data", help="Output directory")
    parser.add_argument("--time-diff", "-t", type=float, default=0.05, 
                        help="Maximum time difference in seconds to consider messages synchronized")
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init(args=None)
    
    # Check if bag_path exists
    if not os.path.exists(args.bag_path):
        print(f"Error: Bag path {args.bag_path} does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process the bag
    synchronizer = MessageSynchronizer(args.bag_path, args.output, args.time_diff)
    synchronizer.process_bag()
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()