#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import datetime
import numpy as np
from pathlib import Path
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class DataSaver(Node):
    def __init__(self):
        super().__init__('data_saver')
        
        # Create directories for saving data
        self.base_dir = Path.home() / 'ros2_data'
        self.odom_dir = self.base_dir / 'odom'
        self.image_dir = self.base_dir / 'images'
        
        self.odom_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Create a timestamp for this run
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.odom_file = self.odom_dir / f'odom_{self.timestamp}.txt'
        
        # Open odometry file and write header
        with open(self.odom_file, 'w') as f:
            f.write('# timestamp,seq,frame_id,child_frame_id,')
            f.write('pos_x,pos_y,pos_z,')
            f.write('orient_x,orient_y,orient_z,orient_w,')
            f.write('lin_vel_x,lin_vel_y,lin_vel_z,')
            f.write('ang_vel_x,ang_vel_y,ang_vel_z\n')
        
        # Create QoS profiles
        # For odometry (standard profile is usually fine)
        odom_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Try multiple QoS profiles for the image topic
        # BEST_EFFORT is typically used for sensor data like images
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        # Create subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/robot0/odom',
            self.odom_callback,
            qos_profile=odom_qos)
        
        self.image_sub = self.create_subscription(
            Image,
            '/robot0/camera/image_raw',
            self.image_callback,
            qos_profile=sensor_qos)
        
        # Add a timer to report status periodically
        self.timer = self.create_timer(5.0, self.status_callback)
        
        # Counters for tracking
        self.image_count = 0
        self.odom_count = 0
        
        self.get_logger().info(f'DataSaver initialized. Saving to {self.base_dir}')
        self.get_logger().info(f'Odometry data: {self.odom_file}')
        self.get_logger().info(f'Images: {self.image_dir}')
        
        # Debug available topics
        self.topic_timer = self.create_timer(10.0, self.check_topics)
    
    def check_topics(self):
        """Print available topics to help with debugging"""
        from rclpy.topic_endpoint_info import TopicEndpointTypeEnum
        
        # Get topic names and types
        topic_names_and_types = self.get_topic_names_and_types()
        
        # Filter for our topics of interest
        odom_topics = [t for t in topic_names_and_types if 'odom' in t[0]]
        image_topics = [t for t in topic_names_and_types if 'image' in t[0]]
        
        self.get_logger().info("Available odometry topics:")
        for topic_name, topic_types in odom_topics:
            self.get_logger().info(f"  - {topic_name}: {topic_types}")
            
            # Get more details about publishers
            publishers = self.get_publishers_info_by_topic(topic_name)
            for pub in publishers:
                qos = pub.qos_profile
                self.get_logger().info(f"    Publisher QoS - reliability: {qos.reliability}, "
                                      f"durability: {qos.durability}, "
                                      f"history: {qos.history}, "
                                      f"depth: {qos.depth}")
        
        self.get_logger().info("Available image topics:")
        for topic_name, topic_types in image_topics:
            self.get_logger().info(f"  - {topic_name}: {topic_types}")
            
            # Get more details about publishers
            publishers = self.get_publishers_info_by_topic(topic_name)
            for pub in publishers:
                qos = pub.qos_profile
                self.get_logger().info(f"    Publisher QoS - reliability: {qos.reliability}, "
                                      f"durability: {qos.durability}, "
                                      f"history: {qos.history}, "
                                      f"depth: {qos.depth}")
        
        # After the first run, we can turn off this timer
        self.topic_timer.cancel()
    
    def status_callback(self):
        """Report status periodically"""
        self.get_logger().info(f'Status: {self.odom_count} odometry msgs, {self.image_count} images saved')
    
    def odom_callback(self, msg):
        """Process odometry messages"""
        # Extract timestamp
        timestamp = self.get_clock().now().to_msg()
        timestamp_str = f"{timestamp.sec}.{timestamp.nanosec:09d}"
        
        # Format data
        data = [
            timestamp_str,
            str(msg.header.stamp.sec) + "." + f"{msg.header.stamp.nanosec:09d}",
            msg.header.frame_id,
            msg.child_frame_id,
            str(msg.pose.pose.position.x),
            str(msg.pose.pose.position.y),
            str(msg.pose.pose.position.z),
            str(msg.pose.pose.orientation.x),
            str(msg.pose.pose.orientation.y),
            str(msg.pose.pose.orientation.z),
            str(msg.pose.pose.orientation.w),
            str(msg.twist.twist.linear.x),
            str(msg.twist.twist.linear.y),
            str(msg.twist.twist.linear.z),
            str(msg.twist.twist.angular.x),
            str(msg.twist.twist.angular.y),
            str(msg.twist.twist.angular.z)
        ]
        
        # Write to file
        with open(self.odom_file, 'a') as f:
            f.write(','.join(data) + '\n')
        
        self.odom_count += 1
    
    def image_callback(self, msg):
        """Process image messages"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Log that we received an image
            self.get_logger().info(f'Received image with shape {cv_image.shape}, encoding {msg.encoding}')
            
            # Create filename with timestamp
            timestamp = self.get_clock().now().to_msg()
            timestamp_str = f"{timestamp.sec}.{timestamp.nanosec:09d}"
            
            # Save image
            filename = self.image_dir / f'image_{self.timestamp}_{self.image_count:06d}.png'
            cv2.imwrite(str(filename), cv_image)
            
            # Save image metadata
            meta_filename = self.image_dir / f'image_{self.timestamp}_{self.image_count:06d}.txt'
            with open(meta_filename, 'w') as f:
                f.write(f'timestamp: {timestamp_str}\n')
                f.write(f'frame_id: {msg.header.frame_id}\n')
                f.write(f'height: {msg.height}\n')
                f.write(f'width: {msg.width}\n')
                f.write(f'encoding: {msg.encoding}\n')
            
            # Increment counter
            self.image_count += 1
            
            self.get_logger().info(f'Saved image: {filename}')
        
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = DataSaver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down due to keyboard interrupt...')
    except Exception as e:
        node.get_logger().error(f'Error: {str(e)}')
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()