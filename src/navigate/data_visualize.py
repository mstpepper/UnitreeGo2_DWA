#!/usr/bin/env python3

import os
import sys
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

def load_data(csv_path, images_dir):
    """Load data from the CSV file and return structured data."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'timestamp': row['timestamp'],
                'image_path': os.path.join(images_dir, row['image_filename']),
                'position': np.array([
                    float(row['odom_x']), 
                    float(row['odom_y']), 
                    float(row['odom_z'])
                ]),
                'orientation': np.array([
                    float(row['odom_qx']), 
                    float(row['odom_qy']), 
                    float(row['odom_qz']), 
                    float(row['odom_qw'])
                ]),
                'gnss': {
                    'latitude': float(row['gnss_latitude']) if row['gnss_latitude'] else None,
                    'longitude': float(row['gnss_longitude']) if row['gnss_longitude'] else None,
                    'altitude': float(row['gnss_altitude']) if row['gnss_altitude'] else None
                }
            })
    return data

def plot_trajectory(data, output_path=None):
    """Plot the 3D trajectory and save it if output_path is provided."""
    positions = np.array([item['position'] for item in data])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Robot Path')
    
    # Plot the start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    
    # Plot points along the path
    step = max(1, len(positions) // 50)  # Plot up to 50 points
    ax.scatter(positions[::step, 0], positions[::step, 1], positions[::step, 2], c='k', s=20, alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Robot Trajectory')
    ax.legend()
    
    # Set aspect ratio
    max_range = np.max([
        np.ptp(positions[:, 0]), 
        np.ptp(positions[:, 1]), 
        np.ptp(positions[:, 2])
    ])
    mid_x = np.mean([np.min(positions[:, 0]), np.max(positions[:, 0])])
    mid_y = np.mean([np.min(positions[:, 1]), np.max(positions[:, 1])])
    mid_z = np.mean([np.min(positions[:, 2]), np.max(positions[:, 2])])
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Trajectory plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_gnss_path(data, output_path=None):
    """Plot the GNSS trajectory on a 2D map and save it if output_path is provided."""
    # Filter out data points without GNSS information
    valid_gnss = [item for item in data if item['gnss']['latitude'] is not None]
    
    if not valid_gnss:
        print("No valid GNSS data available.")
        return
    
    # Extract latitude and longitude
    latitudes = [item['gnss']['latitude'] for item in valid_gnss]
    longitudes = [item['gnss']['longitude'] for item in valid_gnss]
    
    plt.figure(figsize=(10, 8))
    
    # Plot the path
    plt.plot(longitudes, latitudes, 'b-', linewidth=2)
    
    # Plot the start and end points
    plt.scatter(longitudes[0], latitudes[0], c='g', s=100, label='Start')
    plt.scatter(longitudes[-1], latitudes[-1], c='r', s=100, label='End')
    
    # Plot points along the path
    step = max(1, len(latitudes) // 50)  # Plot up to 50 points
    plt.scatter(longitudes[::step], latitudes[::step], c='k', s=20, alpha=0.5)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GNSS Trajectory')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"GNSS plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def create_combined_visualization(data, output_dir, sample_rate=10):
    """Create a combined visualization of trajectory with images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sample the data points
    samples = data[::sample_rate]
    if len(samples) > 20:  # Cap at 20 samples for cleaner visualization
        step = len(samples) // 20
        samples = samples[::step]
    
    # Prepare positions for plotting
    positions = np.array([item['position'] for item in data])
    sample_positions = np.array([item['position'] for item in samples])
    
    # Create a figure for the combined visualization
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 4)
    
    # Plot the 3D trajectory
    ax_traj = fig.add_subplot(gs[:, :2], projection='3d')
    ax_traj.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax_traj.scatter(sample_positions[:, 0], sample_positions[:, 1], sample_positions[:, 2], 
                   c='r', s=50, alpha=0.8)
    
    # Number the sample points
    for i, pos in enumerate(sample_positions):
        ax_traj.text(pos[0], pos[1], pos[2], str(i+1), fontsize=8)
    
    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    ax_traj.set_zlabel('Z (m)')
    ax_traj.set_title('Robot Trajectory with Sample Points')
    
    # Set aspect ratio
    max_range = np.max([np.ptp(positions[:, 0]), np.ptp(positions[:, 1]), np.ptp(positions[:, 2])])
    mid_x = np.mean([np.min(positions[:, 0]), np.max(positions[:, 0])])
    mid_y = np.mean([np.min(positions[:, 1]), np.max(positions[:, 1])])
    mid_z = np.mean([np.min(positions[:, 2]), np.max(positions[:, 2])])
    ax_traj.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax_traj.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax_traj.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Create image grid
    image_axes = []
    for i in range(min(6, len(samples))):
        ax = fig.add_subplot(gs[i//2, i%2+2])
        img = cv2.imread(samples[i]['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"Sample {i+1}")
        ax.axis('off')
        image_axes.append(ax)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'combined_visualization.png')
    plt.savefig(output_path, dpi=200)
    print(f"Combined visualization saved to: {output_path}")
    plt.close()
    
    # Now create individual frames with trajectory and one image
    for i, sample in enumerate(samples):
        fig = plt.figure(figsize=(15, 7))
        gs = gridspec.GridSpec(1, 2)
        
        # Plot trajectory
        ax_traj = fig.add_subplot(gs[0, 0], projection='3d')
        ax_traj.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        
        # Highlight current position
        current_pos = sample['position']
        ax_traj.scatter(current_pos[0], current_pos[1], current_pos[2], c='r', s=100)
        
        ax_traj.set_xlabel('X (m)')
        ax_traj.set_ylabel('Y (m)')
        ax_traj.set_zlabel('Z (m)')
        ax_traj.set_title(f'Robot Position - Sample {i+1}')
        
        # Set consistent view
        ax_traj.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax_traj.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax_traj.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Plot image
        ax_img = fig.add_subplot(gs[0, 1])
        img = cv2.imread(sample['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax_img.imshow(img)
        ax_img.set_title(f"Camera View - Sample {i+1}")
        ax_img.axis('off')
        
        plt.tight_layout()
        frame_path = os.path.join(output_dir, f'frame_{i+1:03d}.png')
        plt.savefig(frame_path, dpi=150)
        plt.close()
    
    print(f"Generated {len(samples)} individual frames in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize collected navigation data")
    parser.add_argument("data_dir", help="Directory containing the synchronized data")
    parser.add_argument("--output", "-o", default="visualizations", help="Output directory for visualizations")
    parser.add_argument("--sample-rate", "-s", type=int, default=10, help="Sampling rate for image selection")
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        sys.exit(1)
    
    # Locate CSV and images directory
    csv_path = os.path.join(args.data_dir, "poses.csv")
    images_dir = os.path.join(args.data_dir, "images")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} does not exist")
        sys.exit(1)
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory {images_dir} does not exist")
        sys.exit(1)
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading data from {csv_path}...")
    data = load_data(csv_path, images_dir)
    print(f"Loaded {len(data)} data points")
    
    # Generate visualizations
    print("Generating trajectory plot...")
    plot_trajectory(data, os.path.join(output_dir, "trajectory_3d.png"))
    
    print("Generating GNSS plot...")
    plot_gnss_path(data, os.path.join(output_dir, "gnss_trajectory.png"))
    
    print("Generating combined visualizations...")
    combined_dir = os.path.join(output_dir, "combined")
    create_combined_visualization(data, combined_dir, args.sample_rate)
    
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()