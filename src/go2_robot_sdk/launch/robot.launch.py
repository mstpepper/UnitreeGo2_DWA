# Copyright (c) 2024, RoboVerse community
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import FrontendLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    with_rviz2 = LaunchConfiguration('rviz2', default='true')
    with_nav2 = LaunchConfiguration('nav2', default='false')
    with_slam = LaunchConfiguration('slam', default='false')
    with_foxglove = LaunchConfiguration('foxglove', default='true')
    with_joystick = LaunchConfiguration('joystick', default='true')
    with_teleop = LaunchConfiguration('teleop', default='true')
    with_realsense = LaunchConfiguration('with_realsense', default='true')
    start_realsense = LaunchConfiguration('start_realsense', default='true')
    realsense_color_topic = LaunchConfiguration(
        'realsense_color_topic',
        default='/camera/camera/color/image_raw'
    )
    realsense_depth_topic = LaunchConfiguration(
        'realsense_depth_topic',
        default='/camera/camera/aligned_depth_to_color/image_raw'
    )
    realsense_camera_info_topic = LaunchConfiguration(
        'realsense_camera_info_topic',
        default='/camera/camera/color/camera_info'
    )
    enable_cyclonedds_motion = LaunchConfiguration(
        'enable_cyclonedds_motion',
        default='false'
    )

    robot_token = os.getenv('ROBOT_TOKEN', '') # how does this work for multiple robots?
    robot_ip = os.getenv('ROBOT_IP', '')
    robot_ip_lst = robot_ip.replace(" ", "").split(",")
    print("IP list:", robot_ip_lst)


    conn_type = os.getenv('CONN_TYPE', 'webrtc')

    conn_mode = "single" if len(robot_ip_lst) == 1 and conn_type != "cyclonedds" else "multi"

    if conn_mode == 'single':
        rviz_config = "single_robot_conf.rviz"
    else:
        rviz_config = "multi_robot_conf.rviz"

    if conn_type == 'cyclonedds':
        rviz_config = "cyclonedds_config.rviz"

    urdf_file_name = 'multi_go2.urdf'
    urdf = os.path.join(
        get_package_share_directory('go2_robot_sdk'),
        "urdf",
        urdf_file_name)
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    robot_desc_modified_lst = []

    for i in range(len(robot_ip_lst)):
        robot_desc_modified_lst.append(robot_desc.format(robot_num=f"robot{i}"))

    urdf_launch_nodes = []

    joy_params = os.path.join(
        get_package_share_directory('go2_robot_sdk'),
        'config', 'joystick.yaml'
    )

    default_config_topics = os.path.join(
        get_package_share_directory('go2_robot_sdk'),
        'config', 'twist_mux.yaml')

    foxglove_launch = os.path.join(
        get_package_share_directory('foxglove_bridge'),
        'launch',
        'foxglove_bridge_launch.xml',
    )

    slam_toolbox_config = os.path.join(
        get_package_share_directory('go2_robot_sdk'),
        'config',
        'mapper_params_online_async.yaml'
    )

    nav2_config = os.path.join(
        get_package_share_directory('go2_robot_sdk'),
        'config',
        'nav2_params.yaml'
    )

    if conn_mode == 'single':

        urdf_file_name = 'go2_with_realsense.urdf'
        urdf = os.path.join(
            get_package_share_directory('go2_robot_sdk'),
            "urdf",
            urdf_file_name)
        with open(urdf, 'r') as infp:
            robot_desc = infp.read()

        urdf_launch_nodes.append(
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time,
                             'robot_description': robot_desc}],
                arguments=[urdf]
            ),
        )
    else:

        for i in range(len(robot_ip_lst)):
            urdf_launch_nodes.append(
                Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    name='robot_state_publisher',
                    output='screen',
                    namespace=f"robot{i}",
                    parameters=[{'use_sim_time': use_sim_time,
                                 'robot_description': robot_desc_modified_lst[i]}],
                    arguments=[urdf]
                ),
            )
    return LaunchDescription([
        DeclareLaunchArgument(
            'with_realsense',
            default_value='true',
            description='Launch the COCO and BEV nodes for the attached D435 camera'
        ),
        DeclareLaunchArgument(
            'start_realsense',
            default_value='true',
            description='Start realsense2_camera here; leave false if it is already running'
        ),
        DeclareLaunchArgument(
            'realsense_color_topic',
            default_value='/camera/camera/color/image_raw'
        ),
        DeclareLaunchArgument(
            'realsense_depth_topic',
            default_value='/camera/camera/aligned_depth_to_color/image_raw'
        ),
        DeclareLaunchArgument(
            'realsense_camera_info_topic',
            default_value='/camera/camera/color/camera_info'
        ),
        DeclareLaunchArgument(
            'enable_cyclonedds_motion',
            default_value='false',
            description='Allow cmd_vel to reach the Go2 through CycloneDDS'
        ),

        *urdf_launch_nodes,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('realsense2_camera'),
                    'launch',
                    'rs_launch.py'
                ])
            ),
            condition=IfCondition(start_realsense),
            launch_arguments={
                'camera_namespace': 'camera',
                'camera_name': 'camera',
                'enable_color': 'true',
                'enable_depth': 'true',
                'align_depth.enable': 'true',
                'pointcloud.enable': 'false',
                'publish_tf': 'true',
            }.items(),
        ),
        Node(
            package='coco_detector',
            executable='coco_detector_node',
            name='coco_detector_node',
            condition=IfCondition(with_realsense),
            parameters=[{
                'image_topic': realsense_color_topic,
                'publish_annotated_image': False,
            }],
        ),
        Node(
            package='go2_robot_sdk',
            executable='bev_obstacle_node',
            name='bev_obstacle_node',
            condition=IfCondition(with_realsense),
            parameters=[{
                'detections_topic': '/detected_objects',
                'depth_topic': realsense_depth_topic,
                'camera_info_topic': realsense_camera_info_topic,
            }],
        ),
        Node(
            package='go2_robot_sdk',
            executable='go2_driver_node',
            parameters=[{
                'robot_ip': robot_ip,
                'token': robot_token,
                'conn_type': conn_type,
                'enable_cyclonedds_motion': enable_cyclonedds_motion,
            }],
        ),
        Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            condition=IfCondition(with_rviz2),
            name='rviz2',
            arguments=['-d' + os.path.join(get_package_share_directory('go2_robot_sdk'), 'config', rviz_config)]
        ),
        Node(
            package='joy',
            executable='joy_node',
            condition=IfCondition(with_joystick),
            parameters=[joy_params]
        ),
        Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='teleop_node',
            condition=IfCondition(with_joystick),
            parameters=[default_config_topics],
        ),
        Node(
            package='twist_mux',
            executable='twist_mux',
            output='screen',
            condition=IfCondition(with_teleop),
            parameters=[
                {'use_sim_time': use_sim_time},
                default_config_topics
            ],
        ),

        IncludeLaunchDescription(
            FrontendLaunchDescriptionSource(foxglove_launch),
            condition=IfCondition(with_foxglove),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory(
                    'slam_toolbox'), 'launch', 'online_async_launch.py')
            ]),
            condition=IfCondition(with_slam),
            launch_arguments={
                'slam_params_file': slam_toolbox_config,
                'use_sim_time': use_sim_time,
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory(
                    'nav2_bringup'), 'launch', 'navigation_launch.py')
            ]),
            condition=IfCondition(with_nav2),
            launch_arguments={
                'params_file': nav2_config,
                'use_sim_time': use_sim_time,
            }.items(),
        ),
    ])
