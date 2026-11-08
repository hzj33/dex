import os
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command, FindExecutable
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('dexh13_right_description')
    default_model_path = os.path.join(pkg_share, 'urdf/dexh13_right.urdf')
    default_rviz_config_path = os.path.join(pkg_share, 'config/urdf.rviz')
    
    gui_arg = DeclareLaunchArgument(name='gui', default_value='true', 
                                    description='Flag to enable joint_state_publisher_gui')
    model_arg = DeclareLaunchArgument(name='model', default_value=default_model_path,
                                      description='Absolute path to robot urdf file')
    rviz_arg = DeclareLaunchArgument(name='rvizconfig', default_value=default_rviz_config_path,
                                     description='Absolute path to rviz config file')
    
    # 使用 xacro 处理 URDF 文件并明确指定为字符串类型
    robot_description_content = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', LaunchConfiguration('model')]),
        value_type=str
    )
    
    # 设置 robot_description 参数
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content}]
    )
    
    # 确保 joint_state_publisher 和 joint_state_publisher_gui 使用相同的 robot_description 参数
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        condition=UnlessCondition(LaunchConfiguration('gui')),
        parameters=[{'robot_description': robot_description_content}]
    )
    
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(LaunchConfiguration('gui')),
        parameters=[{'robot_description': robot_description_content}]
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
        parameters=[{'robot_description': robot_description_content}]
    )
    
    return LaunchDescription([
        gui_arg,
        model_arg,
        rviz_arg,
        robot_state_publisher_node,
        joint_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node
    ])
