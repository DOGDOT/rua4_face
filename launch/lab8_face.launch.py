from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _build_nodes():
    """Tạo cặp node nhận diện biểu cảm khuôn mặt và điều khiển TurtleBot 4."""
    emotion_recognition_node = Node(
        package='rua4_face',
        executable='emotion_recognition_node',
        name='emotion_recognition_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'camera_id': ParameterValue(LaunchConfiguration('camera_id'), value_type=int),
            'source': ParameterValue(LaunchConfiguration('source'), value_type=str),
            'emotion_topic': ParameterValue(LaunchConfiguration('emotion_topic'), value_type=str),
            'detail_topic': ParameterValue(LaunchConfiguration('detail_topic'), value_type=str),
            'status_topic': ParameterValue(LaunchConfiguration('status_topic'), value_type=str),
            'detector_backend': ParameterValue(LaunchConfiguration('detector_backend'), value_type=str),
            'min_confidence': ParameterValue(LaunchConfiguration('min_confidence'), value_type=float),
            'window_size': ParameterValue(LaunchConfiguration('window_size'), value_type=int),
            'show_window': ParameterValue(LaunchConfiguration('show_window'), value_type=bool),
        }],
        additional_env={
            'QT_QPA_PLATFORM': 'xcb',
            'GDK_BACKEND': 'x11',
            'WAYLAND_DISPLAY': '',
            'DEEPFACE_BACKEND_ENGINE': 'onnx',
        },
    )

    emotion_control_node = Node(
        package='rua4_face',
        executable='emotion_control_node',
        name='emotion_control_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'emotion_topic': ParameterValue(LaunchConfiguration('emotion_topic'), value_type=str),
            'cmd_vel_topic': ParameterValue(LaunchConfiguration('cmd_vel_topic'), value_type=str),
            'status_topic': ParameterValue(LaunchConfiguration('status_topic'), value_type=str),
            'enable_lidar_safety': ParameterValue(LaunchConfiguration('enable_lidar_safety'), value_type=bool),
        }],
    )
    return [emotion_recognition_node, emotion_control_node]


def _launch_setup(context, *args, **kwargs):
    pkg_self = get_package_share_directory('rua4_face')
    launch_sim_str = LaunchConfiguration('launch_sim').perform(context).strip().lower()
    use_sim = launch_sim_str in ('true', '1', 'yes')
    rviz_str = LaunchConfiguration('rviz').perform(context).strip().lower()
    use_rviz = rviz_str in ('true', '1', 'yes')

    actions = []
    if use_sim:
        sim_launch = GroupAction(
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        PathJoinSubstitution([pkg_self, 'launch', 'sim.launch.py'])
                    ),
                    launch_arguments={
                        'world': LaunchConfiguration('world'),
                        'rviz': LaunchConfiguration('rviz'),
                        'use_sim_time': 'true',
                    }.items(),
                )
            ],
            scoped=True,
        )
        actions.append(sim_launch)
        actions.append(TimerAction(period=6.0, actions=_build_nodes()))
    else:
        if use_rviz:
            rviz_config = PathJoinSubstitution([pkg_self, 'config', 'rviz', 'rua4_sim.rviz'])
            actions.append(
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', rviz_config],
                    parameters=[{'use_sim_time': True}],
                    additional_env={
                        'QT_QPA_PLATFORM': 'xcb',
                        'GDK_BACKEND': 'x11',
                        'WAYLAND_DISPLAY': '',
                    },
                    output='screen',
                )
            )
        actions.extend(_build_nodes())

    return actions


def generate_launch_description():
    """
    Launch file chuyên biệt cho Bài thực hành 08 (lab8_face.launch.py).
    Khởi chạy đồng thời:
      1. emotion_recognition_node (DeepFace ONNX + OpenCV đọc webcam/nguồn ảnh, publish /detected_emotion)
      2. emotion_control_node (Subscribe /detected_emotion, điều khiển TurtleBot 4 qua /cmd_vel)
    Tùy chọn launch_sim:=true để khởi chạy kèm Gazebo Ignition + TurtleBot 4.
    """
    rmw_env = SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    qt_qpa_env = SetEnvironmentVariable('QT_QPA_PLATFORM', 'xcb')
    gdk_env = SetEnvironmentVariable('GDK_BACKEND', 'x11')
    wayland_env = SetEnvironmentVariable('WAYLAND_DISPLAY', '')

    launch_sim_arg = DeclareLaunchArgument(
        'launch_sim',
        default_value='false',
        choices=['true', 'false'],
        description='Tự động khởi chạy mô phỏng Gazebo + TurtleBot 4 nếu chưa bật',
    )
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='warehouse',
        description='Môi trường Gazebo khi launch_sim=true (warehouse, maze, depot)',
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        choices=['true', 'false'],
        description='Khởi chạy RViz2 trực quan hóa robot & cảm biến',
    )
    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='Chỉ số thiết bị webcam laptop/USB (mặc định: 0 -> /dev/video0)',
    )
    source_arg = DeclareLaunchArgument(
        'source',
        default_value='',
        description='Đường dẫn ảnh/video đầu vào thay cho webcam (để trống = dùng webcam)',
    )
    detector_backend_arg = DeclareLaunchArgument(
        'detector_backend',
        default_value='yunet',
        description='Backend dò khuôn mặt của DeepFace (mặc định: yunet)',
    )
    min_confidence_arg = DeclareLaunchArgument(
        'min_confidence',
        default_value='0.40',
        description='Ngưỡng độ tin cậy tối thiểu để chấp nhận biểu cảm [0.0 - 1.0]',
    )
    window_size_arg = DeclareLaunchArgument(
        'window_size',
        default_value='5',
        description='Kích thước cửa sổ trượt lọc nhiễu biểu cảm (số khung hình)',
    )
    show_window_arg = DeclareLaunchArgument(
        'show_window',
        default_value='true',
        choices=['true', 'false'],
        description='Hiển thị cửa sổ webcam OpenCV theo thời gian thực',
    )
    enable_lidar_safety_arg = DeclareLaunchArgument(
        'enable_lidar_safety',
        default_value='true',
        choices=['true', 'false'],
        description='Bật/tắt chế độ LiDAR chặn va chạm trước/sau',
    )
    emotion_topic_arg = DeclareLaunchArgument(
        'emotion_topic',
        default_value='/detected_emotion',
        description='Topic truyền nhãn biểu cảm khuôn mặt (std_msgs/msg/String)',
    )
    detail_topic_arg = DeclareLaunchArgument(
        'detail_topic',
        default_value='/detected_emotion_detail',
        description='Topic truyền chi tiết xác suất biểu cảm JSON (std_msgs/msg/String)',
    )
    cmd_vel_topic_arg = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value='/cmd_vel',
        description='Topic điều khiển vận tốc TurtleBot 4 (geometry_msgs/msg/Twist)',
    )
    status_topic_arg = DeclareLaunchArgument(
        'status_topic',
        default_value='/face_control_status',
        description='Topic phản hồi trạng thái điều khiển robot (std_msgs/msg/String)',
    )

    ld = LaunchDescription()
    ld.add_action(rmw_env)
    ld.add_action(qt_qpa_env)
    ld.add_action(gdk_env)
    ld.add_action(wayland_env)
    ld.add_action(launch_sim_arg)
    ld.add_action(world_arg)
    ld.add_action(rviz_arg)
    ld.add_action(camera_id_arg)
    ld.add_action(source_arg)
    ld.add_action(detector_backend_arg)
    ld.add_action(min_confidence_arg)
    ld.add_action(window_size_arg)
    ld.add_action(show_window_arg)
    ld.add_action(enable_lidar_safety_arg)
    ld.add_action(emotion_topic_arg)
    ld.add_action(detail_topic_arg)
    ld.add_action(cmd_vel_topic_arg)
    ld.add_action(status_topic_arg)
    ld.add_action(OpaqueFunction(function=_launch_setup))
    return ld
