#!/usr/bin/env python3
"""
ROS 2 Node: emotion_control_node (Bài thực hành 8 - Package rua4_face)
Nhận nhãn biểu cảm khuôn mặt từ /detected_emotion và điều khiển TurtleBot 4 qua /cmd_vel.

Bảng ánh xạ 06 biểu cảm khuôn mặt -> Hành động TurtleBot 4:
  1. happy    (Vui vẻ)      : Di chuyển theo quỹ đạo cong sang trái  (+v, +w)
  2. sad      (Buồn)        : Di chuyển theo quỹ đạo cong sang phải  (+v, -w)
  3. neutral  (Bình thường) : Dừng tại chỗ                           (0.0, 0.0)
  4. angry    (Tức giận)    : Lùi nhẹ về phía sau                    (-v_gentle, 0.0)
  5. surprise (Ngạc nhiên)  : Xoay một vòng 360 độ tại chỗ rồi dừng  (0.0, +w_spin)
  6. fear     (Sợ hãi)      : Tiến nhẹ về phía trước                 (+v_gentle, 0.0)
  -  disgust / none / khác  : Dừng an toàn tại chỗ                   (0.0, 0.0)
"""

import argparse
import math
import time
from typing import Dict, List, Optional, Tuple

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


VALID_EMOTIONS = ('happy', 'sad', 'neutral', 'angry', 'surprise', 'fear')

EMOTION_ALIASES: Dict[str, str] = {
    'happy': 'happy',
    'vui': 'happy',
    'sad': 'sad',
    'buon': 'sad',
    'neutral': 'neutral',
    'binh thuong': 'neutral',
    'angry': 'angry',
    'tuc gian': 'angry',
    'surprise': 'surprise',
    'surprised': 'surprise',
    'ngac nhien': 'surprise',
    'fear': 'fear',
    'so hai': 'fear',
    'disgust': 'disgust',
    'none': 'none',
    'unknown': 'none',
    '': 'none',
}

EMOTION_ACTION_LABELS: Dict[str, str] = {
    'happy': 'HAPPY -> Quy dao cong sang trai',
    'sad': 'SAD -> Quy dao cong sang phai',
    'neutral': 'NEUTRAL -> Dung tai cho',
    'angry': 'ANGRY -> Lui nhe ve phia sau',
    'surprise': 'SURPRISE -> Xoay 1 vong 360 do tai cho',
    'fear': 'FEAR -> Tien nhe ve phia truoc',
    'disgust': 'DISGUST -> Dung an toan',
    'none': 'IDLE -> Cho tin hieu khuon mat',
}


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in ('true', '1', 'yes', 'on'):
            return True
        if norm in ('false', '0', 'no', 'off'):
            return False
    return default


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class EmotionControlNode(Node):
    """Node ROS 2 nhận biểu cảm khuôn mặt và phát lệnh Twist lên /cmd_vel ở 20 Hz."""

    SURPRISE_TARGET_RAD = (2.0 * math.pi) - 0.05

    def __init__(self, cli_args: argparse.Namespace):
        super().__init__('emotion_control_node')

        dyn_desc = ParameterDescriptor(dynamic_typing=True)

        self.declare_parameter('emotion_topic', cli_args.emotion_topic, descriptor=dyn_desc)
        self.declare_parameter('cmd_vel_topic', cli_args.cmd_vel_topic, descriptor=dyn_desc)
        self.declare_parameter('status_topic', cli_args.status_topic, descriptor=dyn_desc)
        self.declare_parameter('odom_topic', cli_args.odom_topic, descriptor=dyn_desc)
        self.declare_parameter('scan_topic', cli_args.scan_topic, descriptor=dyn_desc)
        self.declare_parameter('confirm_frames', cli_args.confirm_frames, descriptor=dyn_desc)
        self.declare_parameter(
            'watchdog_timeout_sec', cli_args.watchdog_timeout_sec, descriptor=dyn_desc
        )
        self.declare_parameter(
            'enable_lidar_safety', not cli_args.no_lidar_safety, descriptor=dyn_desc
        )
        self.declare_parameter(
            'safe_obstacle_dist', cli_args.safe_obstacle_dist, descriptor=dyn_desc
        )
        self.declare_parameter(
            'curve_linear_speed', cli_args.curve_linear_speed, descriptor=dyn_desc
        )
        self.declare_parameter(
            'curve_angular_speed', cli_args.curve_angular_speed, descriptor=dyn_desc
        )
        self.declare_parameter(
            'gentle_linear_speed', cli_args.gentle_linear_speed, descriptor=dyn_desc
        )
        self.declare_parameter(
            'spin_angular_speed', cli_args.spin_angular_speed, descriptor=dyn_desc
        )

        self.emotion_topic: str = str(self.get_parameter('emotion_topic').value)
        self.cmd_vel_topic: str = str(self.get_parameter('cmd_vel_topic').value)
        self.status_topic: str = str(self.get_parameter('status_topic').value)
        self.odom_topic: str = str(self.get_parameter('odom_topic').value)
        self.scan_topic: str = str(self.get_parameter('scan_topic').value)
        self.confirm_frames: int = max(
            1, _as_int(self.get_parameter('confirm_frames').value, 2)
        )
        self.watchdog_timeout_sec: float = max(
            0.2, _as_float(self.get_parameter('watchdog_timeout_sec').value, 1.5)
        )
        self.enable_lidar_safety: bool = _as_bool(
            self.get_parameter('enable_lidar_safety').value, True
        )
        self.safe_obstacle_dist: float = max(
            0.05, _as_float(self.get_parameter('safe_obstacle_dist').value, 0.28)
        )
        self.curve_linear_speed: float = _as_float(
            self.get_parameter('curve_linear_speed').value, 0.18
        )
        self.curve_angular_speed: float = _as_float(
            self.get_parameter('curve_angular_speed').value, 0.55
        )
        self.gentle_linear_speed: float = _as_float(
            self.get_parameter('gentle_linear_speed').value, 0.10
        )
        self.spin_angular_speed: float = _as_float(
            self.get_parameter('spin_angular_speed').value, 1.0
        )

        # Publishers: /cmd_vel (geometry_msgs/msg/Twist) & /face_control_status (std_msgs/msg/String)
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        # Subscribers: /detected_emotion, /odom, /scan
        self.emotion_sub = self.create_subscription(
            String,
            self.emotion_topic,
            self._emotion_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_callback,
            10,
        )
        scan_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            scan_qos,
        )

        # Trạng thái cảm biến & an toàn
        self.has_odom: bool = False
        self.current_odom_yaw: float = 0.0
        self.last_odom_time: float = 0.0
        self.front_min_dist: float = 99.0
        self.rear_min_dist: float = 99.0

        # Bộ lọc chống nhiễu & trạng thái biểu cảm hiện hành
        self.candidate_emotion: str = 'none'
        self.candidate_count: int = 0
        self.active_emotion: str = 'none'
        self.has_received_emotion: bool = False
        self.last_message_time: float = time.monotonic()
        self.watchdog_tripped: bool = False

        # Trạng thái xoay 1 vòng 360 độ cho biểu cảm 'surprise'
        self.surprise_turning: bool = False
        self.surprise_completed: bool = False
        self.surprise_accumulated_yaw: float = 0.0
        self.surprise_last_yaw: Optional[float] = None
        self.surprise_start_time: float = 0.0
        self.last_surprise_progress_deg: int = -1

        self.last_published_status: str = ''

        # Timer 20 Hz phát lệnh Twist liên tục lên /cmd_vel
        self.control_timer = self.create_timer(0.05, self._control_loop)

        self.get_logger().info('=' * 72)
        self.get_logger().info('KHỞI CHẠY NODE ĐIỀU KHIỂN TURTLEBOT 4 THEO BIỂU CẢM (emotion_control_node)')
        self.get_logger().info(f'  - Subscribe Emotion  : {self.emotion_topic} (std_msgs/msg/String)')
        self.get_logger().info(f'  - Subscribe Odom     : {self.odom_topic} (nav_msgs/msg/Odometry)')
        self.get_logger().info(f'  - Subscribe LiDAR    : {self.scan_topic} (Safety={self.enable_lidar_safety}, d={self.safe_obstacle_dist:.2f}m)')
        self.get_logger().info(f'  - Publish Cmd Vel    : {self.cmd_vel_topic} (geometry_msgs/msg/Twist @ 20 Hz)')
        self.get_logger().info(f'  - Confirm Frames     : {self.confirm_frames} | Watchdog: {self.watchdog_timeout_sec:.2f}s')
        self.get_logger().info(
            f'  - Speeds             : curve=({self.curve_linear_speed:.2f} m/s, {self.curve_angular_speed:.2f} rad/s), '
            f'gentle={self.gentle_linear_speed:.2f} m/s, spin={self.spin_angular_speed:.2f} rad/s'
        )
        self.get_logger().info('=' * 72)
        self._publish_status('IDLE -> Cho tin hieu khuon mat', force=True)

    @staticmethod
    def _normalize_emotion(raw_str: str) -> str:
        if not raw_str:
            return 'none'
        token = raw_str.strip().split('|')[0].split('(')[0].strip().lower()
        return EMOTION_ALIASES.get(token, 'none')

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _publish_status(self, text: str, force: bool = False) -> None:
        if not force and text == self.last_published_status:
            return
        self.last_published_status = text
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def _odom_callback(self, msg: Odometry) -> None:
        yaw = self._yaw_from_quaternion(msg.pose.pose.orientation)
        self.current_odom_yaw = yaw
        self.has_odom = True
        self.last_odom_time = time.monotonic()

        if self.active_emotion == 'surprise' and self.surprise_turning and not self.surprise_completed:
            if self.surprise_last_yaw is None:
                self.surprise_last_yaw = yaw
                return
            dyaw = math.atan2(
                math.sin(yaw - self.surprise_last_yaw),
                math.cos(yaw - self.surprise_last_yaw),
            )
            spin_sign = 1.0 if self.spin_angular_speed >= 0.0 else -1.0
            self.surprise_accumulated_yaw += dyaw * spin_sign
            self.surprise_last_yaw = yaw

    def _scan_callback(self, msg: LaserScan) -> None:
        if not msg.ranges:
            return
        front_vals: List[float] = []
        rear_vals: List[float] = []
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r < msg.range_min or r > msg.range_max:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if abs(angle) <= math.radians(25.0):
                front_vals.append(r)
            elif abs(angle) >= math.radians(155.0):
                rear_vals.append(r)

        self.front_min_dist = min(front_vals) if front_vals else 99.0
        self.rear_min_dist = min(rear_vals) if rear_vals else 99.0

    def _emotion_callback(self, msg: String) -> None:
        now = time.monotonic()
        self.has_received_emotion = True
        self.last_message_time = now
        self.watchdog_tripped = False

        emotion = self._normalize_emotion(msg.data)
        if emotion == self.candidate_emotion:
            self.candidate_count += 1
        else:
            self.candidate_emotion = emotion
            self.candidate_count = 1

        if self.candidate_count < self.confirm_frames:
            return

        if emotion == self.active_emotion:
            return

        prev_emotion = self.active_emotion
        self.active_emotion = emotion

        if emotion == 'surprise' and prev_emotion != 'surprise':
            self.surprise_turning = True
            self.surprise_completed = False
            self.surprise_accumulated_yaw = 0.0
            self.surprise_last_yaw = self.current_odom_yaw if self.has_odom else None
            self.surprise_start_time = now
            self.last_surprise_progress_deg = 0
            self.get_logger().info(
                '[BIỂU CẢM: SURPRISE] Bắt đầu xoay 1 vòng 360 độ tại chỗ...'
            )
            self._publish_status('SURPRISE -> Dang xoay 360 do (0/360 deg)', force=True)
        else:
            # Rời khỏi 'surprise' -> tái kích hoạt (re-arm) cho lần surprise tiếp theo
            self.surprise_turning = False
            self.surprise_completed = False
            self.surprise_accumulated_yaw = 0.0
            self.surprise_last_yaw = None
            self.last_surprise_progress_deg = -1

            action_label = EMOTION_ACTION_LABELS.get(emotion, EMOTION_ACTION_LABELS['none'])
            self.get_logger().info(f'[BIỂU CẢM MỚI] {action_label}')
            self._publish_status(action_label, force=True)

    def _publish_cmd_vel(self, linear_x: float = 0.0, angular_z: float = 0.0) -> Tuple[bool, str]:
        """Phát Twist lên /cmd_vel kèm kiểm tra LiDAR an toàn trước/sau."""
        blocked_reason = ''
        if self.enable_lidar_safety:
            if linear_x > 0.0 and self.front_min_dist < self.safe_obstacle_dist:
                self.get_logger().warn(
                    f'[AN TOÀN LIDAR] Vật cản phía trước {self.front_min_dist:.2f}m '
                    f'(< {self.safe_obstacle_dist:.2f}m) -> Chặn tiến!',
                    throttle_duration_sec=1.5,
                )
                linear_x = 0.0
                blocked_reason = f'SAFETY LIDAR (Truoc {self.front_min_dist:.2f}m)'
            elif linear_x < 0.0 and self.rear_min_dist < self.safe_obstacle_dist:
                self.get_logger().warn(
                    f'[AN TOÀN LIDAR] Vật cản phía sau {self.rear_min_dist:.2f}m '
                    f'(< {self.safe_obstacle_dist:.2f}m) -> Chặn lùi!',
                    throttle_duration_sec=1.5,
                )
                linear_x = 0.0
                blocked_reason = f'SAFETY LIDAR (Sau {self.rear_min_dist:.2f}m)'

        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)
        return bool(blocked_reason), blocked_reason

    def _stop_robot_immediately(self) -> None:
        stop_twist = Twist()
        for _ in range(3):
            self.cmd_vel_pub.publish(stop_twist)

    def _compute_motion_for_emotion(self, now: float) -> Tuple[float, float, str]:
        emotion = self.active_emotion

        if emotion == 'happy':
            return +self.curve_linear_speed, +self.curve_angular_speed, EMOTION_ACTION_LABELS['happy']

        if emotion == 'sad':
            return +self.curve_linear_speed, -self.curve_angular_speed, EMOTION_ACTION_LABELS['sad']

        if emotion == 'neutral':
            return 0.0, 0.0, EMOTION_ACTION_LABELS['neutral']

        if emotion == 'angry':
            return -self.gentle_linear_speed, 0.0, EMOTION_ACTION_LABELS['angry']

        if emotion == 'fear':
            return +self.gentle_linear_speed, 0.0, EMOTION_ACTION_LABELS['fear']

        if emotion == 'surprise':
            if self.surprise_completed:
                return 0.0, 0.0, 'SURPRISE -> Hoan tat xoay 360 do (Stopped)'

            odom_fresh = (
                self.has_odom
                and self.surprise_last_yaw is not None
                and (now - self.last_odom_time) < 1.0
            )
            if odom_fresh:
                rotated_rad = self.surprise_accumulated_yaw
                done = rotated_rad >= self.SURPRISE_TARGET_RAD
                progress_deg = max(0, min(360, int(round(math.degrees(rotated_rad)))))
            else:
                expected_dur = (2.0 * math.pi) / max(0.05, abs(self.spin_angular_speed))
                elapsed = max(0.0, now - self.surprise_start_time)
                done = elapsed >= expected_dur
                progress_deg = max(0, min(360, int(round((elapsed / expected_dur) * 360.0))))

            if done:
                self.surprise_turning = False
                self.surprise_completed = True
                self.get_logger().info(
                    '[SURPRISE HOÀN TẤT] Đã xoay đủ 1 vòng 360 độ tại chỗ -> Dừng robot.'
                )
                return 0.0, 0.0, 'SURPRISE -> Hoan tat xoay 360 do (Stopped)'

            bucket_deg = (progress_deg // 45) * 45
            if bucket_deg != self.last_surprise_progress_deg:
                self.last_surprise_progress_deg = bucket_deg
                return 0.0, +self.spin_angular_speed, f'SURPRISE -> Dang xoay 360 do ({progress_deg}/360 deg)'
            return 0.0, +self.spin_angular_speed, self.last_published_status

        # disgust / none / unknown
        return 0.0, 0.0, EMOTION_ACTION_LABELS.get(emotion, EMOTION_ACTION_LABELS['none'])

    def _control_loop(self) -> None:
        now = time.monotonic()

        if self.has_received_emotion and (now - self.last_message_time) > self.watchdog_timeout_sec:
            if not self.watchdog_tripped or self.active_emotion != 'none':
                silence = now - self.last_message_time
                self.get_logger().warn(
                    f'[WATCHDOG] Mất tín hiệu trên {self.emotion_topic} trong {silence:.2f}s '
                    f'(> {self.watchdog_timeout_sec:.2f}s) -> Dừng robot an toàn!',
                    throttle_duration_sec=2.0,
                )
                self.watchdog_tripped = True
                self.active_emotion = 'none'
                self.candidate_emotion = 'none'
                self.candidate_count = 0
                self.surprise_turning = False
                self.surprise_completed = False
                self.surprise_accumulated_yaw = 0.0
                self.surprise_last_yaw = None
                self._stop_robot_immediately()
                self._publish_status('WATCHDOG_STOP -> Mat tin hieu bieu cam', force=True)
            else:
                self._publish_cmd_vel(0.0, 0.0)
            return

        linear_x, angular_z, status_text = self._compute_motion_for_emotion(now)
        blocked, blocked_reason = self._publish_cmd_vel(linear_x, angular_z)
        if blocked:
            self._publish_status(blocked_reason)
        elif status_text:
            self._publish_status(status_text)

    def destroy_node(self) -> None:
        try:
            self._stop_robot_immediately()
        except Exception:
            pass
        super().destroy_node()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ROS 2 Node điều khiển TurtleBot 4 theo biểu cảm khuôn mặt (rua4_face)'
    )
    parser.add_argument(
        '--emotion-topic',
        type=str,
        default='/detected_emotion',
        help='Topic nhận nhãn biểu cảm (mặc định: /detected_emotion)',
    )
    parser.add_argument(
        '--cmd-vel-topic',
        type=str,
        default='/cmd_vel',
        help='Topic phát lệnh Twist (mặc định: /cmd_vel)',
    )
    parser.add_argument(
        '--status-topic',
        type=str,
        default='/face_control_status',
        help='Topic phát trạng thái điều khiển (mặc định: /face_control_status)',
    )
    parser.add_argument(
        '--odom-topic',
        type=str,
        default='/odom',
        help='Topic Odometry để đo góc xoay 360 độ (mặc định: /odom)',
    )
    parser.add_argument(
        '--scan-topic',
        type=str,
        default='/scan',
        help='Topic LaserScan chống va chạm (mặc định: /scan)',
    )
    parser.add_argument(
        '--confirm-frames',
        type=int,
        default=2,
        help='Số bản tin biểu cảm liên tiếp để xác nhận chuyển trạng thái (mặc định: 2)',
    )
    parser.add_argument(
        '--watchdog-timeout-sec',
        type=float,
        default=1.5,
        help='Thời gian chờ tối đa trước khi dừng an toàn nếu mất tín hiệu (mặc định: 1.5s)',
    )
    parser.add_argument(
        '--no-lidar-safety',
        action='store_true',
        help='Tắt kiểm tra vật cản LiDAR trên /scan',
    )
    parser.add_argument(
        '--safe-obstacle-dist',
        type=float,
        default=0.28,
        help='Khoảng cách an toàn tối thiểu (mặc định: 0.28m)',
    )
    parser.add_argument(
        '--curve-linear-speed',
        type=float,
        default=0.18,
        help='Vận tốc tiến khi đi quỹ đạo cong Happy/Sad (mặc định: 0.18 m/s)',
    )
    parser.add_argument(
        '--curve-angular-speed',
        type=float,
        default=0.55,
        help='Vận tốc góc khi đi quỹ đạo cong Happy/Sad (mặc định: 0.55 rad/s)',
    )
    parser.add_argument(
        '--gentle-linear-speed',
        type=float,
        default=0.10,
        help='Vận tốc tiến/lùi nhẹ cho Fear/Angry (mặc định: 0.10 m/s)',
    )
    parser.add_argument(
        '--spin-angular-speed',
        type=float,
        default=1.0,
        help='Vận tốc góc xoay 360 độ tại chỗ cho Surprise (mặc định: 1.0 rad/s)',
    )
    cleaned_argv = rclpy.utilities.remove_ros_args(args=argv)
    return parser.parse_args(cleaned_argv[1:])


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    cli_args = parse_args(args)
    node = None
    try:
        node = EmotionControlNode(cli_args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
