#!/usr/bin/env python3
"""
ROS 2 Node: emotion_recognition_node (Bài tập thực hành 08 - Package rua4_face)
Nhận diện biểu cảm khuôn mặt thời gian thực bằng DeepFace (ONNX) & OpenCV,
publish kết quả lên /detected_emotion và /detected_emotion_detail để điều khiển TurtleBot 4.

06 biểu cảm mục tiêu & hành vi TurtleBot 4 tương ứng:
  - happy    (Vui vẻ)      -> Di chuyển quỹ đạo cong sang trái
  - sad      (Buồn)        -> Di chuyển quỹ đạo cong sang phải
  - neutral  (Bình thường) -> Dừng tại chỗ
  - angry    (Tức giận)    -> Lùi nhẹ về phía sau
  - surprise (Ngạc nhiên)  -> Xoay một vòng 360 độ tại chỗ
  - fear     (Sợ hãi)      -> Tiến nhẹ về phía trước
"""

import argparse
from collections import Counter, deque
from datetime import datetime
import json
import os
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

# Bắt buộc cấu hình backend ONNX trước khi import deepface
os.environ.setdefault('DEEPFACE_BACKEND_ENGINE', 'onnx')

import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    DEEPFACE_IMPORT_ERROR = ''
except Exception as exc:  # pragma: no cover - phụ thuộc môi trường runtime
    DeepFace = None
    DEEPFACE_AVAILABLE = False
    DEEPFACE_IMPORT_ERROR = str(exc)


VALID_EMOTIONS: Tuple[str, ...] = (
    'happy',
    'sad',
    'neutral',
    'angry',
    'surprise',
    'fear',
)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

# (Nhãn tiếng Anh, Nhãn tiếng Việt không dấu cho OpenCV Hershey font, Hành vi TurtleBot, Màu BGR)
EMOTION_INFO: Dict[str, Tuple[str, str, str, Tuple[int, int, int]]] = {
    'happy': (
        'Happy',
        'Vui ve',
        'Curved Left (Re trai quy dao cong)',
        (60, 220, 60),
    ),
    'sad': (
        'Sad',
        'Buon',
        'Curved Right (Re phai quy dao cong)',
        (255, 175, 55),
    ),
    'neutral': (
        'Neutral',
        'Binh thuong',
        'Stop in Place (Dung tai cho)',
        (210, 210, 210),
    ),
    'angry': (
        'Angry',
        'Tuc gian',
        'Gentle Backward (Lui nhe ve sau)',
        (45, 65, 255),
    ),
    'surprise': (
        'Surprise',
        'Ngac nhien',
        'Spin 360 deg (Xoay 1 vong tai cho)',
        (0, 215, 255),
    ),
    'fear': (
        'Fear',
        'So hai',
        'Gentle Forward (Tien nhe ve truoc)',
        (230, 105, 255),
    ),
    'none': (
        'None',
        'Khong co',
        'Standby / Stop (Cho khuon mat)',
        (145, 145, 145),
    ),
}

SHORT_LABELS: Dict[str, str] = {
    'happy': 'Happy',
    'sad': 'Sad',
    'neutral': 'Neutral',
    'angry': 'Angry',
    'surprise': 'Surprise',
    'fear': 'Fear',
}


def resolve_default_results_dir(explicit_dir: str = '') -> str:
    """Xác định thư mục lưu ảnh snapshot kết quả."""
    if explicit_dir and explicit_dir.strip():
        target = os.path.expanduser(explicit_dir.strip())
    elif os.path.isdir('/home/gdon/ur_ws/src/rua4_face/results'):
        target = '/home/gdon/ur_ws/src/rua4_face/results'
    else:
        target = os.path.expanduser('~/.cache/rua4_face/results')

    try:
        os.makedirs(target, exist_ok=True)
        return target
    except OSError:
        fallback = os.path.expanduser('~/.cache/rua4_face/results')
        try:
            os.makedirs(fallback, exist_ok=True)
            return fallback
        except OSError:
            return '/tmp'


class EmotionRecognitionNode(Node):
    """Node ROS 2 nhận diện biểu cảm khuôn mặt theo thời gian thực bằng DeepFace (ONNX) + OpenCV."""

    def __init__(self, cli_args: argparse.Namespace):
        super().__init__('emotion_recognition_node')

        dyn_desc = ParameterDescriptor(dynamic_typing=True)

        self.declare_parameter('camera_id', cli_args.camera_id, descriptor=dyn_desc)
        self.declare_parameter('source', cli_args.source, descriptor=dyn_desc)
        self.declare_parameter('emotion_topic', cli_args.emotion_topic, descriptor=dyn_desc)
        self.declare_parameter('detail_topic', cli_args.detail_topic, descriptor=dyn_desc)
        self.declare_parameter('status_topic', cli_args.status_topic, descriptor=dyn_desc)
        self.declare_parameter('detector_backend', cli_args.detector_backend, descriptor=dyn_desc)
        self.declare_parameter('min_confidence', cli_args.min_confidence, descriptor=dyn_desc)
        self.declare_parameter('window_size', cli_args.window_size, descriptor=dyn_desc)
        self.declare_parameter('infer_interval_sec', cli_args.infer_interval_sec, descriptor=dyn_desc)
        self.declare_parameter('show_window', cli_args.show_window, descriptor=dyn_desc)
        self.declare_parameter('results_dir', cli_args.results_dir, descriptor=dyn_desc)

        self.camera_id: int = self._parse_int(self.get_parameter('camera_id').value, 0)
        self.source: str = str(self.get_parameter('source').value or '').strip()
        self.emotion_topic: str = str(
            self.get_parameter('emotion_topic').value or '/detected_emotion'
        ).strip()
        self.detail_topic: str = str(
            self.get_parameter('detail_topic').value or '/detected_emotion_detail'
        ).strip()
        self.status_topic: str = str(
            self.get_parameter('status_topic').value or '/face_control_status'
        ).strip()
        self.detector_backend: str = str(
            self.get_parameter('detector_backend').value or 'opencv'
        ).strip()
        self.min_confidence: float = min(
            1.0, max(0.0, self._parse_float(self.get_parameter('min_confidence').value, 0.40))
        )
        self.window_size: int = max(
            1, self._parse_int(self.get_parameter('window_size').value, 5)
        )
        self.infer_interval_sec: float = max(
            0.02, self._parse_float(self.get_parameter('infer_interval_sec').value, 0.15)
        )
        self.show_window: bool = self._parse_bool(
            self.get_parameter('show_window').value, True
        )
        raw_results_dir = str(self.get_parameter('results_dir').value or '').strip()
        self.results_dir: str = resolve_default_results_dir(raw_results_dir)

        if not DEEPFACE_AVAILABLE:
            self.get_logger().error(
                "Không thể nạp thư viện 'deepface' "
                f"({DEEPFACE_IMPORT_ERROR}). "
                "Vui lòng cài đặt: DEEPFACE_BACKEND_ENGINE=onnx deepface==0.0.105 onnxruntime."
            )
            raise RuntimeError("Missing 'deepface' library")

        # Publisher /detected_emotion và /detected_emotion_detail
        self.emotion_pub = self.create_publisher(String, self.emotion_topic, 10)
        self.detail_pub = self.create_publisher(String, self.detail_topic, 10)

        # Subscriber /face_control_status từ emotion_control_node
        self.robot_status_text: str = 'IDLE (Cho bieu cam khuon mat)'
        self.status_sub = self.create_subscription(
            String,
            self.status_topic,
            self._status_callback,
            10,
        )

        # Thiết lập nguồn video (ảnh tĩnh, file video, hoặc webcam)
        self.source_mode: str = 'webcam'
        self.static_image: Optional[np.ndarray] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self._init_input_source()

        # Bộ đệm cửa sổ trượt chống nhiễu (Sliding Window Majority Voting)
        self._emotion_window: Deque[str] = deque(maxlen=self.window_size)
        self._conf_window: Deque[float] = deque(maxlen=self.window_size)

        # Trạng thái đồng bộ hóa giữa main loop và worker thread suy luận DeepFace
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._pending_frame: Optional[np.ndarray] = None

        self._result_lock = threading.Lock()
        self._latest_result: Dict[str, Any] = {
            'emotion': 'none',
            'raw_emotion': 'none',
            'confidence': 0.0,
            'scores': {emo: 0.0 for emo in VALID_EMOTIONS},
            'face_box': None,
            'infer_ms': 0.0,
        }

        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._inference_worker,
            name='deepface_onnx_worker',
            daemon=True,
        )
        self._worker_thread.start()

        # Thống kê FPS, logging & HUD
        self.last_frame_time: float = time.monotonic()
        self.fps_ema: float = 0.0
        self.last_logged_emotion: str = 'none'
        self.last_log_time: float = 0.0
        self.window_name: str = 'TurtleBot 4 - AI Facial Emotion Recognition (rua4_face)'
        self.toast_message: str = ''
        self.toast_until: float = 0.0
        self._shutdown_requested: bool = False

        # Timer ~30 Hz cho vòng lặp đọc ảnh, hiển thị HUD và publish topic
        self.timer = self.create_timer(0.033, self._process_frame)

        self.get_logger().info('=' * 70)
        self.get_logger().info('KHỞI CHẠY NODE NHẬN DIỆN BIỂU CẢM KHUÔN MẶT (emotion_recognition_node)')
        self.get_logger().info(f"  - DeepFace Engine     : {os.environ.get('DEEPFACE_BACKEND_ENGINE', 'onnx')}")
        self.get_logger().info(f'  - Detector Backend    : {self.detector_backend}')
        self.get_logger().info(f'  - Nguồn đầu vào       : {self.source if self.source else f"Webcam ID {self.camera_id}"}')
        self.get_logger().info(f'  - Emotion Topic       : {self.emotion_topic} (std_msgs/msg/String)')
        self.get_logger().info(f'  - Detail Topic        : {self.detail_topic} (std_msgs/msg/String JSON)')
        self.get_logger().info(f'  - Min Confidence      : {self.min_confidence:.2f}')
        self.get_logger().info(f'  - Sliding Window      : {self.window_size} inferences')
        self.get_logger().info(f'  - Infer Interval      : {self.infer_interval_sec:.2f} s')
        self.get_logger().info(f'  - Results Dir         : {self.results_dir}')
        self.get_logger().info('=' * 70)

    @staticmethod
    def _parse_bool(value: Any, default: bool = True) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            norm = value.strip().lower()
            if norm in ('true', '1', 'yes', 'on'):
                return True
            if norm in ('false', '0', 'no', 'off', ''):
                return False
        return default

    @staticmethod
    def _parse_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return default

    def _status_callback(self, msg: String) -> None:
        if msg.data:
            self.robot_status_text = msg.data.strip()

    def _init_input_source(self) -> None:
        """Khởi tạo nguồn hình ảnh từ --source (ảnh/video) hoặc webcam."""
        if not self.source:
            self.source_mode = 'webcam'
            self.cap = self._open_webcam(self.camera_id)
            return

        expanded = os.path.expanduser(self.source)
        ext = os.path.splitext(expanded)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            if not os.path.isfile(expanded):
                raise RuntimeError(f'Không tìm thấy file ảnh nguồn: {expanded}')
            img = cv2.imread(expanded)
            if img is None or img.size == 0:
                raise RuntimeError(f'Không thể đọc file ảnh nguồn: {expanded}')
            self.source_mode = 'image'
            self.static_image = img
            self.get_logger().info(
                f'Đã nạp ảnh tĩnh nguồn: {expanded} ({img.shape[1]}x{img.shape[0]}).'
            )
            return

        # File video hoặc stream URL
        if os.path.exists(expanded):
            cap_target: Any = expanded
        elif self.source.isdigit():
            self.source_mode = 'webcam'
            self.camera_id = int(self.source)
            self.cap = self._open_webcam(self.camera_id)
            return
        else:
            cap_target = self.source

        self.source_mode = 'video'
        cap = cv2.VideoCapture(cap_target)
        if cap is None or not cap.isOpened():
            raise RuntimeError(f'Không thể mở nguồn video: {self.source}')
        self.cap = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.get_logger().info(f'Đã mở nguồn video: {self.source} ({w}x{h}).')

    def _open_webcam(self, preferred_id: int) -> Optional[cv2.VideoCapture]:
        """Mở webcam ở 640x480@30fps, thử V4L2 rồi ANY từ preferred_id và 0..3."""
        candidate_ids = [preferred_id] + [i for i in range(4) if i != preferred_id]
        for cam_id in candidate_ids:
            for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
                cap = cv2.VideoCapture(cam_id, backend)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        self.camera_id = cam_id
                        self.get_logger().info(
                            f'Đã kết nối Webcam ID {cam_id} '
                            f'({test_frame.shape[1]}x{test_frame.shape[0]}).'
                        )
                        return cap
                    cap.release()

        self.get_logger().warn(
            'Không tìm thấy thiết bị webcam (/dev/video*). '
            'Kiểm tra kết nối webcam hoặc dùng --source <path> để thử nghiệm.'
        )
        return None

    def _read_next_frame(self) -> Optional[np.ndarray]:
        """Đọc khung hình tiếp theo từ ảnh tĩnh, video hoặc webcam."""
        if self.source_mode == 'image':
            return self.static_image.copy() if self.static_image is not None else None

        if self.source_mode == 'video':
            if self.cap is None or not self.cap.isOpened():
                return None
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Lặp lại video từ đầu khi phát hết file
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret or frame is None:
                self.get_logger().warn(
                    f'Không đọc được khung hình từ video {self.source}!',
                    throttle_duration_sec=3.0,
                )
                return None
            return frame

        # Webcam mode
        if self.cap is None or not self.cap.isOpened():
            self.cap = self._open_webcam(self.camera_id)
            if self.cap is None:
                return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn(
                'Không đọc được khung hình từ webcam!',
                throttle_duration_sec=3.0,
            )
            return None

        return cv2.flip(frame, 1)

    def _submit_frame_for_inference(self, frame_bgr: np.ndarray) -> None:
        """Cập nhật khung hình mới nhất cho luồng nền (chỉ giữ 1 frame, không xếp hàng)."""
        with self._frame_lock:
            self._pending_frame = frame_bgr.copy()
            self._frame_event.set()

    @staticmethod
    def _select_primary_face(analysis: Any) -> Optional[Dict[str, Any]]:
        """Chọn khuôn mặt lớn nhất nếu DeepFace trả về danh sách nhiều khuôn mặt."""
        if isinstance(analysis, dict):
            return analysis
        if isinstance(analysis, list) and analysis:
            valid_items = [item for item in analysis if isinstance(item, dict)]
            if not valid_items:
                return None

            def face_area(item: Dict[str, Any]) -> int:
                reg = item.get('region') or {}
                try:
                    return int(reg.get('w', 0)) * int(reg.get('h', 0))
                except (TypeError, ValueError):
                    return 0

            return max(valid_items, key=face_area)
        return None

    @staticmethod
    def _normalize_scores(raw_emotion_dict: Dict[str, Any]) -> Dict[str, float]:
        """Chuẩn hóa điểm xác suất cảm xúc từ 0..100 (hoặc 0..1) sang [0.0, 1.0]."""
        parsed: Dict[str, float] = {}
        for k, v in raw_emotion_dict.items():
            try:
                parsed[str(k).strip().lower()] = float(v)
            except (TypeError, ValueError):
                continue

        if parsed and (max(parsed.values()) > 1.0 or sum(parsed.values()) > 1.5):
            parsed = {k: v / 100.0 for k, v in parsed.items()}

        return {
            emo: min(1.0, max(0.0, float(parsed.get(emo, 0.0))))
            for emo in VALID_EMOTIONS
        }

    def _smooth_emotion(
        self, candidate_emotion: str, candidate_conf: float
    ) -> Tuple[str, float]:
        """Lọc nhiễu theo cửa sổ trượt (Sliding Window Majority Voting)."""
        self._emotion_window.append(candidate_emotion)
        self._conf_window.append(candidate_conf)

        counts = Counter(self._emotion_window)
        most_common_emo, freq = counts.most_common(1)[0]

        min_consensus = (
            1 if self.window_size == 1 else max(2, (len(self._emotion_window) + 1) // 2)
        )
        if most_common_emo != 'none' and freq >= min_consensus:
            matching_confs = [
                c
                for e, c in zip(self._emotion_window, self._conf_window)
                if e == most_common_emo
            ]
            avg_conf = sum(matching_confs) / max(1, len(matching_confs))
            return most_common_emo, avg_conf

        return 'none', 0.0

    def _analyze_bgr_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[str, float, Dict[str, float], Optional[Tuple[int, int, int, int]]]:
        """Chạy DeepFace.analyze trên ảnh BGR và lọc còn 06 biểu cảm hợp lệ."""
        empty_scores = {emo: 0.0 for emo in VALID_EMOTIONS}
        try:
            analysis = DeepFace.analyze(
                img_path=frame_bgr,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=True,
                silent=True,
            )
        except ValueError:
            # Không phát hiện thấy khuôn mặt trong khung hình
            return 'none', 0.0, empty_scores, None
        except Exception as exc:
            self.get_logger().warn(
                f'Lỗi suy luận DeepFace ({exc})',
                throttle_duration_sec=3.0,
            )
            return 'none', 0.0, empty_scores, None

        face_item = self._select_primary_face(analysis)
        if face_item is None:
            return 'none', 0.0, empty_scores, None

        raw_scores_dict = face_item.get('emotion') or {}
        scores_6 = self._normalize_scores(raw_scores_dict)

        # Lọc bỏ 'disgust': chọn biểu cảm cao nhất trong 06 lớp hợp lệ
        best_valid = max(VALID_EMOTIONS, key=lambda emo: scores_6[emo])
        best_conf = scores_6[best_valid]

        reg = face_item.get('region') or {}
        face_box: Optional[Tuple[int, int, int, int]] = None
        try:
            fx = int(reg.get('x', 0))
            fy = int(reg.get('y', 0))
            fw = int(reg.get('w', 0))
            fh = int(reg.get('h', 0))
            if fw > 0 and fh > 0:
                face_box = (fx, fy, fw, fh)
        except (TypeError, ValueError):
            face_box = None

        if best_conf < self.min_confidence:
            return 'none', best_conf, scores_6, face_box

        return best_valid, best_conf, scores_6, face_box

    def _inference_worker(self) -> None:
        """Luồng nền chạy DeepFace ONNX bất đồng bộ để vòng lặp hiển thị luôn mượt."""
        last_infer_start = 0.0
        while not self._stop_event.is_set():
            triggered = self._frame_event.wait(timeout=0.1)
            if self._stop_event.is_set():
                break
            if not triggered:
                continue

            now = time.monotonic()
            elapsed = now - last_infer_start
            if elapsed < self.infer_interval_sec:
                if self._stop_event.wait(self.infer_interval_sec - elapsed):
                    break

            with self._frame_lock:
                frame = self._pending_frame
                self._pending_frame = None
                self._frame_event.clear()

            if frame is None:
                continue

            last_infer_start = time.monotonic()
            raw_emo, raw_conf, scores_6, face_box = self._analyze_bgr_frame(frame)
            infer_ms = (time.monotonic() - last_infer_start) * 1000.0

            stable_emo, stable_conf = self._smooth_emotion(raw_emo, raw_conf)
            final_conf = stable_conf if stable_emo != 'none' else raw_conf

            with self._result_lock:
                self._latest_result = {
                    'emotion': stable_emo,
                    'raw_emotion': raw_emo,
                    'confidence': float(final_conf),
                    'scores': scores_6,
                    'face_box': face_box,
                    'infer_ms': float(infer_ms),
                }

    @staticmethod
    def _prepare_display_canvas(
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """Đảm bảo canvas hiển thị đủ lớn (>= 640x480) để vẽ HUD rõ ràng."""
        h, w = frame.shape[:2]
        scale = 1.0
        if w < 640 or h < 480:
            scale = max(640.0 / max(1, w), 480.0 / max(1, h))
        elif w > 1280 or h > 720:
            scale = min(1280.0 / w, 720.0 / h)

        if abs(scale - 1.0) > 1e-3:
            new_w = max(640, int(round(w * scale)))
            new_h = max(480, int(round(h * scale)))
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return resized, new_w / float(w), new_h / float(h)

        return frame.copy(), 1.0, 1.0

    def _draw_hud(
        self,
        frame: np.ndarray,
        state: Dict[str, Any],
        now: float,
    ) -> np.ndarray:
        """Vẽ bounding box khuôn mặt, nhãn EN/VI, biểu đồ 6 thanh xác suất và trạng thái robot."""
        canvas, sx, sy = self._prepare_display_canvas(frame)
        h, w = canvas.shape[:2]

        emotion: str = state['emotion']
        raw_emotion: str = state['raw_emotion']
        confidence: float = float(state['confidence'])
        scores: Dict[str, float] = state['scores']
        face_box: Optional[Tuple[int, int, int, int]] = state['face_box']

        en_label, vn_label, action_desc, accent_color = EMOTION_INFO.get(
            emotion, EMOTION_INFO['none']
        )

        # 1. Vẽ nền bán trong suốt cho Header, Footer và Panel biểu đồ 6 thanh xác suất
        overlay = canvas.copy()
        header_h = 88
        footer_h = 30
        cv2.rectangle(overlay, (0, 0), (w, header_h), (18, 20, 26), -1)
        cv2.rectangle(overlay, (0, h - footer_h), (w, h), (18, 20, 26), -1)

        panel_w = 218
        panel_h = 166
        panel_x1 = max(8, w - panel_w - 10)
        panel_y2 = h - footer_h - 10
        panel_y1 = max(header_h + 8, panel_y2 - panel_h)
        panel_x2 = panel_x1 + panel_w
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (18, 20, 26), -1)

        cv2.addWeighted(overlay, 0.76, canvas, 0.24, 0, canvas)
        cv2.rectangle(
            canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (80, 88, 102), 1, cv2.LINE_AA
        )

        # 2. Vẽ khung khuôn mặt + nhãn (vẽ sau overlay để không bị làm tối hay che khuất)
        if face_box is not None:
            fx, fy, fw, fh = face_box
            x1 = max(0, int(round(fx * sx)))
            y1 = max(0, int(round(fy * sy)))
            x2 = min(w - 1, int(round((fx + fw) * sx)))
            y2 = min(h - 1, int(round((fy + fh) * sy)))

            cv2.rectangle(canvas, (x1, y1), (x2, y2), accent_color, 2, cv2.LINE_AA)

            # Vẽ 4 góc nhấn cho khung nhận diện khuôn mặt
            corner = max(12, min(24, (x2 - x1) // 5, (y2 - y1) // 5))
            thick = 3
            cv2.line(canvas, (x1, y1), (x1 + corner, y1), accent_color, thick, cv2.LINE_AA)
            cv2.line(canvas, (x1, y1), (x1, y1 + corner), accent_color, thick, cv2.LINE_AA)
            cv2.line(canvas, (x2, y1), (x2 - corner, y1), accent_color, thick, cv2.LINE_AA)
            cv2.line(canvas, (x2, y1), (x2, y1 + corner), accent_color, thick, cv2.LINE_AA)
            cv2.line(canvas, (x1, y2), (x1 + corner, y2), accent_color, thick, cv2.LINE_AA)
            cv2.line(canvas, (x1, y2), (x1, y2 - corner), accent_color, thick, cv2.LINE_AA)
            cv2.line(canvas, (x2, y2), (x2 - corner, y2), accent_color, thick, cv2.LINE_AA)
            cv2.line(canvas, (x2, y2), (x2, y2 - corner), accent_color, thick, cv2.LINE_AA)

            if emotion != 'none':
                tag = f'{en_label} / {vn_label} ({confidence * 100:.1f}%)'
            elif raw_emotion != 'none':
                tag = f'Pending: {raw_emotion} ({confidence * 100:.1f}%)'
            else:
                tag = f'Face ({confidence * 100:.1f}%)'

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1)
            pill_w = tw + 14
            pill_x1 = min(max(4, x1), max(4, w - pill_w - 4))
            pill_x2 = min(w - 2, pill_x1 + pill_w)

            # Ưu tiên đặt nhãn phía trên box, nếu vướng header thì đặt phía dưới hoặc trong box
            if y1 - th - 14 >= header_h + 2:
                tag_bottom = y1 - 4
            elif y2 + th + 14 <= h - footer_h - 2:
                tag_bottom = y2 + th + 12
            else:
                tag_bottom = min(h - footer_h - 4, max(header_h + th + 12, y1 + th + 12))

            cv2.rectangle(
                canvas,
                (pill_x1, tag_bottom - th - 10),
                (pill_x2, tag_bottom + 2),
                accent_color,
                -1,
            )
            cv2.putText(
                canvas,
                tag,
                (pill_x1 + 7, tag_bottom - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )

        # 3. Nội dung Header trên cùng
        conf_pct = confidence * 100.0 if emotion != 'none' else 0.0
        header_title = (
            f'Emotion: {en_label} ({vn_label})  -  {conf_pct:.1f}%'
            if emotion != 'none'
            else 'Emotion: None (Dang cho bieu cam ro net)'
        )
        cv2.putText(
            canvas,
            header_title,
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            accent_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f'Action: {action_desc}  |  FPS: {self.fps_ema:.1f}',
            (14, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        robot_line = f'Robot: {self.robot_status_text}'
        if len(robot_line) > 74:
            robot_line = robot_line[:71] + '...'
        cv2.putText(
            canvas,
            robot_line,
            (14, 77),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (120, 245, 160),
            1,
            cv2.LINE_AA,
        )

        # 4. Biểu đồ 6 thanh xác suất cảm xúc (live 6-bar probability chart)
        cv2.putText(
            canvas,
            'PROBABILITIES (6 CLASSES)',
            (panel_x1 + 10, panel_y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (215, 225, 240),
            1,
            cv2.LINE_AA,
        )
        bar_x = panel_x1 + 82
        bar_max_w = 84
        bar_h = 11
        for idx, emo in enumerate(VALID_EMOTIONS):
            row_y = panel_y1 + 39 + idx * 21
            if row_y + 4 >= panel_y2:
                break
            prob = min(1.0, max(0.0, float(scores.get(emo, 0.0))))
            emo_color = EMOTION_INFO[emo][3]
            is_active = emo == emotion
            label_color = (255, 255, 255) if is_active else (185, 190, 200)
            prefix = '>' if is_active else ' '

            cv2.putText(
                canvas,
                f'{prefix} {SHORT_LABELS[emo]}',
                (panel_x1 + 8, row_y + 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                emo_color if is_active else label_color,
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                canvas,
                (bar_x, row_y),
                (bar_x + bar_max_w, row_y + bar_h),
                (52, 56, 66),
                -1,
            )
            fill_w = int(round(prob * bar_max_w))
            if fill_w > 0:
                cv2.rectangle(
                    canvas,
                    (bar_x, row_y),
                    (bar_x + fill_w, row_y + bar_h),
                    emo_color,
                    -1,
                )
            cv2.putText(
                canvas,
                f'{prob * 100:4.0f}%',
                (bar_x + bar_max_w + 6, row_y + 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                label_color,
                1,
                cv2.LINE_AA,
            )

        # 5. Footer hướng dẫn phím bấm & thông báo lưu ảnh
        footer_text = (
            f'Q/ESC: Quit  |  S: Save snapshot  |  Detector: {self.detector_backend} (ONNX)'
        )
        if now < self.toast_until and self.toast_message:
            footer_text = self.toast_message
        cv2.putText(
            canvas,
            footer_text,
            (12, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (120, 245, 160) if now < self.toast_until else (220, 225, 230),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def _save_snapshot(self, annotated_frame: np.ndarray, emotion: str, now: float) -> None:
        """Lưu khung hình đã chú thích HUD xuống thư mục results_dir."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        safe_emo = emotion if emotion in VALID_EMOTIONS else 'none'
        filename = f'emotion_{safe_emo}_{ts}.jpg'
        out_path = os.path.join(self.results_dir, filename)
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            ok = cv2.imwrite(out_path, annotated_frame)
            if ok:
                self.get_logger().info(f'Đã lưu ảnh kết quả: {out_path}')
                self.toast_message = f'Saved: {filename}'
                self.toast_until = now + 2.5
            else:
                self.get_logger().warn(f'Không thể ghi ảnh snapshot vào: {out_path}')
        except Exception as exc:
            self.get_logger().warn(f'Lỗi lưu ảnh snapshot ({exc})')

    def _process_frame(self) -> None:
        """Timer callback đọc frame, gửi sang worker thread, publish ROS 2 và vẽ HUD."""
        now = time.monotonic()
        dt = max(1e-4, now - self.last_frame_time)
        self.last_frame_time = now
        inst_fps = 1.0 / dt
        self.fps_ema = (
            inst_fps if self.fps_ema == 0.0 else (0.85 * self.fps_ema + 0.15 * inst_fps)
        )

        frame = self._read_next_frame()
        if frame is None:
            return

        # Gửi bản sao frame mới nhất sang luồng suy luận nền
        self._submit_frame_for_inference(frame)

        # Lấy kết quả suy luận mới nhất (không chặn vòng lặp hiển thị)
        with self._result_lock:
            state = {
                'emotion': self._latest_result['emotion'],
                'raw_emotion': self._latest_result['raw_emotion'],
                'confidence': self._latest_result['confidence'],
                'scores': dict(self._latest_result['scores']),
                'face_box': self._latest_result['face_box'],
                'infer_ms': self._latest_result['infer_ms'],
            }

        stable_emotion: str = state['emotion']
        stable_conf: float = float(state['confidence'])

        # Publish /detected_emotion (std_msgs/msg/String)
        emo_msg = String()
        emo_msg.data = stable_emotion
        self.emotion_pub.publish(emo_msg)

        # Publish /detected_emotion_detail (JSON std_msgs/msg/String)
        detail_payload = {
            'emotion': stable_emotion,
            'raw_emotion': state['raw_emotion'],
            'confidence': round(stable_conf, 4),
            'scores': {k: round(float(v), 4) for k, v in state['scores'].items()},
            'fps': round(float(self.fps_ema), 2),
        }
        detail_msg = String()
        detail_msg.data = json.dumps(detail_payload, ensure_ascii=False)
        self.detail_pub.publish(detail_msg)

        # In log ra Terminal khi có biểu cảm mới hoặc định kỳ mỗi 2.0s
        if stable_emotion != 'none' and (
            stable_emotion != self.last_logged_emotion or (now - self.last_log_time) >= 2.0
        ):
            en_label, vn_label, action_desc, _ = EMOTION_INFO.get(
                stable_emotion, EMOTION_INFO['none']
            )
            self.get_logger().info(
                f'[DETECTED] {stable_emotion} ({en_label} / {vn_label}) | '
                f'Confidence: {stable_conf * 100:.1f}% -> {action_desc}'
            )
            self.last_logged_emotion = stable_emotion
            self.last_log_time = now
        elif stable_emotion == 'none':
            self.last_logged_emotion = 'none'

        if self.show_window:
            annotated = self._draw_hud(frame, state, now)
            try:
                cv2.imshow(self.window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    self.get_logger().info(
                        'Người dùng nhấn Q/ESC -> Đóng node nhận diện biểu cảm.'
                    )
                    self._shutdown_requested = True
                    self.timer.cancel()
                    if rclpy.ok():
                        rclpy.shutdown()
                elif key in (ord('s'), ord('S')):
                    self._save_snapshot(annotated, stable_emotion, now)
            except Exception as exc:
                self.get_logger().warn(
                    f'Không thể mở cửa sổ cv2.imshow ({exc}) -> Chuyển sang chế độ ẩn (headless).'
                )
                self.show_window = False

    def destroy_node(self) -> None:
        self._stop_event.set()
        self._frame_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ROS 2 Node nhận diện biểu cảm khuôn mặt bằng DeepFace (ONNX) + OpenCV (rua4_face)'
    )
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='Chỉ số thiết bị webcam (mặc định: 0 -> /dev/video0)',
    )
    parser.add_argument(
        '--source',
        type=str,
        default='',
        help='Đường dẫn file ảnh hoặc video kiểm thử (để trống = dùng webcam)',
    )
    parser.add_argument(
        '--emotion-topic',
        type=str,
        default='/detected_emotion',
        help='Topic publish nhãn biểu cảm (mặc định: /detected_emotion)',
    )
    parser.add_argument(
        '--detail-topic',
        type=str,
        default='/detected_emotion_detail',
        help='Topic publish chi tiết JSON điểm số biểu cảm (mặc định: /detected_emotion_detail)',
    )
    parser.add_argument(
        '--status-topic',
        type=str,
        default='/face_control_status',
        help='Topic subscribe trạng thái robot từ emotion_control_node',
    )
    parser.add_argument(
        '--detector-backend',
        type=str,
        default='opencv',
        help='Backend phát hiện khuôn mặt của DeepFace (mặc định: opencv)',
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.40,
        help='Ngưỡng độ tin cậy tối thiểu [0.0 - 1.0] (mặc định: 0.40)',
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=5,
        help='Kích thước cửa sổ trượt bỏ phiếu đa số (mặc định: 5)',
    )
    parser.add_argument(
        '--infer-interval-sec',
        type=float,
        default=0.15,
        help='Chu kỳ tối thiểu giữa 2 lần suy luận DeepFace (giây, mặc định: 0.15)',
    )
    parser.add_argument(
        '--show-window',
        dest='show_window',
        action='store_true',
        default=True,
        help='Hiển thị cửa sổ OpenCV HUD (mặc định: bật)',
    )
    parser.add_argument(
        '--no-show-window',
        dest='show_window',
        action='store_false',
        help='Chạy ẩn không bật cửa sổ cv2.imshow',
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='',
        help='Thư mục lưu ảnh chụp HUD khi nhấn phím S',
    )

    cleaned_argv = rclpy.utilities.remove_ros_args(args=argv)
    return parser.parse_args(cleaned_argv[1:])


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    cli_args = parse_args(args)
    node = None
    try:
        node = EmotionRecognitionNode(cli_args)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as exc:
        print(f'[emotion_recognition_node] Dừng node: {exc}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
