# Package `rua4_face` — Nhận diện Biểu cảm Khuôn mặt & Điều khiển TurtleBot 4 (Bài thực hành 8)

Package **ROS 2 Humble** hoạt động độc lập cho robot **TurtleBot 4** trong môi trường mô phỏng **Gazebo Sim (Ignition)**, thực hiện đầy đủ các yêu cầu của **Bài thực hành 8: Nhận diện biểu cảm khuôn mặt và điều khiển TurtleBot**:

1. **Nhận diện biểu cảm khuôn mặt thời gian thực (`emotion_recognition_node`)**:
   - Đọc luồng hình ảnh từ webcam laptop/USB (`/dev/video0`, hoặc file ảnh/video qua `--source`) bằng **OpenCV** (`cv2.VideoCapture`).
   - Sử dụng **DeepFace** với backend **ONNX** (`DEEPFACE_BACKEND_ENGINE=onnx`) và bộ dò **YuNet** mặc định (`detector_backend:=yunet`), kết hợp **Focus Zone ROI** (`focus_center_only=true`, `min_face_height_ratio=0.16`, `min_face_confidence=0.65`) và theo dõi bám nét tâm khung hình để chỉ phân tích người ngồi trước camera và bỏ qua các khuôn mặt nhỏ ở nền (`BG (Ignored)`).
   - Lọc theo ngưỡng tin cậy (`min_confidence=0.40`), làm mượt bằng **Sliding Window Majority Voting** (`window_size=5`), hiển thị HUD OpenCV thời gian thực, publish `/detected_emotion` & `/detected_emotion_detail` (`std_msgs/msg/String`) và tự động cập nhật ảnh kết quả `results/latest_<emotion>.jpg` (`auto_save_results=true`, tắt bằng `--no-auto-save`) bên cạnh chụp thủ công bằng phím `S` (`results/emotion_<emotion>_<timestamp>.jpg`).
2. **Điều khiển hành vi TurtleBot 4 (`emotion_control_node`)**:
   - Subscribe `/detected_emotion`, `/odom` và `/scan`, phát lệnh vận tốc `geometry_msgs/msg/Twist` lên `/cmd_vel` ở tần số **20 Hz**.
   - Tích hợp debounce (`confirm_frames`), **Watchdog Timeout** (`1.5s`), bảo vệ chống va chạm LiDAR (`/scan`) và dừng an toàn khi tắt node.

---

## 📁 Cấu trúc Thư mục Package

```text
rua4_face/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/rua4_face
├── rua4_face/
│   ├── __init__.py
│   ├── emotion_recognition_node.py   # Node AI nhận diện 06 biểu cảm khuôn mặt (DeepFace ONNX + OpenCV)
│   └── emotion_control_node.py       # Node điều khiển TurtleBot 4 qua /cmd_vel theo biểu cảm
├── launch/
│   ├── sim.launch.py                 # Khởi chạy Ignition Gazebo + Spawn TurtleBot 4 (spawn_dock:=false)
│   ├── turtlebot4_spawn.launch.py
│   ├── lab8_face.launch.py           # Khởi chạy đồng thời emotion_recognition_node + emotion_control_node
│   └── include/
│       ├── control.launch.py
│       └── create3_nodes.launch.py
├── config/rviz/rua4_sim.rviz
└── results/
    ├── latest_happy.jpg              # Ảnh kết quả HUD biểu cảm Happy (Vui vẻ)
    ├── latest_sad.jpg                # Ảnh kết quả HUD biểu cảm Sad (Buồn)
    ├── latest_neutral.jpg            # Ảnh kết quả HUD biểu cảm Neutral (Bình thường)
    ├── latest_angry.jpg              # Ảnh kết quả HUD biểu cảm Angry (Tức giận)
    ├── latest_surprise.jpg           # Ảnh kết quả HUD biểu cảm Surprise (Ngạc nhiên)
    ├── latest_fear.jpg               # Ảnh kết quả HUD biểu cảm Fear (Sợ hãi)
    ├── lab8_gazebo_rviz.png          # Ảnh mô phỏng Gazebo Ignition + RViz2
    ├── lab8_gazebo_trajectories.png  # Đồ thị quỹ đạo kiểm thử 06 biểu cảm
    └── lab8_verification_report.txt  # Báo cáo số liệu kiểm thử định lượng Gazebo
```

---

## 🙂 Bảng Ánh xạ 06 Biểu cảm Khuôn mặt & Hành động TurtleBot 4

| STT | Biểu cảm | Nhãn `/detected_emotion` | Hành động TurtleBot 4 | Vận tốc `/cmd_vel` (`Twist`) |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **Happy (Vui vẻ)** | `happy` | Di chuyển theo quỹ đạo cong sang trái | `linear.x = +0.18 m/s`, `angular.z = +0.55 rad/s` |
| 2 | **Sad (Buồn)** | `sad` | Di chuyển theo quỹ đạo cong sang phải | `linear.x = +0.18 m/s`, `angular.z = -0.55 rad/s` |
| 3 | **Neutral (Bình thường)** | `neutral` | Dừng tại chỗ | `linear.x = 0.0`, `angular.z = 0.0` |
| 4 | **Angry (Tức giận)** | `angry` | Lùi nhẹ về phía sau | `linear.x = -0.10 m/s`, `angular.z = 0.0` |
| 5 | **Surprise (Ngạc nhiên)** | `surprise` | Xoay một vòng ($360^\circ$) tại chỗ rồi dừng | `linear.x = 0.0`, `angular.z = +1.0 rad/s` (theo `/odom`) |
| 6 | **Fear (Sợ hãi)** | `fear` | Tiến nhẹ về phía trước | `linear.x = +0.10 m/s`, `angular.z = 0.0` |

*(Các trạng thái ngoài 6 biểu cảm trên như `disgust`, `none`, không phát hiện khuôn mặt đều đưa robot về trạng thái dừng an toàn).*

---

## 🛠️ Build & Chạy Thử nghiệm (ROS 2 Humble)

### 1. Build Package
```bash
cd ~/ur_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select rua4_face
source install/setup.bash
```

### 2. Chạy Mô phỏng Gazebo & Hệ thống Nhận diện Biểu cảm
- **Terminal 1 — Khởi chạy Gazebo + TurtleBot 4**:
  ```bash
  cd ~/ur_ws
  source /opt/ros/humble/setup.bash && source install/setup.bash
  ros2 launch rua4_face sim.launch.py world:=warehouse
  ```
- **Terminal 2 — Khởi chạy Nhận diện Biểu cảm & Điều khiển TurtleBot 4**:
  ```bash
  cd ~/ur_ws
  source /opt/ros/humble/setup.bash && source install/setup.bash
  ros2 launch rua4_face lab8_face.launch.py
  ```
*(Hoặc khởi chạy gộp cả Gazebo và 2 node bằng 1 lệnh: `ros2 launch rua4_face lab8_face.launch.py launch_sim:=true`)*

### 3. Chạy Kiểm thử Riêng Từng Node & Lưu Kết quả Tự động
```bash
QT_QPA_PLATFORM=xcb GDK_BACKEND=x11 WAYLAND_DISPLAY="" ros2 run rua4_face emotion_recognition_node --camera-id 0
ros2 run rua4_face emotion_control_node
```

- **Tự động cập nhật ảnh kết quả (`results/`)**: Mặc định `auto_save_results=true` sẽ tự động ghi đè nguyên tử ảnh HUD mới nhất vào `results/latest_<emotion>.jpg` (chu kỳ tối thiểu `2.0s` hoặc ngay khi đổi sang biểu cảm hợp lệ mới). Có thể tắt bằng cờ `--no-auto-save` (`-p auto_save_results:=false`), hoặc nhấn phím **`S`** trên cửa sổ HUD để chụp thêm ảnh đánh dấu thời gian `results/emotion_<emotion>_<timestamp>.jpg`.
- **Hiển thị đầy đủ mô hình Robot & TF trên RViz2**: `launch/include/create3_nodes.launch.py` bổ sung node `joint_state_publisher` (gộp `/joint_states` từ `diffdrive_controller` và phát các khớp `wheel_drop_left_joint`, `wheel_drop_right_joint` cho `robot_state_publisher`), đồng thời `config/rviz/rua4_sim.rviz` đặt `Durability Policy: Transient Local` cho topic `/robot_description`.
- **Ghi chú hiển thị GUI trên GNOME Wayland / Distrobox**: Các file `sim.launch.py` và `lab8_face.launch.py` tự động thiết lập `QT_QPA_PLATFORM=xcb`, `GDK_BACKEND=x11`, `WAYLAND_DISPLAY=""` (và `DEEPFACE_BACKEND_ENGINE=onnx`) để cửa sổ Gazebo Ignition, RViz2 và OpenCV HUD mở ổn định qua XWayland.

