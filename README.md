# Package `rua4_face` — Nhận diện Biểu cảm Khuôn mặt & Điều khiển TurtleBot 4 (Bài thực hành 8)

Package **ROS 2 Humble** hoạt động độc lập cho robot **TurtleBot 4** trong môi trường mô phỏng **Gazebo Sim (Ignition)**, thực hiện đầy đủ các yêu cầu của **Bài thực hành 8: Nhận diện biểu cảm khuôn mặt và điều khiển TurtleBot**:

1. **Nhận diện biểu cảm khuôn mặt thời gian thực (`emotion_recognition_node`)**:
   - Đọc luồng hình ảnh từ webcam laptop/USB (`/dev/video0`, hoặc file ảnh/video qua `--source`) bằng **OpenCV** (`cv2.VideoCapture`).
   - Sử dụng **DeepFace** với backend **ONNX** (`DEEPFACE_BACKEND_ENGINE=onnx`) phân loại biểu cảm khuôn mặt, lọc theo ngưỡng tin cậy và làm mượt bằng **Sliding Window Majority Voting**.
   - Hiển thị trực quan khung hình camera thời gian thực (khung khuôn mặt, nhãn biểu cảm, biểu đồ xác suất 6 trạng thái, FPS, trạng thái robot) và publish lên `/detected_emotion` (`std_msgs/msg/String`) và `/detected_emotion_detail` (`std_msgs/msg/String`).
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

### 3. Chạy Kiểm thử Riêng Từng Node
```bash
QT_QPA_PLATFORM=xcb GDK_BACKEND=x11 WAYLAND_DISPLAY="" ros2 run rua4_face emotion_recognition_node --camera-id 0
ros2 run rua4_face emotion_control_node
```

> **Ghi chú hiển thị GUI trên GNOME Wayland / Distrobox**: Các file `sim.launch.py` và `lab8_face.launch.py` đã tự động thiết lập `QT_QPA_PLATFORM=xcb`, `GDK_BACKEND=x11`, `WAYLAND_DISPLAY=""` (và `DEEPFACE_BACKEND_ENGINE=onnx`) để cửa sổ Gazebo Ignition, RViz2 và OpenCV HUD mở ổn định qua XWayland.

