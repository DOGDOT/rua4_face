import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'rua4_face'


def collect_files(pattern):
    return [p for p in glob(pattern) if os.path.isfile(p)]


setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            collect_files(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'launch', 'include'),
            collect_files(os.path.join('launch', 'include', '*.launch.py'))),
        (os.path.join('share', package_name, 'config', 'rviz'),
            collect_files(os.path.join('config', 'rviz', '*.rviz'))),
        (os.path.join('share', package_name, 'results'),
            collect_files(os.path.join('results', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Phạm Thành Đông',
    maintainer_email='23020735@vnu.edu.vn',
    description='Nhận diện biểu cảm khuôn mặt thời gian thực (DeepFace + OpenCV) và điều khiển TurtleBot 4 qua /cmd_vel (ROS 2 Humble)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'emotion_recognition_node = rua4_face.emotion_recognition_node:main',
            'emotion_control_node = rua4_face.emotion_control_node:main',
        ],
    },
)
