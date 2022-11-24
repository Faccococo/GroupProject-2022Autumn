import cv2
import pyglet
from openni import openni2

from Render import Render

from offRender import offRender


# Import matplotlib libraries
# Some modules to display an animation using imageio.
# Import matplotlib libraries

# xc, yc, zc:摄像头坐标
# xz, yz, zz:监控摄像头正对一米处坐标
# xup, yup:人眼二维坐标
# w, h：屏幕宽与高


# d：深度
# seita：监控摄像头视场角

if __name__ == "__main__":
    pyglet.options['shadow_window'] = False

    openni2.initialize()
    dev = openni2.Device.open_any()
    print(dev.get_device_info())
    depth_stream = dev.create_depth_stream()
    dev.set_image_registration_mode(True)
    depth_stream.start()

    capture = cv2.VideoCapture(0)

    # 画面渲染
    offRender(depth_stream, capture)
