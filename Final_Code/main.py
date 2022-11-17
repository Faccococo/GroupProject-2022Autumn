import cv2
import pyglet
from openni import openni2

import Render

# Import matplotlib libraries
# Some modules to display an animation using imageio.
# Import matplotlib libraries

pyglet.options['shadow_window'] = False

openni2.initialize()
dev = openni2.Device.open_any()
print(dev.get_device_info())
depth_stream = dev.create_depth_stream()
dev.set_image_registration_mode(True)
depth_stream.start()

capture = cv2.VideoCapture(0)

# 画面渲染
Render(depth_stream, capture)
