import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
from openni import openni2
import numpy as np
import cv2
 
# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

import imageio
from IPython.display import HTML, display

import pyglet
import os
import numpy as np
import trimesh
import pyrr

from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags

pyglet.options['shadow_window'] = False

openni2.initialize()
dev = openni2.Device.open_any()
print(dev.get_device_info())
depth_stream = dev.create_depth_stream()
dev.set_image_registration_mode(True)
depth_stream.start()

capture = cv2.VideoCapture(0)
    
