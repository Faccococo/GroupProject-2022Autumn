from pyrender import PerspectiveCamera, \
    DirectionalLight, SpotLight, PointLight, \
    Mesh, Node, Scene, \
    Viewer, OffscreenRenderer
import time
from deformation import deformation_curve
from Locator import Locator
import pyrr
import trimesh
import numpy as np
import pyglet
import cv2
import math

pyglet.options['shadow_window'] = False

def getLocation(position_x, position_y, position_depth):
    x, y, z = Locator(position_x, position_y, position_depth)
    return x, y, z


def createCamPos(x=0.5, y=0, z=0.4):
    y_d, x_d, z_d = x, y, z
    x_d = 0 - x_d
    cam_pose = pyrr.matrix44.create_look_at((y_d, x_d, z_d), (0, 0, 0), (0, 0, 1))
    cam_pose = np.linalg.inv(cam_pose.T)
    return cam_pose


def getCamPosByCap(position_x, position_y, position_depth):
    x, y, z = getLocation(position_x, position_y, position_depth)
    return createCamPos(x, y, z)


# Fuze trimesh
fuze_trimesh = trimesh.load('../Final_Code/examples/models/fuze.obj')
fuze_mesh = Mesh.from_trimesh(fuze_trimesh)

# Drill trimesh
drill_trimesh = trimesh.load('../Final_Code/examples/models/drill.obj')
drill_mesh = Mesh.from_trimesh(drill_trimesh)
drill_pose = np.eye(4)
drill_pose[0, 3] = 0.1
drill_pose[2, 3] = -np.min(drill_trimesh.vertices[:, 2])

# Wood trimesh
wood_trimesh = trimesh.load('../Final_Code/examples/models/wood.obj')
wood_mesh = Mesh.from_trimesh(wood_trimesh)

# Water bottle trimesh
bottle_gltf = trimesh.load('../Final_Code/examples/models/WaterBottle.glb')
bottle_trimesh = bottle_gltf.geometry[list(bottle_gltf.geometry.keys())[0]]
bottle_mesh = Mesh.from_trimesh(bottle_trimesh)

bottle_pose = np.array([
    [1.0, 0.0, 0.0, 0.1],
    [0.0, 0.0, -1.0, -0.16],
    [0.0, 1.0, 0.0, 0.13],
    [0.0, 0.0, 0.0, 1.0],
])

boxv_trimesh = trimesh.creation.box(extents=0.1 * np.ones(3))
boxv_vertex_colors = np.random.uniform(size=(boxv_trimesh.vertices.shape))
boxv_trimesh.visual.vertex_colors = boxv_vertex_colors
boxv_mesh = Mesh.from_trimesh(boxv_trimesh, smooth=False)

boxf_trimesh = trimesh.creation.box(extents=0.1 * np.ones(3))
boxf_face_colors = np.random.uniform(size=boxf_trimesh.faces.shape)
boxf_trimesh.visual.face_colors = boxf_face_colors
boxf_mesh = Mesh.from_trimesh(boxf_trimesh, smooth=False)

points = trimesh.creation.icosphere(radius=0.05).vertices
point_colors = np.random.uniform(size=points.shape)
points_mesh = Mesh.from_points(points, colors=point_colors)

direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
spot_l = SpotLight(color=np.ones(3), intensity=10.0,
                    innerConeAngle=np.pi / 16, outerConeAngle=np.pi / 6)
point_l = PointLight(color=np.ones(3), intensity=10.0)

scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

pose = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],
    [1.0, 0.0,           0.0,           0.0],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],
    [0.0,  0.0,           0.0,          1.0]
])

direc_l_node = scene.add(direc_l, pose=pose)
spot_l_node = scene.add(spot_l, pose=pose)

fuze_node = Node(mesh=fuze_mesh, translation=np.array(
    [0.1, 0.15, -np.min(fuze_trimesh.vertices[:, 2])]))
scene.add_node(fuze_node)
boxv_node = Node(mesh=boxv_mesh, translation=np.array([-0.1, 0.10, 0.05]))
scene.add_node(boxv_node)
boxf_node = Node(mesh=boxf_mesh, translation=np.array([-0.1, -0.10, 0.05]))
scene.add_node(boxf_node)

drill_node = scene.add(drill_mesh, pose=drill_pose)
bottle_node = scene.add(bottle_mesh, pose=bottle_pose)
wood_node = scene.add(wood_mesh)

# add camera to scene
cam = PerspectiveCamera(yfov=(np.pi / 3.0), aspectRatio=16 / 9)

position_x, position_y, position_depth = 0, 0, 2, 
cam_pose = getCamPosByCap(position_x, position_y, position_depth)

cam_node = scene.add(cam, cam_pose)

# create viewerqq
r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)

i = 0.00
last_depth = position_depth

while True:
    time.sleep(1 / 60)

    position_x += 0.1

    try:
        if abs(position_depth - last_depth) > 0.1:
            position_depth = 0.95 * last_depth + 0.05 * position_depth
        # position_depth = 0.95 * last_depth + 0.05 * position_depth

        scene.set_pose(node=cam_node, pose=getCamPosByCap(
            position_x, position_y, position_depth))

    except ValueError:
        cam_pose = pyrr.matrix44.create_look_at(
            (0.5, 0, 0.4), (0, 0, 0), (0, 0, 1))
        cam_pose = np.linalg.inv(cam_pose.T)
        cam_pose[-1] = [0, 0, 0, 1]
        scene.set_pose(node=cam_node, pose=cam_pose)

    i += 0.01
    last_depth = position_depth
    color, _ = r.render(scene)
    x, y, z = getLocation(position_x, position_y, position_depth)
    cv2.imshow("origin",color)
    cv2.imshow("result", deformation_curve(4, 8, math.pi / 2, (x, y), color))

    key = cv2.waitKey(50)
    if key == ord('q'):  # 判断是哪一个键按下
        break



