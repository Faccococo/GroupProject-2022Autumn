import trimesh
import pyrender
Baymax_trimesh = trimesh.load('../models/BaymaxWhiteOBJ/Bigmax_White_OBJ.obj')
scene = pyrender.Scene.from_trimesh_scene(Baymax_trimesh)
dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
scene.add(dl)
pyrender.Viewer(scene, use_raymond_lighting=True)