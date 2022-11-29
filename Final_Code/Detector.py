import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Import matplotlib libraries

# Some modules to display an animation using imageio.

module = hub.load("./Final_Code/movenet_singlepose_thunder_4")
image_size = 256
# module = hub.load("./Final_Code/movenet_singlepose_lightning_4")
# image_size = 192


def rebound(a):
    if a > 1:
        return 1
    else:
        return a


def movenet(input_image):
    model = module.signatures['serving_default']
    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores


def Detector(depth_stream, capture):
    frame = depth_stream.read_frame()
    # 转换数据格式
    dframe_data = np.array(frame.get_buffer_as_triplet()
                           ).reshape([480, 640, 2])
    dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
    dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')

    dpt2 *= 255
    # 对于为什么要乘于255的解答
    # 深度图像的深度值 是按照16位长度（两字节）的数据格式存储的，也可以认为前八位是高字节，后八位是低字节。
    # 因此一张深度图像如果是 640*480分辨率的话，那么图像字节大小 就是 640480*2，其中一个字节是8位（255）
    dpt = dpt1 + dpt2

    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)  # 镜像操作'
    frame = cv2.resize(frame, (image_size, image_size))
    input_image = tf.expand_dims(frame, axis=0)
    input_image = tf.image.resize_with_pad(input_image, image_size, image_size)
    points = movenet(input_image)
    point_1 = points[0][0][1]
    point_2 = points[0][0][2]
    for point in points[0][0][1:3]:
        # print((point[0] * image_size, point[1] * image_size))
        cv2.circle(frame, center=(int(point[1] * image_size), int(
            point[0] * image_size)), radius=3, color=(255, 0, 0), thickness=3)

    # 取平均
    position_x = (point_1[1] + point_2[1]) / 2
    position_y = (point_1[0] + point_2[0]) / 2
    position_depth = ((dpt[int(rebound(point_1[0]) * 480) - 1, int(rebound(point_1[1]) * 640) - 1] +
                      dpt[int(rebound(point_2[0]) * 480) - 1, int(rebound(point_2[1]) * 640) - 1]) / 2) * 0.001

    return position_x, position_y, position_depth, frame
