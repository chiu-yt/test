"""KITTI box geometry adapted from HengLan/Visualize-KITTI-Objects-in-Videos.

MIT License

Copyright (c) 2022 Fan Lab@UNT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np


def transform_3dbox_to_pointcloud(dimension, location, rotation):
    height, width, lenght = dimension
    x, y, z = location
    x_corners = [lenght/2, lenght/2, -lenght/2, -lenght/2, lenght/2, lenght/2, -lenght/2, -lenght/2]
    y_corners = [0, 0, 0, 0, -height, -height, -height, -height]
    z_corners = [width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    R_matrix = np.array([[np.cos(rotation), 0, np.sin(rotation)],
                         [0, 1, 0],
                         [-np.sin(rotation), 0, np.cos(rotation)]])
    corners_3d = np.dot(R_matrix, corners_3d).T
    corners_3d = corners_3d + np.array([x, y, z])
    corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])
    return corners_3d


def transform_3dbox_to_image(dimension, location, rotation, calib):
    height, width, lenght = dimension
    x, y, z = location
    x_corners = [lenght/2, lenght/2, -lenght/2, -lenght/2, lenght/2, lenght/2, -lenght/2, -lenght/2]
    y_corners = [0, 0, 0, 0, -height, -height, -height, -height]
    z_corners = [width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    R_matrix = np.array([[np.cos(rotation), 0, np.sin(rotation)],
                         [0, 1, 0],
                         [-np.sin(rotation), 0, np.cos(rotation)]])
    corners_3d = np.dot(R_matrix, corners_3d).T
    corners_3d = corners_3d + np.array([x, y, z])
    if np.any(corners_3d[:, 2] < 0.1):
        corners_3d_img = None
    else:
        corners_3d_temp = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
        corners_3d_img = np.matmul(corners_3d_temp, calib['P2'].T)
        corners_3d_img = corners_3d_img[:, :2] / corners_3d_img[:, 2][:, None]
    return corners_3d_img
