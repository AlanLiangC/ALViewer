import os
import pickle as pkl
import mmengine
from mmengine.fileio import get
import numpy as np

import matplotlib.cm as cm
import matplotlib as mpl

from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui


class AL_viewer(gl.GLViewWidget):
    
    def __init__(self, parent=None, devicePixelRatio=None, rotationMethod='euler'):
        super().__init__(parent, devicePixelRatio, rotationMethod)

        self.noRepeatKeys = [Qt.Key.Key_W, Qt.Key.Key_S, Qt.Key.Key_A, Qt.Key.Key_D, Qt.Key.Key_Q, Qt.Key.Key_E,
            Qt.Key.Key_Right, Qt.Key.Key_Left, Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown]
        
        self.speed = 1
        
    def evalKeyState(self):
        vel_speed = 10 * self.speed 
        if len(self.keysPressed) > 0:
            for key in self.keysPressed:
                if key == Qt.Key.Key_Right:
                    self.orbit(azim=-self.speed, elev=0)
                elif key == Qt.Key.Key_Left:
                    self.orbit(azim=self.speed, elev=0)
                elif key == Qt.Key.Key_Up:
                    self.orbit(azim=0, elev=-self.speed)
                elif key == Qt.Key.Key_Down:
                    self.orbit(azim=0, elev=self.speed)
                elif key == Qt.Key.Key_A:
                    self.pan(vel_speed * self.speed, 0, 0, 'view-upright')
                elif key == Qt.Key.Key_D:
                    self.pan(-vel_speed, 0, 0, 'view-upright')
                elif key == Qt.Key.Key_W:
                    self.pan(0, vel_speed, 0, 'view-upright')
                elif key == Qt.Key.Key_S:
                    self.pan(0, -vel_speed, 0, 'view-upright')
                elif key == Qt.Key.Key_Q:
                    self.pan(0, 0, vel_speed, 'view-upright')
                elif key == Qt.Key.Key_E:
                    self.pan(0, 0, -vel_speed, 'view-upright')
                elif key == Qt.Key.Key_PageUp:
                    pass
                elif key == Qt.Key.Key_PageDown:
                    pass
                self.keyTimer.start(16)
        else:
            self.keyTimer.stop()


def get_data_list(data_pkl_path):
    with open(data_pkl_path, 'rb') as f:
        data = pkl.load(f)
    data_list = data['data_list']
    f.close()

    return data_list


def load_points(pts_filename: str) -> np.ndarray:
    """Private function to load point clouds data.

    Args:
        pts_filename (str): Filename of point clouds data.

    Returns:
        np.ndarray: An array containing point clouds data.
    """
    try:
        pts_bytes = get(pts_filename)
        points = np.frombuffer(pts_bytes, dtype=np.float32)
    except ConnectionError:
        mmengine.check_file_exist(pts_filename)
        if pts_filename.endswith('.npy'):
            points = np.load(pts_filename)
        else:
            points = np.fromfile(pts_filename, dtype=np.float32)
    return points


def get_colors(color_dict: dict) -> dict:
    color_feature = color_dict['color_feature']
    pc = color_dict['pc']
    min_value = color_dict['min_value']
    max_value = color_dict['max_value']

    # create colormap
    if color_feature == 0:

        success = True
        feature = pc[:, 0]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 1:

        success = True
        feature = pc[:, 1]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 2:

        success = True
        feature = pc[:, 2]
        min_value = -1.5
        max_value = 0.5

    elif color_feature == 3:

        success = True
        feature = pc[:, 3]
        min_value = 0
        max_value = 255

    elif color_feature == 4:

        success = True
        feature = np.linalg.norm(pc[:, 0:3], axis=1)

        try:
            min_value = np.min(feature)
            max_value = np.max(feature)
        except ValueError:
            min_value = 0
            max_value = np.inf

    elif color_feature == 5:

        success = True
        feature = np.arctan2(pc[:, 1], pc[:, 0]) + np.pi
        min_value = 0
        max_value = 2 * np.pi

    else:  # self.color_feature == 6:

        try:
            feature = pc[:, 4]
            success = True

        except IndexError:
            feature = pc[:, 3]

    norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)

    if color_feature == 5:
        cmap = cm.hsv  # cyclic
    else:
        cmap = cm.jet  # sequential

    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = m.to_rgba(feature)
    colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
    colors[:, 3] = 0.5

    color_dict.update({
        'colors': colors,
        'success': success
    })

    return color_dict


def create_boxes(annotations, COLORS):
    boxes = {}
    box_items = []
    l1_items = []
    l2_items = []
    # create annotation boxes
    for annotation in annotations:

        x, y, z, w, l, h, rotation, category = annotation

        rotation = np.rad2deg(rotation) + 90

        try:
            color = COLORS[int(category) - 1]
        except IndexError:
            color = (255, 255, 255, 255)

        box = gl.GLBoxItem(QtGui.QVector3D(1, 1, 1), color=color)

        box.setSize(l, w, h)
        box.translate(-l / 2, -w / 2, -h / 2)
        box.rotate(angle=rotation, x=0, y=0, z=1)
        box.translate(x, y, z)

        box_items.append(box)

        #################
        # heading lines #
        #################

        p1 = [-l / 2, -w / 2, -h / 2]
        p2 = [l / 2, -w / 2, h / 2]

        pts = np.array([p1, p2])

        l1 = gl.GLLinePlotItem(pos=pts, width=2 / 3, color=color, antialias=True, mode='lines')
        l1.rotate(angle=rotation, x=0, y=0, z=1)
        l1.translate(x, y, z)

        l1_items.append(l1)

        p3 = [-l / 2, -w / 2, h / 2]
        p4 = [l / 2, -w / 2, -h / 2]

        pts = np.array([p3, p4])

        l2 = gl.GLLinePlotItem(pos=pts, width=2 / 3, color=color, antialias=True, mode='lines')
        l2.rotate(angle=rotation, x=0, y=0, z=1)
        l2.translate(x, y, z)

        l2_items.append(l2)

        distance = np.linalg.norm([x, y, z], axis=0)
        boxes[distance] = (box, l1, l2)

    box_info = {
        'boxes' : boxes,
        'box_items' : box_items,
        'l1_items' : l1_items,
        'l2_items' : l2_items
    }

    return box_info