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

from mmdet3d.structures import Box3DMode

MAX_LABEL = {
    'nuScenes': 31,
    'SemanticKITTI': 259
}

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


def load_points(pts_filename: str, data_type: str = 'np.float32') -> np.ndarray:
    """Private function to load point clouds data.

    Args:
        pts_filename (str): Filename of point clouds data.

    Returns:
        np.ndarray: An array containing point clouds data.
    """

    data_type = eval(data_type)
    try:
        pts_bytes = get(pts_filename)
        points = np.frombuffer(pts_bytes, dtype=data_type)
    except ConnectionError:
        mmengine.check_file_exist(pts_filename)
        if pts_filename.endswith('.npy'):
            points = np.load(pts_filename)
        else:
            points = np.fromfile(pts_filename, dtype=data_type)
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

    elif color_feature == 6:

        try:
            feature = pc[:, 4]
            success = True

        except IndexError:
            feature = pc[:, 3]

    # elif color_feature == 7: # semantic mode 

    #     assert color_dict.get('sem_info', None) is not None
    #     assert color_dict.get('dataset', None) is not None

    #     if color_dict['dataset'] == 'nuScenes':
    #         sem_label = color_dict['sem_info']['sem_label']
    #         label_mapping = color_dict['sem_info']['label_mapping']
    #         converted_pts_sem_mask = label_mapping[sem_label]

    else:
        raise IndexError('Please check the index of color feature!')

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

def limit_period(val: np.ndarray,
                 offset: float = 0.5,
                 period: float = np.pi):

    limited_val = val - (val / period + offset) * period
    return limited_val


def create_boxes(bboxes_3d, COLORS, dataset):
    boxes = {}
    box_items = []
    l1_items = []
    l2_items = []

    if dataset == "KITTI":
        bboxes_3d = Box3DMode.convert(bboxes_3d, Box3DMode.CAM,
                                                    Box3DMode.LIDAR)
        bboxes_3d[:,2] += bboxes_3d[:,5] / 2
    elif dataset == 'nuScenes':
        pass
    else:
        raise TypeError('Not set dataset')

    # create annotation boxes
    for annotation in bboxes_3d:

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

def parse_ann_info(info: dict):

        name_mapping = {
            'bbox_label_3d': 'gt_labels_3d',
            'bbox_label': 'gt_bboxes_labels',
            'bbox': 'gt_bboxes',
            'bbox_3d': 'gt_bboxes_3d',
            'depth': 'depths',
            'center_2d': 'centers_2d',
            'attr_label': 'attr_labels',
            'velocity': 'velocities',
        }
        instances = info['instances']
        # empty gt
        if len(instances) == 0:
            return None
        else:
            keys = list(instances[0].keys())
            ann_info = dict()
            for ann_name in keys:
                temp_anns = [item[ann_name] for item in instances]
                # map the original dataset label to training label

                if ann_name in name_mapping:
                    mapped_ann_name = name_mapping[ann_name]
                else:
                    mapped_ann_name = ann_name

                if 'label' in ann_name:
                    temp_anns = np.array(temp_anns).astype(np.int64) + 1
                elif ann_name in name_mapping:
                    temp_anns = np.array(temp_anns).astype(np.float32)
                else:
                    temp_anns = np.array(temp_anns)

                ann_info[mapped_ann_name] = temp_anns
            ann_info['instances'] = info['instances']

        ann_info['gt_bboxes_3d'] = np.hstack([ann_info['gt_bboxes_3d'], ann_info['gt_bboxes_labels'].reshape(-1,1)])

        return ann_info

def update_pts_color(colors):
    assert len(colors) > 0
    for i in range(len(colors)):
        single_color = colors[i]
        if len(single_color) == 3: # rgb mode
            if type(single_color) == tuple:
                single_color = list(single_color)
            single_color.append(255)
        single_color = tuple(single_color)
        colors[i] = single_color
    return colors
        

def creat_sem_points(sem_dict):
    dataset = sem_dict['dataset']
    data_type = 'np.uint8' if dataset == 'nuScenes' else 'np.int32'
    lidar_sem_label_path = sem_dict['lidar_sem_label_path']
    sem_info = sem_dict['sem_info']
    sem_label = load_points(pts_filename=lidar_sem_label_path, data_type=data_type)
    if dataset == 'SemanticKITTI':
        sem_label = sem_label.astype(np.int64)
        sem_label = sem_label % 2**16

    label_mapping = sem_info['label_mapping']
    seg_label_mapping = np.ones(MAX_LABEL[dataset] + 1, dtype=np.int64)
    for idx in label_mapping:
        seg_label_mapping[idx] = label_mapping[idx]

    converted_pts_sem_mask = seg_label_mapping[sem_label].astype(np.int64)
    # COLORS = update_pts_color(sem_info['COLORS'])
    # sem_colors = COLORS[converted_pts_sem_mask]
    norm = mpl.colors.Normalize(vmin=np.min(seg_label_mapping), vmax=np.max(seg_label_mapping))
    cmap = cm.jet  # sequential

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    sem_colors = m.to_rgba(converted_pts_sem_mask)
    sem_colors[:, [2, 1, 0, 3]] = sem_colors[:, [0, 1, 2, 3]]
    sem_colors[:, 3] = 0.5
    sem_dict.update({
        'sem_colors': sem_colors
    })

    return sem_dict
    