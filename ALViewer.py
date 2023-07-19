__author__  = "Alan Liang"
__contact__ = "liangao@sia.cn"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import os
import copy
import socket
from typing import Optional, Union
import logging
import argparse
import pandas
import numpy as np
from pathlib import Path



from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget

from mmengine.config import Config
import multiprocessing as mp
import pyqtgraph.opengl as gl
import matplotlib.cm as cm
import matplotlib as mpl
from windows.common import (AL_viewer, get_data_list, load_points, get_colors, create_boxes, parse_ann_info)
from windows.image_window import ImageWindow

def parse_args():
    parser = argparse.ArgumentParser('Use AlanLiangs Viewer')
    parser.add_argument('-d', '--config', type=str, help='path to which config you use',
                        default='./configs/main_window/main_window.py')
    parser.add_argument('-e', '--experiments', type=str, help='path to where you store your OpenPCDet experiments',
                        default=str(Path.home() / 'repositories/PCDet/output'))
    args = parser.parse_args()

    return args

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class ALWindow(QMainWindow):

    def __init__(self, cfg) -> None:
        super(ALWindow, self).__init__()

        self.cfg = cfg
        host_name = socket.gethostname()
        assert host_name == 'Liang'
        self.monitor = QDesktopWidget().screenGeometry(0)
        self.monitor.setHeight(self.monitor.height())
        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)
        self.image_window = ImageWindow()

        self.num_cpus = mp.cpu_count()
        self.pool = mp.Pool(self.num_cpus)

        self.color_dict = cfg.color_dict
        self.grid_dimensions = cfg.grid_dimensions

        self.dataset = cfg.dataset
        self.dataset_path = None
        self.data_prefix = None
        self.min_value = cfg.min_value
        self.max_value = cfg.max_value
        self.num_features = cfg.num_features
        self.color_feature = cfg.color_feature
        self.color_name = cfg.color_name
        self.point_size = cfg.point_size
        self.extension = cfg.extension
        self.d_type = np.float32
        self.intensity_multiplier = cfg.intensity_multiplier
        self.file_name = None
        self.file_list = None
        self.data_list = None
        self.lastDir = None
        self.current_mesh = None
        self.success = cfg.success
        self.boxes = {}
        self.index = -1
        self.row_height = 20
        self.always_show_det_or_sem = False

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)
        self.layout = QGridLayout()

        self.viewer = AL_viewer()
        self.grid = gl.GLGridItem()

        # Buttons
        self.reset_btn = QPushButton("reset")
        self.load_kitti_btn = QPushButton("KITTI")
        self.load_nuscenes_btn = QPushButton("nuScenes")
        self.choose_dir_btn = QPushButton("choose custom directory")
        self.show_sem_btn = QPushButton("show semantic")
        self.show_det_btn = QPushButton("show detection")
        self.show_img_btn = QPushButton("show images")
        self.show_sem_btn.setEnabled(False)
        self.show_det_btn.setEnabled(False)
        self.show_img_btn.setEnabled(False)
        

        self.prev_btn = QPushButton("<-")
        self.next_btn = QPushButton("->")
        self.next_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.color_title = QLabel("color code")
        self.color_label = QLabel(self.color_name)
        self.color_slider = QSlider(Qt.Horizontal)
        

        self.num_info = QLabel("")
        self.log_info = QLabel("")
        self.file_name_label = QLabel()

        self.always_show_ann = QCheckBox("continued anns")
        
        self.init_window()

    def init_window(self):
        self.centerWidget.setLayout(self.layout)

        self.viewer.setWindowTitle('drag & drop point cloud viewer')
        self.viewer.setCameraPosition(distance=2 * self.grid_dimensions)
        self.layout.addWidget(self.viewer, 0, 0, 1, 7)

        # grid
        self.grid.setSize(self.grid_dimensions, self.grid_dimensions)
        self.grid.setSpacing(1, 1)
        self.grid.translate(0, 0, -2)
        self.viewer.addItem(self.grid)

        self.reset_btn.clicked.connect(self.reset)
        self.layout.addWidget(self.reset_btn, 1, 0)
        self.always_show_ann.setEnabled(False)
        self.always_show_ann.stateChanged.connect(self.update_show_ann)
        self.layout.addWidget(self.always_show_ann, 1, 2)
        self.show_sem_btn.clicked.connect(self.show_sem)
        self.layout.addWidget(self.show_sem_btn, 1, 3)
        self.show_det_btn.clicked.connect(self.show_det)
        self.layout.addWidget(self.show_det_btn, 1, 4)
        self.show_img_btn.clicked.connect(self.show_img)
        self.layout.addWidget(self.show_img_btn, 1, 5)
        self.load_kitti_btn.clicked.connect(self.load_kitti)
        self.layout.addWidget(self.load_kitti_btn, 2, 3)
        self.load_nuscenes_btn.clicked.connect(self.load_nuscenes)
        self.layout.addWidget(self.load_nuscenes_btn, 2, 4)
        self.choose_dir_btn.clicked.connect(self.show_directory_dialog)
        self.layout.addWidget(self.choose_dir_btn, 2, 1)
        self.prev_btn.clicked.connect(self.decrement_index)
        self.next_btn.clicked.connect(self.increment_index)
        self.layout.addWidget(self.prev_btn, 2, 0)
        self.layout.addWidget(self.next_btn, 2, 2)
        self.color_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.color_title, 3, 0)
        self.color_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.color_label, 3, 2)
        self.color_slider.setMinimum(0)
        self.color_slider.setMaximum(6)
        self.color_slider.setValue(self.color_feature)
        self.color_slider.setTickPosition(QSlider.TicksBelow)
        self.color_slider.setTickInterval(1)
        self.layout.addWidget(self.color_slider, 3, 1)
        self.color_slider.valueChanged.connect(self.color_slider_change)



        self.current_row = 5

        self.num_info.setAlignment(Qt.AlignLeft)
        self.num_info.setMaximumSize(self.monitor.width(), self.row_height)
        self.layout.addWidget(self.num_info, self.current_row, 0)
        self.log_info.setAlignment(Qt.AlignLeft)
        self.log_info.setMaximumSize(self.monitor.width(), self.row_height)
        self.layout.addWidget(self.log_info, self.current_row, 1, 1, 3)
        self.file_name_label.setAlignment(Qt.AlignCenter)
        self.file_name_label.setMaximumSize(self.monitor.width(), 20)
        self.layout.addWidget(self.file_name_label, 1, 1, 1, 1)




    def reset(self) -> None:
        self.reset_viewer()

    def reset_viewer(self) -> None:

        self.num_info.setText(f'sequence_size: {len(self.data_list)}')
        self.viewer.items = []
        self.viewer.addItem(self.grid)

    def show_directory_dialog(self) -> None:

        directory = Path(os.getenv("HOME")) / 'AlanLiang/Projects/3D_Perception/ALViewer'

        if self.lastDir:
            directory = self.lastDir

        dir_name = QFileDialog.getExistingDirectory(self, "Open Directory", str(directory),
                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)

        if dir_name:
            # self.create_file_list(dir_name)
            self.lastDir = Path(dir_name)

    def log_string(self, pc: np.ndarray) -> None:

        log_string = f'intensity [ ' + f'{int(min(pc[:, 3]))}'.rjust(3, ' ') + \
                     f', ' + f'{int(max(pc[:, 3]))}'.rjust(3, ' ') + ']' + ' ' + \
                     f'median ' + f'{int(np.round(np.median(pc[:, 3])))}'.rjust(3, ' ') + ' ' + \
                     f'mean ' + f'{int(np.round(np.mean(pc[:, 3])))}'.rjust(3, ' ') + ' ' + \
                     f'std ' + f'{int(np.round(np.std(pc[:, 3])))}'.rjust(3, ' ')

        self.log_info.setText(log_string)

    def update_show_ann(self):
        
        if self.always_show_ann.isChecked():
            self.always_show_det_or_sem = True
            self.show_det()
        else:
            self.always_show_det_or_sem = False

    def color_slider_change(self) -> None:

        self.color_feature = self.color_slider.value()

        self.color_name = self.color_dict[self.color_feature]
        self.color_label.setText(self.color_name)

        if self.current_mesh:

            if self.data_list is not None:
                self.show_mmdet_dict(self.data_list[self.index])
            elif self.file_list is not None:
                self.show_pointcloud(self.file_list[self.index])
            else:
                raise TypeError('No data is load!')
            
    def show_sem(self):
        pass

    def show_det(self):
        #########
        # boxes #
        #########
        file_dict = self.data_list[self.index]
        annotations = parse_ann_info(file_dict)
        bboxes_3d = annotations['gt_bboxes_3d']
        box_info = create_boxes(bboxes_3d=bboxes_3d, COLORS=self.cfg.COLORS, dataset=self.dataset)

        self.boxes = box_info['boxes']
        box_items = box_info['box_items']
        l1_items = box_info['l1_items']
        l2_items = box_info['l2_items']

        for box_item, l1_item, l2_item in zip(box_items, l1_items, l2_items):
            self.viewer.addItem(box_item)
            self.viewer.addItem(l1_item)
            self.viewer.addItem(l2_item)

    def show_img(self):
        self.image_window.show()
        
    def show_pointcloud(self, filename: Union[dict, str]) -> None:
        pass

    def show_mmdet_dict(self, file_dict: dict) -> None:
        assert self.dataset_path is not None
        assert self.data_prefix is not None

        self.reset_viewer()

        if self.dataset in ['KITTI', 'nuScenes']:
            self.show_img_btn.setEnabled(True)
        
        lidar_points_path = file_dict['lidar_points']['lidar_path']
        lidar_points_path = os.path.join(self.dataset_path,self.data_prefix['pts'],lidar_points_path)
        self.file_name_label.setText(str(Path(lidar_points_path).name))

        if len(self.data_list) > 1:
            self.next_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
        else:
            self.next_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)

        ##########
        # points #
        ##########
        pc = load_points(lidar_points_path)
        pc = pc.reshape(-1, self.load_dim)
        self.log_string(pc)

        color_dict = {}
        color_dict.update({
            'pc': pc,
            'color_feature': self.color_feature,
            'min_value': self.min_value,
            'max_value': self.max_value
        })
        color_dict = get_colors(color_dict=color_dict)
        colors = color_dict['colors']
        self.success = color_dict['success']

        mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors)
        self.current_mesh = mesh
        self.current_pc = copy.deepcopy(pc)

        self.viewer.addItem(mesh)

        if file_dict.get('instances', None) is not None:
            self.show_det_btn.setEnabled(True)
            self.always_show_ann.setEnabled(True)
            
            if self.always_show_det_or_sem:
                self.show_det()


    def check_index_overflow(self) -> None:

        if self.index == -1:
            self.index = len(self.data_list) - 1

        if self.index >= len(self.data_list):
            self.index = 0
    
    def decrement_index(self) -> None:

        if self.index != -1:
            self.index -= 1
            self.check_index_overflow()

            if self.data_list is not None:
                self.show_mmdet_dict(self.data_list[self.index])
            elif self.file_list is not None:
                self.show_pointcloud(self.file_list[self.index])
            else:
                raise TypeError('No data is load!')


    def increment_index(self) -> None:

        if self.index != -1:

            self.index += 1
            self.check_index_overflow()

            if self.data_list is not None:
                self.show_mmdet_dict(self.data_list[self.index])
            elif self.file_list is not None:
                self.show_pointcloud(self.file_list[self.index])
            else:
                raise TypeError('No data is load!')


    def set_kitti(self) -> None:
        self.dataset = 'KITTI'
        self.min_value = -1
        self.max_value = -1
        self.num_features = 4
        self.load_dim = 4
        self.extension = 'bin'
        self.d_type = np.float32
        self.intensity_multiplier = 255
        self.color_dict[6] = 'not available'
        self.data_prefix=dict(pts='training/velodyne_reduced')

    def set_nuscenes(self) -> None:
        self.dataset = 'nuScenes'
        self.min_value = 0
        self.max_value = 31
        self.num_features = 5
        self.load_dim = 5
        self.extension = 'bin'
        self.intensity_multiplier = 1
        self.color_dict[6] = 'channel'
        self.data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')

    def load_kitti(self) -> None:
        self.dataset_path = Path(self.cfg.KITTI)
        kitti_pkl_path = self.dataset_path / 'kitti_infos_train.pkl'
        self.data_list = get_data_list(data_pkl_path=kitti_pkl_path)
        self.index = 0
        self.set_kitti()
        self.show_mmdet_dict(self.data_list[self.index])

    def load_nuscenes(self) -> None:
        self.dataset_path = Path(self.cfg.NUSCENES)
        nuscenes_pkl_path = self.dataset_path / 'nuscenes_infos_val.pkl'
        self.data_list = get_data_list(data_pkl_path=nuscenes_pkl_path)
        self.index = 0
        self.set_nuscenes()
        self.show_mmdet_dict(self.data_list[self.index])


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    app = QtWidgets.QApplication([])
    window = ALWindow(cfg=cfg)
    window.show()
    app.exec_()


if __name__ == '__main__':

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.debug(pandas.__version__)

    main()