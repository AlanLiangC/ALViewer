import os
import argparse
import logging
import socket
from mmengine.config import Config

from pathlib import Path
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget

from tools.mmdet_inference import mmdet_inference
from windows.common import create_boxes

CFG_PATH = Path(os.getcwd()) / 'detection_tasks' / 'configs'
PRETRAINED_MODELS_PATH = Path(os.getcwd()) / 'detection_tasks' / 'pretrained_models'

class ALDetWindow(QMainWindow):

    def __init__(self, main_window: QMainWindow, det_task_config) -> None:

        super(ALDetWindow, self).__init__()

        self.main_window = main_window
        self.det_task_config = det_task_config

        host_name = socket.gethostname()
        if host_name == 'Liang':
            self.monitor = QDesktopWidget().screenGeometry(1)
            self.monitor.setHeight(int(0.3 * self.monitor.height()))
            self.monitor.setWidth(int(0.3 * self.monitor.width()))
        else:
            self.monitor = QDesktopWidget().screenGeometry(0)
            self.monitor.setHeight(int(0.3 * self.monitor.height()))
            self.monitor.setWidth(int(0.3 * self.monitor.width()))

        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)
        self.layout = QGridLayout()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.select_cfg_info = QLabel("Select config file:")
        self.select_cfg_info.setAlignment(Qt.AlignCenter)
        self.select_pretrained_model_info = QLabel("Select pretrained model:")
        self.select_pretrained_model_info.setAlignment(Qt.AlignCenter)

        self.select_cfg_file = QComboBox()
        self.select_pretrained_model = QComboBox()
        self.show_gt_btn = QPushButton("show gt boxes")
        self.show_gt_btn.setEnabled(False)   
        self.refresh_window_btn = QPushButton("refresh window")
        self.inference_btn = QPushButton("inference")
        self.clear_boxes_btn = QPushButton("clear anno")

        self.use_window_points_box = QCheckBox("use window points")

        self.init_window()
        
    def init_window(self):
        self.centerWidget.setLayout(self.layout)
        self.layout.addWidget(self.image_label, 0, 0, 1, 3)
        self.width, self.height = self.image_label.width(), self.image_label.height()
        pixmap = QPixmap(str('pics/ALDetection.png'))
        pixmap = pixmap.scaled(self.width, self.height, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

        self.layout.addWidget(self.select_cfg_info, 1, 0)
        self.layout.addWidget(self.select_pretrained_model_info, 2, 0)
        self.layout.addWidget(self.select_cfg_file, 1, 1, 1, 2)
        self.layout.addWidget(self.select_pretrained_model, 2, 1, 1, 2)
        self.init_det_files()

        self.layout.addWidget(self.refresh_window_btn, 3, 0, 2, 1)
        self.refresh_window_btn.setStyleSheet("background-color: red;")
        self.refresh_window_btn.setFixedSize(180, 60)
        self.refresh_window_btn.clicked.connect(self.defresh_window)
        self.layout.addWidget(self.show_gt_btn, 3, 1)
        self.show_gt_btn.clicked.connect(self.show_gt)
        self.layout.addWidget(self.inference_btn, 4, 2)
        self.inference_btn.clicked.connect(self.inference)
        self.layout.addWidget(self.clear_boxes_btn, 3, 2)
        self.clear_boxes_btn.clicked.connect(self.clear_boxes)
        self.layout.addWidget(self.use_window_points_box, 4, 1)
        

    def init_det_files(self):

        try:
            configs = os.listdir(CFG_PATH)
            pretrained_models = os.listdir(PRETRAINED_MODELS_PATH)
            self.select_cfg_file.addItems(configs)
            self.select_pretrained_model.addItems(pretrained_models)
        except:
            raise TypeError('Its a empty folder!')

    def defresh_window(self):

        if self.main_window.success:
            self.show_gt_btn.setEnabled(True)    
        else:
            self.show_gt_btn.setEnabled(False)

    def show_gt(self):
        if self.main_window.success:
            self.main_window.show_det()

    def init_det_cfgs(self):

        try:
            self.cfg_file = self.select_cfg_file.currentText()
            self.pretrained_model = self.select_pretrained_model.currentText()

            self.cfg_file = CFG_PATH / self.cfg_file
            self.pretrained_model = str(PRETRAINED_MODELS_PATH / self.pretrained_model)

        except:
            raise ValueError('At least one model show be listed in the folder!')
        
    def clear_boxes(self):
        self.main_window.clear_boxes()

        
    def inference(self):

        self.init_det_cfgs()
        
        assert hasattr(self, 'cfg_file')
        assert hasattr(self, 'pretrained_model')

        
        assert getattr(self.main_window, 'lidar_points_path', None) is not None

        pcd_file = self.main_window.lidar_points_path

        self.det_task_config.update(
            {
                'pcd': pcd_file,
                'config': self.cfg_file,
                'checkpoint': self.pretrained_model
            }
        )

        if self.use_window_points_box.isChecked():
            if hasattr(self.main_window, 'current_pc'):
                self.det_task_config['pcd'] = self.main_window.current_pc

        bboxes_3d = mmdet_inference(cfgs=self.det_task_config)

        box_info = create_boxes(bboxes_3d=bboxes_3d, COLORS=self.main_window.cfg.COLORS, dataset=self.main_window.dataset, mode='inference')

        # self.boxes = box_info['boxes']
        box_items = box_info['box_items']
        l1_items = box_info['l1_items']
        l2_items = box_info['l2_items']

        for box_item, l1_item, l2_item in zip(box_items, l1_items, l2_items):
            self.main_window.viewer.addItem(box_item)
            self.main_window.viewer.addItem(l1_item)
            self.main_window.viewer.addItem(l2_item)
        
        print('Inference OK!')
        

