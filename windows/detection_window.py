import os
import argparse
import logging

from mmengine.config import Config

from pathlib import Path
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget

CFG_PATH = Path(os.getcwd()) / 'detection_tasks' / 'configs'
PRETRAINED_MODELS_PATH = Path(os.getcwd()) / 'detection_tasks' / 'pretrained_models'

class ALDetWindow(QMainWindow):

    def __init__(self, main_window: QMainWindow) -> None:

        super(ALDetWindow, self).__init__()

        self.main_window = main_window
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

        self.layout.addWidget(self.refresh_window_btn, 3, 0)
        self.refresh_window_btn.setStyleSheet("background-color: red;")
        self.refresh_window_btn.clicked.connect(self.defresh_window)
        self.layout.addWidget(self.show_gt_btn, 3, 1)
        self.show_gt_btn.clicked.connect(self.show_gt)
        self.layout.addWidget(self.inference_btn, 3, 2)
        self.show_gt_btn.clicked.connect(self.inference)
        

    def init_det_files(self):

        configs = os.listdir(CFG_PATH)
        pretrained_models = os.listdir(PRETRAINED_MODELS_PATH)
        self.select_cfg_file.addItems(configs)
        self.select_pretrained_model.addItems(pretrained_models)

    def defresh_window(self):

        if self.main_window.success:
            self.show_gt_btn.setEnabled(True)    

    def show_gt(self):
        if self.main_window.success:
            self.main_window.show_det()

    def inference(self):
        pass

