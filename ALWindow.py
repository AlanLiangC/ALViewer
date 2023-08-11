import argparse
import logging

from mmengine.config import Config

from pathlib import Path
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget

from windows.main_window import ALMainWindow
from windows.detection_window import ALDetWindow


def parse_args():
    parser = argparse.ArgumentParser('Use AlanLiangs Viewer')
    parser.add_argument('-d', '--config', type=str, help='path to which config you use',
                        default='./configs/main_window/main_window.py')
    parser.add_argument('-e', '--experiments', type=str, help='path to where you store your OpenPCDet experiments',
                        default=str(Path.home() / 'repositories/PCDet/output'))
    args = parser.parse_args()

    return args

class ALWindow(QMainWindow):

    def __init__(self, cfg) -> None:

        super(ALWindow, self).__init__()

        self.cfg = cfg
        self.monitor = QDesktopWidget().screenGeometry(0)
        self.monitor.setHeight(int(0.3 * self.monitor.height()))
        self.monitor.setWidth(int(0.3 * self.monitor.width()))
        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)
        self.layout = QGridLayout()

        self.main_window = ALMainWindow(cfg=self.cfg)
        self.det_window = ALDetWindow(self.main_window)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.main_window_btn = QPushButton("open main window")
        self.det_task_btn = QPushButton("3D detection task")
        self.sem_task_btn = QPushButton("3D segmentation task")

        self.init_window()
        
    def init_window(self):
        self.centerWidget.setLayout(self.layout)
        self.layout.addWidget(self.image_label, 0, 0, 1, 3)
        self.width, self.height = self.image_label.width(), self.image_label.height()
        pixmap = QPixmap(str('pics/AlanLiang.png'))
        pixmap = pixmap.scaled(self.width, self.height, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

        self.layout.addWidget(self.main_window_btn, 1, 0)
        self.layout.addWidget(self.det_task_btn, 1, 1)
        self.layout.addWidget(self.sem_task_btn, 1, 2)
        self.main_window_btn.clicked.connect(self.open_main_window)
        self.det_task_btn.clicked.connect(self.detection_task)
        self.sem_task_btn.clicked.connect(self.segmentation_task)

    def open_main_window(self):
        
        self.main_window.reset()
        self.main_window.show()

    def detection_task(self):
        
        self.det_window.show()

    def segmentation_task(self):
        pass


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

    main()


        
