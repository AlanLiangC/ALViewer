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



from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget
from pyqtgraph.Qt import QtGui

import multiprocessing as mp
import pyqtgraph.opengl as gl
import matplotlib.cm as cm
import matplotlib as mpl


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

class ALWindow(QMainWindow):

    def __init__(self) -> None:
        super(ALWindow, self).__init__()

        host_name = socket.gethostname()
        assert host_name == 'Liang'
        self.monitor = QDesktopWidget().screenGeometry(0)
        self.monitor.setHeight(self.monitor.height())
        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)

        self.num_cpus = mp.cpu_count()
        self.pool = mp.Pool(self.num_cpus)


        self.grid_dimensions = 20


        self.d_type = np.float32
        self.file_name = None
        self.file_list = None
        self.data_list = None
        self.lastDir = None
        self.current_mesh = None
        self.boxes = {}
        self.index = -1
        self.row_height = 20

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)
        self.layout = QGridLayout()

        self.viewer = AL_viewer()
        self.grid = gl.GLGridItem()

        self.init_window()

    def init_window(self):
        self.centerWidget.setLayout(self.layout)

        self.viewer.setWindowTitle('drag & drop point cloud viewer')
        self.viewer.setCameraPosition(distance=2 * self.grid_dimensions)
        self.layout.addWidget(self.viewer, 0, 0, 1, 6)

        # grid
        self.grid.setSize(self.grid_dimensions, self.grid_dimensions)
        self.grid.setSpacing(1, 1)
        self.grid.translate(0, 0, -2)
        self.viewer.addItem(self.grid)


def main():
    args = parse_args()

    # Load config

    app = QtWidgets.QApplication([])
    window = ALWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.debug(pandas.__version__)

    main()