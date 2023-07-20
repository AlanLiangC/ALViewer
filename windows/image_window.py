import os
import copy
from pathlib import Path
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget


class ImageWindow(QMainWindow):

    def __init__(self) -> None:

        super(ImageWindow, self).__init__()

        self.monitor = QDesktopWidget().screenGeometry(0)
        self.monitor.setHeight(int(0.45 * self.monitor.height()))
        self.monitor.setWidth(int(0.45 * self.monitor.width()))

        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)

        self.index = -1
        self.img_file_list = []
        self.img_name_list = []
        self.dataset_path = None
        self.base_image_path = None

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)
        self.layout = QGridLayout()
        

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.prev_btn = QPushButton("<-")
        self.next_btn = QPushButton("->")
        self.next_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.file_name_label = QLabel()
        self.file_name_label.setAlignment(Qt.AlignCenter)
        self.choose_img = QComboBox()
        self.choose_img.setEnabled(False)
    
        self.init_window()

    def init_window(self):
        self.centerWidget.setLayout(self.layout)
        self.layout.addWidget(self.image_label, 0, 0, 1, 3)
        self.width, self.height = self.image_label.width(), self.image_label.height()

        self.layout.addWidget(self.prev_btn, 1, 0)
        self.layout.addWidget(self.next_btn, 1, 2)
        self.prev_btn.clicked.connect(self.decrement_index)
        self.next_btn.clicked.connect(self.increment_index)

        self.layout.addWidget(self.file_name_label, 1, 1)

        self.choose_img.currentIndexChanged.connect(self.image_selection_change)
        
        self.layout.addWidget(self.choose_img,2,0,1,3)

    def check_index_overflow(self) -> None:

        if self.index == -1:
            self.index = len(self.img_file_list) - 1

        if self.index >= len(self.img_file_list):
            self.index = 0
    
    def decrement_index(self):
        if self.index != -1:
            self.index -= 1
            self.check_index_overflow()

            self.show_image(image_path=self.img_file_list[self.index])


    def increment_index(self):
        if self.index != -1:
            self.index += 1
            self.check_index_overflow()
            self.show_image(image_path=self.img_file_list[self.index])

    def image_selection_change(self):
        self.index = self.choose_img.currentIndex()
        self.show_image(image_path=self.img_file_list[self.index])

    def show_image(self, image_path):

        file_name_label = self.img_name_list[self.index] + str(self.base_image_path)
        self.file_name_label.setText(file_name_label)
        self.choose_img.setCurrentIndex(self.index)

        pixmap = QPixmap(str(image_path))
        pixmap = pixmap.scaled(self.width, self.height, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        
    def update_kitti_image_file_list(self):
        self.dataset_path = Path(self.dataset_path) / 'training'
        data_list = os.listdir(Path(self.dataset_path))
        for sub_data in data_list:
            if sub_data == 'image_3':
                new_img_path = os.path.join(self.dataset_path, str(sub_data), self.base_image_path)
                self.img_file_list.append(new_img_path)
                self.img_name_list.append("Right view: ")

            elif sub_data == 'prev_2':
                for i in range(3):
                    prev_img_name = Path(str(self.base_image_path.split('.')[0]) + '_0{}'.format(i+1))
                    new_img_path = os.path.join(self.dataset_path, str(sub_data), prev_img_name)
                    self.img_file_list.append(new_img_path)
                    self.img_name_list.append("Pre left view {}: ".format(i+1))

            elif sub_data == 'prev_3':
                for i in range(3):
                    prev_img_name = Path(str(self.base_image_path.split('.')[0]) + '_0{}'.format(i+1))
                    new_img_path = os.path.join(self.dataset_path, str(sub_data), prev_img_name)
                    self.img_file_list.append(new_img_path)
                    self.img_name_list.append("Pre right view {}: ".format(i+1))

            else:
                continue

        assert len(self.img_file_list) == len(self.img_name_list)

        if len(self.img_file_list) > 1:
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)

    def update_nuscenes_image_file_list(self, img_dict):
        self.dataset_path = img_dict['dataset_path']
        data_prefix = img_dict['data_prefix']
        image_info = img_dict['data_info']['images']

        for key, value in image_info.items():

            image_path = value['img_path']
            image_path = os.path.join(self.dataset_path,data_prefix[key],image_path)
            self.img_file_list.append(image_path)
            self.img_name_list.append(str(key))

        assert len(self.img_file_list) == len(self.img_name_list)

        if len(self.img_file_list) > 1:
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
 
        
    def update_image_window(self):
        self.choose_img.setEnabled(True)
        self.choose_img.addItems(self.img_name_list)
        
    def show_kitti_image(self, img_dict):
        self.reset_img_space()

        # dataset = img_dict['dataset']
        self.dataset_path = img_dict['dataset_path']
        data_prefix = img_dict['data_prefix']
        data_info = img_dict['data_info']

        lidar_points_path = data_info['lidar_points']['lidar_path']
        self.base_image_path = Path(lidar_points_path).name.replace('.bin', '.png')
        image_path = os.path.join(self.dataset_path,data_prefix['img'],self.base_image_path)
        
        assert os.path.isfile(image_path)
        self.img_file_list.append(image_path)
        self.img_name_list.append("Left view: ")
        self.index += 1

        self.update_kitti_image_file_list()
        self.update_image_window()
        self.show_image(image_path=self.img_file_list[self.index])

    def show_nuscenes_image(self, img_dict):
        self.reset_img_space()

        try:
            self.update_nuscenes_image_file_list(img_dict = img_dict)
            self.update_image_window()
            self.index += 1
        except:
            raise TypeError('Load nuScenes error!')
        self.show_image(image_path=self.img_file_list[self.index])


    def reset_img_space(self):

        self.img_file_list = []
        self.img_name_list = []
        self.index = -1



        