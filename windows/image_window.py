from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget


class ImageWindow(QMainWindow):

    def __init__(self) -> None:

        super(ImageWindow, self).__init__()

        self.monitor = QDesktopWidget().screenGeometry(0)
        self.monitor.setHeight(int(0.45 * self.monitor.height()))

        self.setGeometry(self.monitor)

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)

        self.layout = QGridLayout()
        self.centerWidget.setLayout(self.layout)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label, 0, 0)