import sys
import numpy as np
from mayavi import mlab
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

# 创建一个随机的体素网格
voxelgrid = np.random.randint(0, 2, size=(10, 10, 10))

# 创建PyQtGraph的窗口
app = QtGui.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Voxelgrid Example")
view = win.addViewBox()
view.setAspectLocked(True)

# 创建Mayavi的场景
fig = mlab.figure(size=(500, 500), bgcolor=(0.7, 0.7, 0.7))
mlab.clf()

# 将体素网格数据传递给Mayavi
src = mlab.pipeline.scalar_field(voxelgrid)

# 创建体素网格的等值面
mlab.pipeline.iso_surface(src, contours=[0.5], opacity=0.5)

# 将Mayavi的场景渲染到PyQtGraph的窗口中
mlab.view(azimuth=45, elevation=45)
mlab.orientation_axes()
mlab.savefig("voxelgrid.png")
mlab.show()

# 将Mayavi的渲染结果显示在PyQtGraph的窗口中
img = pg.ImageItem(np.array(mlab.screenshot(antialiased=True)))
view.addItem(img)

# 运行PyQtGraph的事件循环
if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()
