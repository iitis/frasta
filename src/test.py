import sys
import numpy as np
from pyqtgraph.Qt import QtWidgets, QtGui
import pyqtgraph.opengl as gl

app = QtWidgets.QApplication(sys.argv)
w = gl.GLViewWidget()
#w.setBackgroundColor('w')
w.show()
w.setWindowTitle('GLScatterPlotItem Test')

N = 10000
points = np.random.normal(size=(N,3)) * 100 + 500
points = points.astype(np.float32)

sp = gl.GLScatterPlotItem(pos=points, color=(1,0,0,1), size=1)#, pxMode=True)
w.addItem(sp)
center = np.mean(points, axis=0)
w.setCameraPosition(pos=QtGui.QVector3D(*center), distance=1000)

sys.exit(app.exec_())
