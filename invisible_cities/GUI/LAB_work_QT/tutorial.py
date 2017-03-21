import sys
import numpy as np
import csv
from PyQt5 import QtCore, QtWidgets, uic
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from PyQt5.uic import loadUiType

#
qtCreatorFile = "tutorial.ui" # Enter file here.
Ui_MainWindow, QMainWindow = loadUiType(qtCreatorFile)

class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.fig_dict = {}

        self.mplfigs.itemClicked.connect(self.changefig)

        fig = Figure()
        self.addmpl(fig)

    def addfig(self, name, fig):
        self.fig_dict[name] = fig
        self.mplfigs.addItem(name)

    def changefig(self, item):
        text = item.text()
        self.rmmpl()
        self.addmpl(self.fig_dict[text])

    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas,
                self.mplwindow, coordinates=True)
        self.mplvl.addWidget(self.toolbar)
    # def addmpl(self, fig):
    #     self.canvas = FigureCanvas(fig)
    #     self.mplvl.addWidget(self.canvas)
    #     self.canvas.draw()
    #     self.toolbar = NavigationToolbar(self.canvas,
    #             self, coordinates=True)
    #     self.addToolBar(self.toolbar)
    def rmmpl(self,):
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()



if __name__ == '__main__':
    import sys
    #from PyQt5 import QtGui
    import numpy as np

    fig1 = Figure()
    ax1f1 = fig1.add_subplot(111)
    ax1f1.plot(np.random.rand(5))

    fig2 = Figure()
    ax1f2 = fig2.add_subplot(121)
    ax1f2.plot(np.random.rand(5))
    ax2f2 = fig2.add_subplot(122)
    ax2f2.plot(np.random.rand(10))

    fig3 = Figure()
    ax1f3 = fig3.add_subplot(111)
    ax1f3.pcolormesh(np.random.rand(20,20))

    app = QtWidgets.QApplication(sys.argv)
    #app = QtGui.QApplication(sys.argv)
    main = Main()
    main.addfig('One plot', fig1)
    main.addfig('Two plots', fig2)
    main.addfig('Pcolormesh', fig3)
    main.show()
    sys.exit(app.exec_())
