"""
Define plotting functions that return a Figure (rather than using pyplot)
Used by GUI applications
"""
import sys
import numpy as np

from matplotlib.figure import Figure
import invisible_cities.core.mpl_functions import set_plot_labels
from invisible_cities.core.core_functions import define_window
from invisible_cities.core.core_functions import loc_elem_1d
from   invisible_cities.core.system_of_units_c import units


def _hist_outline(dataIn, *args, **kwargs):
    (histIn, binsIn) = np.histogram(dataIn, bins='auto', *args, **kwargs)

    stepSize = binsIn[1] - binsIn[0]

    bins = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    data = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2*bb + 1] = histIn[bb]
            data[2*bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0

    return (bins, data)

def hist_1d(data, xlo, xhi):
    """Returns a Figure corresponding to 1d histogram"""
    (bins, n) = _hist_outline(data)
    ylo = 0
    yhi = max(n) * 1.1

    fig = Figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(bins, n, 'k-')
    ax1.axis([xlo, xhi, ylo, yhi])

    return fig

def plot_vector(v):
    """Plot vector v """
    fig = Figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(v)
    return fig


def plot_xy(x,y):
    """Plot y vs x """
    fig = Figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x,y)
    return fig


def plot_pmt_waveforms(pmtwfdf, zoom=False, window_size=800):
    """Take as input a vector storing the PMT wf and plot the waveforms"""
    fig = Figure(figsize=(12, 12))
    for i in range(len(pmtwfdf)):
        first, last = 0, len(pmtwfdf[i])
        if zoom:
            first, last = define_window(pmtwfdf[i], window_size)

        ax = fig.add_subplot(3, 4, i+1)
        set_plot_labels(xlabel="samples", ylabel="adc")
        ax.plot(pmtwfdf[i][first:last])
    return fig


def plot_pmt_signals_vs_time_mus(pmt_signals,
                                 pmt_active,
                                 t_min      =    0,
                                 t_max      = 1200,
                                 signal_min =    0,
                                 signal_max =  200):
    """Plot all the PMT signals versus time in mus (tmin, tmax in mus)."""

    tstep = 25
    PMTWL = pmt_signals[0].shape[0]
    signal_t = np.arange(0., PMTWL * tstep, tstep)/units.mus
    fig = Figure(figsize=(12, 12))

    for j, i in enumerate(pmt_active):
        ax1 = fig.add_subplot(3, 4, j+1)
        ax1.set_xlim([t_min, t_max])
        ax1.set_ylim([signal_min, signal_max])
        set_plot_labels(xlabel = "t (mus)",
                        ylabel = "signal (pes/adc)")

        ax1.plot(signal_t, pmt_signals[i])

    return fig


def plot_signal_vs_time_mus(signal,
                            t_min      =    0,
                            t_max      = 1200,
                            signal_min =    0,
                            signal_max =  200):
    """Plot signal versus time in mus (tmin, tmax in mus). """
    tstep = 25 # in ns
    PMTWL = signal.shape[0]
    signal_t = np.arange(0., PMTWL * tstep, tstep)/units.mus

    fig = Figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlim([t_min, t_max])
    ax1.set_ylim([signal_min, signal_max])
    set_plot_labels(xlabel = "t (mus)",
                    ylabel = "signal (pes/adc)")
    ax1.plot(signal_t, signal)
    return fig



def plot_s12(S12):
    """Plot the peaks of input S12.

    S12 is a dictionary
    S12[i] for i in keys() are the S12 peaks
    """
    fig = Figure(figsize=(12, 12))
    set_plot_labels(xlabel = "t (mus)",
                    ylabel = "S12 (pes)")
    xy = len(S12)
    if xy == 1:
        t = S12[0][0]
        E = S12[0][1]
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(t, E)
    else:
        x = 3
        y = xy/x
        if y % xy != 0:
            y = int(xy/x) + 1
        for i in S12.keys():
            ax1 = fig.add_subplot(x, y, i+1)
            t = S12[i][0]
            E = S12[i][1]
            ax1.plot(t, E)
    return fig




qtCreatorFile = "event_maps.ui" # Enter file here.
Ui_MainWindow, QMainWindow = loadUiType(qtCreatorFile)

class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.fig_dict = {}

        self.mplfigs.itemClicked.connect(self.change_figure)

        fig = Figure()
        self.add_mpl(fig)

    def add_figure(self, name, fig):
        test_key = self.fig_dict.pop(name, None)
        self.fig_dict[name] = fig
        if not test_key: # key not in dict
            self.mplfigs.addItem(name)

    def change_figure(self, item):
        text = item.text()
        self.rm_mpl()
        self.add_mpl(self.fig_dict[text])

    def add_mpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas,
                self.mplwindow, coordinates=True)
        self.mplvl.addWidget(self.toolbar)

    def rm_mpl(self,):
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
