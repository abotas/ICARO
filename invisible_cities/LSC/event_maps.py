import sys
import numpy as np

from PyQt5 import QtCore, QtWidgets, uic
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from PyQt5.uic import loadUiType
from collections import namedtuple
import invisible_cities.core.mpl_functions as mpl
from invisible_cities.core.core_functions import define_window
from invisible_cities.core.core_functions import loc_elem_1d
from   invisible_cities.core.system_of_units_c import units
#

#S12F = namedtuple('S12F','tmin tmax tpeak w etot emax er')

class S12F:
    """
    Defines the global features of an S12 peak, namely:
    1) peak start (tmin), end (tmax) and width
    2) peak maximum (both energy and time)
    3) energy total
    4) ratio peak/total energy
    """

    def __init__(self):
        self.w    = []
        self.tmin = []
        self.tmax = []
        self.tpeak = []
        self.emax = []
        self.etot = []
        self.er   = []

    def add_features(self, S12, peak_number=0):
        t = S12[peak_number][0]
        E = S12[peak_number][1]

        tmin = t[0]
        tmax = t[-1]
        i_t = loc_elem_1d(E, emax)
        tpeak = t[i_t]

        emax = np.max(E)
        etot = np.sum(E)
        er = 9e+9
        if etot > 0:
            er = emax/etot

        self.w.append(tmax - tmin)
        self.tmin.append(tmin)
        self.tmax.append(tmax)
        self.tpeak.append(tpeak)
        self.emax.append(emax)
        self.etot.append(etot)
        self.etot.append(er)


def pmt_calib_vectors():
    """Provisional fix for calib vectors"""
    channel_id = np.array([0,1,4,5,8,18,22,23,26,27,31])
    coeff_blr = np.array([1.61,1.62,1.61,1.61,1.61,
                      0.8,0.8,0.8,0.8,0.8,1.60,
                      1.0]) * 0.001
    coeff_c = np.array([2.94,2.75,3.09,2.81,2.88,
                    1.,1.,1.,1.,1.,2.76,
                    1.0]) * 1e-6
    adc_to_pes = np.array([25.17,22.15,33.57,23.88,21.55,
                       26.49,25.39,27.74,23.78,20.83,26.56,
                       0.])
    pmt_active = list(range(11))

    CalibVectors = namedtuple('CalibVectors',
                    'channel_id coeff_blr coeff_c adc_to_pes pmt_active')
    return CalibVectors(channel_id = channel_id,
                        coeff_blr  = coeff_blr,
                        coeff_c    = coeff_c,
                        adc_to_pes = adc_to_pes,
                        pmt_active  = pmt_active)


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

    (bins, n) = _hist_outline(data)
    #xlo = -max(abs(bins))
    #xhi = max(abs(bins))
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
        mpl.set_plot_labels(xlabel="samples", ylabel="adc")
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
    j=0
    for i in pmt_active:
        ax1 = fig.add_subplot(3, 4, j+1)
        ax1.set_xlim([t_min, t_max])
        ax1.set_ylim([signal_min, signal_max])
        mpl.set_plot_labels(xlabel = "t (mus)",
                        ylabel = "signal (pes/adc)")

        ax1.plot(signal_t, pmt_signals[i])
        j+=1
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
    mpl.set_plot_labels(xlabel = "t (mus)",
                    ylabel = "signal (pes/adc)")
    ax1.plot(signal_t, signal)
    return fig


def print_s12(S12):
    """Print peaks of input S12.

    S12 is a dictionary
    S12[i] for i in keys() are the S12 peaks
    """
    print('number of peaks = {}'.format(len(S12)))
    for i in S12:
        print('S12 number = {}, samples = {} sum in pes ={}'
              .format(i, len(S12[i][0]), np.sum(S12[i][1])))
        print('time vector (mus) = {}'.format(S12[i][0]/units.mus))
        print('energy vector (pes) = {}'.format(S12[i][1]/units.pes))


def scan_s12(S12):

    S12t = []
    S12l = []
    S12e = []
    for i in S12.keys():
        S12t.append(S12[i][0][0])
        S12l.append(len(S12[i][0])),
        S12e.append(np.sum(S12[i][1]))
    return np.array(S12t), np.array(S12l), np.array(S12e)

def plot_s12(S12):
    """Plot the peaks of input S12.

    S12 is a dictionary
    S12[i] for i in keys() are the S12 peaks
    """
    fig = Figure(figsize=(12, 12))
    mpl.set_plot_labels(xlabel = "t (mus)",
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

def compare_S1(S1, PMT_S1):
    n_match_s1 = 0
    t = S1[0]
    E = S1[1]
    for pmt in PMT_S1:
        if len (PMT_S1[pmt]) > 0:
            for peak, (t2,E2) in PMT_S1[pmt].items():
                diff = abs(t2[0] - t[0])
                #print('for pmt = {}, peak = {}'.format(pmt, peak))
                #print('PMT_ S1 t (mus) = {}'.format(t2/units.mus))
                #print('diff (mus)= {}'.format(diff/units.mus))
                if diff < 1*units.mus:
                    #print('found mach between S1 and S1_PMT peak = {}'.\
                    #         format(peak))
                    n_match_s1 +=1
                    break
    return n_match_s1


def event_maps(P, pmtrwf, sipmrwf, s1par, thr_s1=0.3*units.pes, event=0):
    """ Events maps"""
    RWF = pmtrwf[event]
    CWF                  = blr.deconv_pmt(        RWF, P.coeff_c, P.coeff_blr,
                                                  n_baseline=48000,
                                                  thr_trigger=5)
    CAL_PMT, CAL_PMT_MAU = cpf.calibrated_pmt_mau(CWF,
                                                  P.adc_to_pes,
                                                  pmt_active = P.pmt_active,
                                                  n_MAU = 100,
                                                  thr_MAU =  3)


    csum, csum_mau       = cpf.calibrated_pmt_sum(CWF,
                                                  P.adc_to_pes,
                                                  P.pmt_active)

    s1_ene, s1_indx      = cpf.wfzs(              csum_mau,
                                                  threshold=thr_s1)
    S1                   =  cpf.find_S12(         s1_ene,
                                                  s1_indx,
                                                  **s1par._asdict())
    for i in S1:
        print('S1 number {}'.format(i))
        s1f = features_s12(S1[i])
        print('S1 features = {}'.format(s1f))
        t = S1[i][0]
        tmin = t[0] - 100*units.ns
        tmax = t[-1] + 100*units.ns

        #print('S1 match region: [tmin:tmax] (mus) = {}:{}'.format(tmin/units.mus,tmax/units.mus))
        s1par_PMT = S12Params(tmin=tmin, tmax=tmax, lmin=3, lmax=20, stride=4, rebin=False)

        PMT_S1 = {}
        for pmt in P.pmt_active:
            s1_ene, s1_indx = cpf.wfzs(CAL_PMT_MAU[pmt], threshold=0.1)
            PMT_S1[pmt] = cpf.find_S12(s1_ene, s1_indx, **s1par_PMT._asdict())
        nm = compare_S1(S1[i], PMT_S1)
        print('number of PMT matches = {}'.format(nm))

    s2_par = S12Params(tmin=640*units.mus, tmax=800*units.mus, stride=40,
    lmin=80, lmax=20000, rebin=True)
    s2_ene, s2_indx = cpf.wfzs(csum, threshold=1.0)
    S2    = cpf.find_S12(s2_ene, s2_indx, **s2_par._asdict())

    print_s12(S2)
    if len(S2) == 0:
        return 0

    s2f = features_s12(S2[0])
    print('S2 features = {}'.format(s2f))
    t = S2[0][0]
    tmin = t[-1] + 1*units.mus
    print('S1p search starts at {}'.format(tmin))
    s1p_params = S12Params(tmin=tmin, tmax=1300*units.mus, stride=4, lmin=4,
     lmax=20, rebin=False)
    s1_ene, s1_indx = cpf.wfzs(csum_mau, threshold=0.1)
    S1p =  cpf.find_S12(s1_ene, s1_indx, **s1p_params._asdict())
    S12t, S12l, S12e = scan_s12(S1p)

    if(len(S1) == 1 and len(S2) == 1):
        dt = s2f.tpeak - s1f.tpeak

        print('drif time = {} mus'.format(dt/units.mus))

    print('***S2Si****')
    sipm = cpf.signal_sipm(sipmrwf[event], adc_to_pes_sipm, thr=3.5*units.pes, n_MAU=100)
    SIPM = cpf.select_sipm(sipm)
    S2Si = pf.sipm_s2_dict(SIPM, S2, thr=30*units.pes)


    main.add_figure('RWF', plot_pmt_waveforms(RWF, zoom=False, window_size=10000))
    main.add_figure('CWF_vs_time_mus',
                plot_pmt_signals_vs_time_mus(CWF,
                                             P.pmt_active,
                                             t_min      =    0,
                                             t_max      = 1200,
                                             signal_min =    -5,
                                             signal_max =  200))
    main.add_figure('Calibrated_PMTs',
                plot_pmt_signals_vs_time_mus(CAL_PMT,
                                             P.pmt_active,
                                             t_min      = 400,
                                             t_max      = 800,
                                             signal_min =  -2,
                                             signal_max =  10))

    main.add_figure('PMT_rms',plot_xy(P.channel_id, rms))
    main.add_figure('Calibrated_SUM',
                plot_signal_vs_time_mus(csum, t_min=0, t_max=1300, signal_min=-5, signal_max=60))
    main.add_figure('Calibrated_SUM_S1',
                plot_signal_vs_time_mus(csum, t_min=0, t_max=640, signal_min=-2, signal_max=10))
    main.add_figure('Calibrated_SUM_S2',
                plot_signal_vs_time_mus(csum, t_min=640, t_max=660, signal_min=-2, signal_max=100))
    main.add_figure('Calibrated_PMT_S1',
                plot_pmt_signals_vs_time_mus(CAL_PMT,
                                             P.pmt_active,
                                             t_min      = 0,
                                             t_max      = 640,
                                             signal_min =  -2,
                                             signal_max =  2))
    if len(S1) > 0:
        main.add_figure('S1', plot_s12(S1))
    if len(S2) > 0:
        main.add_figure('S2', plot_s12(S2))
    main.add_figure('S1p_t',hist_1d(S12t, xlo=650*units.mus, xhi=1300*units.mus))
    main.add_figure('S1p_l',hist_1d(S12l, xlo=0, xhi=20))
    main.add_figure('S1p_e',hist_1d(S12e, xlo=0, xhi=10))
    pmf.scan_s2si_map(S2Si)


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
