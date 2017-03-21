"""
Compute PMAPS and PMAP features
"""
import sys
import numpy as np

from   invisible_cities.core.system_of_units_c import units
from invisible_cities.core.core_functions import loc_elem_1d
from collections import namedtuple

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


def event_pmaps(P, pmtrwf, sipmrwf, s1par, thr, s12f, event=0):
    """Compute Event pmaps and pmaps features.
    input:
    P           : CalibVectors named tuple.
                  ('CalibVectors',
                  'channel_id coeff_blr coeff_c adc_to_pes pmt_active')
    pmtrwf      : raw waveform for pmts
    sipmrwf     : raw waveform for SiPMs
    thr         : threshold for PMAP searches.
                  ('ThresholdParams',
                  'thr_s1 thr_s2 thr_MAU thr_sipm thr_SIPM')
    s12f        : instance of a S12F class (s12 features)
    event       : event number
    """

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
                                                  threshold=thr.thr_s1)
    S1                   =  cpf.find_S12(         s1_ene,
                                                  s1_indx,
                                                  **s1par._asdict())
    for i in S1:
        add_features(self, S12, peak_number=0)
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
