"""
Compute PMAPS and PMAP features
"""
import sys
import numpy as np

from invisible_cities.database import load_db
import invisible_cities.sierpe.blr as blr
import invisible_cities.reco.peak_functions_c as cpf
import invisible_cities.reco.peak_functions as pf
from   invisible_cities.core.system_of_units_c import units
from invisible_cities.core.core_functions import loc_elem_1d
from invisible_cities.reco.params import S12Params, ThresholdParams,\
                CalibratedSum, PMaps, CalibVectors, DeconvParams
from collections import namedtuple
from enum import Enum

KrConditions = Enum('KrConditions',
  'csum_is_zero s1_multiplicity s2_multiplicity si_multiplicity')
KrSelection = namedtuple('KrSelection',
               's1_multiplicity s2_multiplicity si_multiplicity')

class KrBox:
    """Container of krypton events"""
    def __init__(self, run_number):
        self.run_number = run_number
        self.event_ = []
        self.s1f_ = S12F()
        self.s2f_ = S12F()
        self.qs1_ = []
        self.qs2_ = []
        self.drift_time_ = []
        self.X_ = []
        self.Y_ = []
        self.Z_ = []
        self.R_ = []
        self.Phi_ = []

    def add_position(self, X, Y, Z, R, Phi, drift_time):
        self.X_.append(X)
        self.Y_.append(Y)
        self.Z_.append(Z)
        self.R_.append(R)
        self.Phi_.append(Phi)
        self.drift_time_.append(drift_time)

    def event(self):
        return np.array(self.event_)
    def s1f(self):
        return np.array(self.s1f)
    def s2f(self):
        return np.array(self.s2f)
    def qs1(self):
        return np.array(self.qs1)
    def qs2(self):
        return np.array(self.qs2)
    def drift_time(self):
        return np.array(self.drift_time_)
    def X(self):
        return np.array(self.X_)
    def Y(self):
        return np.array(self.Y_)
    def Z(self):
        return np.array(self.Z_)
    def R(self):
        return np.array(self.R_)
    def Phi(self):
        return np.array(self.Phi_)


class S12F:
    """
    Defines the global features of an S12 peak, namely:
    1) peak start (tmin), end (tmax) and width
    2) peak maximum (both energy and time)
    3) energy total
    4) ratio peak/total energy
    """

    def __init__(self):
        """Define event lists."""
        self.event_ = []
        self.peak_ = []
        self.width_ = []
        self.tmin_ = []
        self.tmax_ = []
        self.tpeak_ = []
        self.emax_ = []
        self.etot_ = []
        self.er_   = []

    def add_features(self, event, S12, peak_number=0):
        """Add event features."""

        t = S12[peak_number][0]
        E = S12[peak_number][1]

        emax = np.max(E)
        etot = np.sum(E)
        er = 9e+9
        if etot > 0:
            er = emax/etot

        tmin = t[0]
        tmax = t[-1]
        i_t = loc_elem_1d(E, emax)
        tpeak = t[i_t]

        self.event_.append(event)
        self.peak_.append(peak_number)
        self.width_.append(tmax - tmin)
        self.tmin_.append(tmin)
        self.tmax_.append(tmax)
        self.tpeak_.append(tpeak)
        self.emax_.append(emax)
        self.etot_.append(etot)
        self.er_.append(er)

    def event(self):
        return np.array(self.event_)
    def peak(self):
        return np.array(self.peak_)
    def width(self):
        return np.array(self.width_)
    def tpeak(self):
        return np.array(self.tpeak_)
    def tmin(self):
        return np.array(self.tmin_)
    def tmax(self):
        return np.array(self.tmax_)
    def etot(self):
        return np.array(self.etot_)
    def emax(self):
        return np.array(self.emax_)
    def emax_over_etot(self):
        return np.array(self.er_)

    def __str__(self):
        w = """ (event  ={}
                 peak = {}
                 width (mus) = {}
                 tmin  (mus) = {}
                 tmax  (mus) = {}
                 tpeak (mus) = {}
                 etot (pes)  = {}
                 emax (pes)  = {}
                 er          = {})
        """.format(self.event(),
                   self.peak(),
                   self.width()/units.mus,
                   self.tmin()/units.mus,
                   self.tmax()/units.mus,
                   self.tpeak()/units.mus,
                   self.etot(),
                   self.emax(),
                   self.er())
        return w
    def __repr__(self):
        return self.__str__()

def compare_S1(S1, PMT_S1, peak=0, tol=0.5*units.mus):
    """Compare sum S1 with S1 in individual PMT

    input:
    S1 computed with the sum
    PMT_S1 computed with individual PMTs.
    tol is the matching tolerance.

    Return number of matches

    """
    n_match_s1 = 0
    t = S1[peak][0]
    E = S1[peak][1]
    for pmt in PMT_S1:
        if len (PMT_S1[pmt]) > 0:
            for peak, (t2,E2) in PMT_S1[pmt].items():
                diff = abs(t2[0] - t[0])
                if diff < tol:
                    n_match_s1 +=1
                    break  # if one peak is matched look no further
    return n_match_s1


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

def print_s2si(S2Si):
    """Scan the S2Si objects."""
    for peak, sipm_set in S2Si.items():
        print('S2Si for peak number = {}'.format(peak))
        for sipm, e_array in sipm_set.items():
            print('sipm number = {}, energy = {}'.format(sipm,
                                                         np.sum(e_array)))

class EventPmaps:
    """Compute event pmaps

    calib_vectors : named tuple.
                    ('CalibVectors',
                    'channel_id coeff_blr coeff_c adc_to_pes pmt_active')

    deconv_params : named tuple.
                    ('DeconvParams', 'n_baseline thr_trigger')

    """

    def __init__(self, run_number, pmtrwf, sipmrwf,
                 s1par, s2par, thr,
                 verbose=True):
        """
        input:

        pmtrwf        : raw waveform for pmts
        sipmrwf       : raw waveform for SiPMs
        s1par, s2par  : named tuples
                            ('S12Params' , 'tmin tmax stride lmin lmax rebin')
        thr           : named tuple.
                          ('ThresholdParams',
                          'thr_s1 thr_s2 thr_MAU thr_sipm thr_SIPM')
        verbose       : to make it talk.
        """
        self.run_number = run_number
        DataPMT = load_db.DataPMT(run_number)
        DataSiPM = load_db.DataSiPM(run_number)
        self.xs = DataSiPM.X.values
        self.ys = DataSiPM.Y.values


        self.P = CalibVectors(channel_id = DataPMT.ChannelID.values,
                              coeff_blr = abs(DataPMT.coeff_blr   .values),
                              coeff_c = abs(DataPMT.coeff_c   .values),
                              adc_to_pes = abs(DataPMT.adc_to_pes.values),
                              adc_to_pes_sipm = abs(DataSiPM.adc_to_pes.values),
                              pmt_active = np.nonzero(
                                           DataPMT.Active.values)[0].tolist())

        #self._calib_vectors()
        self.pmtrwf        = pmtrwf
        self.sipmrwf       = sipmrwf
        self.D             = DeconvParams(n_baseline = 48000,
                                          thr_trigger = 5)
        self.s1par         = s1par
        self.s2par         = s2par
        self.thr           = thr

        # instances of s12fF
        self.s1f = S12F()
        self.s2f = S12F()

        self.verbose = verbose

    def calibrated_sum(self, event):
        """Compute calibrated sums (with/out) MAU."""
        self.RWF = self.pmtrwf[event]

        self.CWF = blr.deconv_pmt(self.RWF,
                             self.P.coeff_c,
                             self.P.coeff_blr,
                             self.P.pmt_active,
                             n_baseline  = self.D.n_baseline,
                             thr_trigger = self.D.thr_trigger)


        self.csum, self.csum_mau = cpf.calibrated_pmt_sum(self.CWF,
                                                self.P.adc_to_pes,
                                                pmt_active = self.P.pmt_active,
                                                n_MAU      = 100,
                                                thr_MAU    = self.thr.thr_MAU)
        return np.sum(csum)

    def find_s1(self, event):
        """Compute S1."""
        s1_ene, s1_indx = cpf.wfzs(self.csum_mau, threshold  =self.thr.thr_s1)
        self.S1         = cpf.find_S12(s1_ene, s1_indx, **self.s1par._asdict())
        for peak in self.S1:
            self.s1f.add_features(event, self.S1, peak_number=peak)
        if self.verbose:
            print_s12(self.S1)

        return len(S1)

    def find_s2(self, event):
        """Compute S2."""
        s2_ene, s2_indx = cpf.wfzs(self.csum, threshold=self.thr.thr_s2)
        self.S2         = cpf.find_S12(s2_ene, s2_indx, **self.s2par._asdict())
        for peak in self.S2:
            self.s2f.add_features(event, self.S2, peak_number=peak)
        if self.verbose:
            print_s12(self.S2)

        return len(S2)

    def find_s2si(self, event):
        """Compute S2Si"""

        sipm = cpf.signal_sipm(self.sipmrwf[event], self.P.adc_to_pes_sipm,
                               thr=self.thr.thr_sipm, n_MAU=100)
        SIPM = cpf.select_sipm(sipm)
        self.S2Si = pf.sipm_s2_dict(SIPM, self.S2, thr=self.thr.thr_SIPM)
        if self.verbose:
            print_s2si(self.S2Si)

        return len(S2Si)

    def charge_and_position(self, peak_number=0):
        """
        Charge and position from S2Si
        """
        s2si = self.S2Si[peak_number]
        xsipm = []
        ysipm = []
        Q = []
        for key, value in s2si.items():
            xsipm.append(xs[key])
            ysipm.append(ys[key])
            Q.append(np.sum(value))
        return np.array(xsipm), np.array(ysipm), np.array(Q)
