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
                                         CalibratedSum, PMaps
from collections import namedtuple
from enum import Enum

EventPmaps = Enum('EventPmaps', 'not_s1 not_s2 s1_not_1 s2_not_1')
DeconvParams = namedtuple('DeconvParams', 'n_baseline thr_trigger')
S12Features = namedtuple('S12Features', 's1f s2f')
CalibVectors = namedtuple('CalibVectors',
    'channel_id coeff_blr coeff_c adc_to_pes_pmt adc_to_pes_sipm pmt_active')

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
        self.event = []
        self.peak = []
        self.w    = []
        self.tmin = []
        self.tmax = []
        self.tpeak = []
        self.emax = []
        self.etot = []
        self.er   = []
        self.nm   = [] # number of matches

    def add_features(self, event, S12, peak_number=0):
        """Add event features."""

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
        self.event.append(event)
        self.peak.append(peak_number)
        self.w.append(tmax - tmin)
        self.tmin.append(tmin)
        self.tmax.append(tmax)
        self.tpeak.append(tpeak)
        self.emax.append(emax)
        self.etot.append(etot)
        self.er.append(er)

    def add_number_of_matches(self, nm):
        self.nm.append(nm)

    def __str__(self):
        w = """ (event  ={}
                 peak = {}
                 width (mus) = {}
                 tmin  (mus) = {}
                 tmax  (mus) = {}
                 tpeak (mus) = {}
                 etot (pes)  = {}
                 epeak (pes) = {}
                 er          = {})
        """.format(self.event, self.peak,
                   np.array(self.width)/units.mus,
                   np.array(self.tmin)/units.mus,
                   np.array(self.tmax)/units.mus,
                   np.array(self.tpeak)/units.mus,
                   np.array(self.etot),
                   np.array(self.epeak),
                   np.array(self.er)
                   )
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
def print_s12f(s12f):
    """Print the """

class EventPmaps:
    """Compute event pmaps

    calib_vectors : named tuple.
                    ('CalibVectors',
                    'channel_id coeff_blr coeff_c adc_to_pes pmt_active')

    deconv_params : named tuple.
                    ('DeconvParams', 'n_baseline thr_trigger')

    """

    def __init__(self, pmtrwf, sipmrwf,
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
        self._calib_vectors()
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

    @property
    def calib_vectors(self):
        return self.P


    def _calib_vectors(self):
        """Provisional fix for calib vectors"""
        channel_id = np.array([0,1,4,5,8,18,22,23,26,27,30])
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

        DataSiPM = load_db.DataSiPM()
        self.P   = CalibVectors(channel_id = channel_id,
                                coeff_blr  = coeff_blr,
                                coeff_c    = coeff_c,
                                adc_to_pes_pmt = adc_to_pes,
                                adc_to_pes_sipm = DataSiPM.adc_to_pes.values,
                                pmt_active  = pmt_active)


    def calibrated_sum(self, event):
        """Compute calibrated sums (with/out) MAU."""
        self.RWF = self.pmtrwf[event]
        self.CWF = blr.deconv_pmt(self.RWF,
                             self.P.coeff_c,
                             self.P.coeff_blr,
                             n_baseline  = self.D.n_baseline,
                             thr_trigger = self.D.thr_trigger)


        self.csum, self.csum_mau = cpf.calibrated_pmt_sum(self.CWF,
                                                self.P.adc_to_pes_pmt,
                                                pmt_active = self.P.pmt_active,
                                                n_MAU      = 100,
                                                thr_MAU    = self.thr.thr_MAU)

    def find_s1(self, event):
        """Compute S1."""
        s1_ene, s1_indx = cpf.wfzs(self.csum_mau, threshold  =self.thr.thr_s1)
        self.S1         = cpf.find_S12(s1_ene, s1_indx, **self.s1par._asdict())
        if self.verbose:
            print_s12(self.S1)

    def find_s2(self, event):
        """Compute S2."""
        s2_ene, s2_indx = cpf.wfzs(self.csum, threshold=self.thr.thr_s2)
        self.S2         = cpf.find_S12(s2_ene, s2_indx, **self.s2par._asdict())
        if self.verbose:
            print_s12(self.S2)

    def find_s2si(self, event):
        """Compute S2Si"""

        sipm = cpf.signal_sipm(self.sipmrwf[event], self.P.adc_to_pes_sipm,
                               thr=self.thr.thr_sipm, n_MAU=100)
        SIPM = cpf.select_sipm(sipm)
        self.S2Si = pf.sipm_s2_dict(SIPM, self.S2, thr=self.thr.thr_SIPM)
        if self.verbose:
            print_s2si(self.S2Si)

    def s1_features(self, event):
        """Add S1 features."""
        for i in self.S1:
            if self.verbose:
                print('S1: adding features for peak number {}'.format(i))
        self.s1f.add_features(event, self.S1, peak_number=i)

    def s2_features(self, event):
        """Add S2 features."""
        for i in self.S2:
            if self.verbose:
                print('S2: adding features for peak number {}'.format(i))
        self.s2f.add_features(event, self.S2, peak_number=i)
