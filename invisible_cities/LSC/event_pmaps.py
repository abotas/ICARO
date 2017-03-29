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
from kr_base import S12F
from s12_functions import  print_s12, print_s2si, compare_S1, compare_S1_ext

KrConditions = Enum('KrConditions',
  'csum_is_zero s1_multiplicity s2_multiplicity si_multiplicity')
KrSelection = namedtuple('KrSelection',
               's1_multiplicity s2_multiplicity si_multiplicity')


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
        
        self.CAL_PMT, self.CAL_PMT_MAU = cpf.calibrated_pmt_mau(
            self.CWF,
            self.P.adc_to_pes,
            pmt_active = self.P.pmt_active,
            n_MAU   = 100,
            thr_MAU =   3)

        self.csum, self.csum_mau = cpf.calibrated_pmt_sum(self.CWF,
                                                self.P.adc_to_pes,
                                                pmt_active = self.P.pmt_active,
                                                n_MAU      = 100,
                                                thr_MAU    = self.thr.thr_MAU)
        return np.sum(self.csum)       
        
        


    def find_s1(self, event):
        """Compute S1."""
        s1_ene, s1_indx = cpf.wfzs(self.csum_mau, threshold  =self.thr.thr_s1)
        self.S1         = cpf.find_S12(s1_ene, s1_indx, **self.s1par._asdict())
        for peak in self.S1:
            self.s1f.add_features(event, self.S1, peak_number=peak)
        if self.verbose:
            print_s12(self.S1)

        return len(self.S1)

    def find_s2(self, event):
        """Compute S2."""
        s2_ene, s2_indx = cpf.wfzs(self.csum, threshold=self.thr.thr_s2)
        self.S2         = cpf.find_S12(s2_ene, s2_indx, **self.s2par._asdict())
        for peak in self.S2:
            self.s2f.add_features(event, self.S2, peak_number=peak)
        if self.verbose:
            print_s12(self.S2)

        return len(self.S2)

    def find_s2si(self, event):
        """Compute S2Si"""

        sipm = cpf.signal_sipm(self.sipmrwf[event], self.P.adc_to_pes_sipm,
                               thr=self.thr.thr_sipm, n_MAU=100)
        SIPM = cpf.select_sipm(sipm)
        self.S2Si = pf.sipm_s2_dict(SIPM, self.S2, thr=self.thr.thr_SIPM)
        if self.verbose:
            print_s2si(self.S2Si)

        return len(self.S2Si)

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
