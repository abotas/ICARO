"""
Define utility functions for S12 objects
"""

import sys
import numpy as np
from   invisible_cities.core.system_of_units_c import units


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
