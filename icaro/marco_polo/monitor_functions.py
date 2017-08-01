import tables as tb
import numpy  as np
import matplotlib.pyplot as plt
from invisible_cities.core.system_of_units_c import units

def sample_events(n_sample_events, rwf_path, all_events=False):
    ev_sample = []
    with tb.open_file(rwf_path, 'r+') as f_rwfs:
        events = [f_rwfs.root.Run.events[i][0] for i in range(len(f_rwfs.root.Run.events[:]))]
        if all_events: return events
        ev_sample = np.random.choice(events[1:-1], size=n_sample_events-2, replace=False)
        ev_sample = np.append(np.insert(ev_sample, 0, events[0]), events[-1])
        ev_sample.sort()
        return ev_sample
    
def calibrated_waveforms(events, irene, rwf_path, pmts=True, sipms=True):
    with tb.open_file(rwf_path ,'r') as f:
        CSUMs  = {}
        csipm  = {}
        for ev in events:
            if pmts : 
                CSUMs[ev] = irene.calibrated_pmt_sum(irene.deconv_pmt(f.root.RD.pmtrwf[ev]))[0]
                
                
            if sipms: csipm[ev] = irene.calibrated_signal_sipm(f.root.RD.sipmrwf[ev]).sum(axis=0)
    return CSUMs, csipm

def plot_wf_vs_s12_peak(csum, peak,
                        subplot = -1, 
                        wfbs    = 25*units.ns,
                        pkbs    = 25*units.ns,
                        S12     = 'S1', 
                        plotwf  = plt.scatter, 
                        plotp   = plt.scatter,
                        pc      = 'r'):
    
    if subplot != -1: plt.subplot(subplot)
        
    # helpful since rebinning algorithm record average time 
    t = np.copy(peak.t)
    if S12=='S2Si': t-= 500  *units.ns # typically  1mus time bins 
        
    plotwf(np.arange(len(csum))*wfbs/units.mus, csum, label='wf', s=300, alpha=.3) # Plot wf
    plotp(t/units.mus, peak.E *wfbs/pkbs, label=S12, alpha=.5, c=pc)              # Plot peak
    plt.xlim((peak.t[0] - 2*pkbs) / units.mus, (t[-1] + 2*pkbs)/units.mus) # Set xlim around peak
    plt.ylim(0, peak.E.max()*wfbs/pkbs + .25*peak.E.max()*wfbs/pkbs)      # Set ylim around peak
    plt.legend()
    plt.grid(True)
    plt.ylabel('pes')
    plt.xlabel('time (microseconds)')