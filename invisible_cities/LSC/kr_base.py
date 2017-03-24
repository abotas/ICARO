import tables as tb
import invisible_cities.reco.tbl_functions as tbl
import invisible_cities.reco.nh5 as table_formats
from invisible_cities.reco.pmap_io import event_writer, run_writer,
     _make_run_event_tables
from collections import namedtuple

class S12Ft(tb.IsDescription):
    """Store for a S12F
    The table maps a S12F:
    peak is the index of the S12 dictionary, running over the number of peaks found
    time and energy of the peak.
    """

    event  = tb.  Int32Col(pos=0)
    peak   = tb.  UInt8Col(pos=2) # peak number
    tmin   = tb.Float32Col(pos=3) # time in ns
    tmax   = tb.Float32Col(pos=4) # time in ns
    tpeak   = tb.Float32Col(pos=5) # time in ns
    etot    = tb.Float32Col(pos=6) # energy in pes
    epeak    = tb.Float32Col(pos=7) # energy in pes


S12Features = namedtuple('S12Features',
                       'tmin tmax tpeak etot epeak')

class kr_writer:

    def __init__(self, filename, compression = 'ZLIB4'):
        self._hdf5_file = tb.open_file(filename, 'w')
        self._run_tables = _make_run_event_tables(self._hdf5_file,
                                                          compression)
        self._s12f_tables = _make_s12_tables(     self._hdf5_file,
                                                          compression)

    def __call__(self, run_number, event_number, timestamp, s12f):
        s12f.store(self._s12f_table, event_number)
        run_writer(self._run_tables[0], run_number)
        event_writer(self._run_tables[1], event_number, timestamp)

    def close(self):
        self._hdf5_file.close()

    @property
    def file(self):
        return self._hdf5_file

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _make_s12f_tables(hdf5_file, compression):

    c = tbl.filters(compression)

    s12f_group  = hdf5_file.create_group(hdf5_file.root, 'S12F')

    MKT = hdf5_file.create_table
    s12f_table = MKT(s12f_group, 'S12F'  ,  S12Ft, "S12F Table", c)
    s12f_table.cols.event.create_index()

    return s12f_table


class S12F(dict):
    """
    Defines the global features of an S12 peak.
    peak: peak number
    peak goes from tmin to tmax, maximum in tpeak
    width: tmax - tmin
    etot: integrated energy
    emax: energy in tpeak
    er:   emax/etot
    S12F --> {peak:namedtuple('S12Features','tmin tmax tpeak etot epeak')}
    """

    def store(self, table, event_number):
        row = table.row
        for peak_number, s12f in self.s12f.items():
            row["event"] = event_number
            row["peak"]  =  peak_number
            row["tmin"]  = s12f.tmin
            row["tmax"]  = s12f.tmax
            row["tpeak"]  = s12f.tpeak
            row["etot"]  = s12f.etot
            row["epeak"]   = s12f.epeak
            row.append()

    def add_features(self, event, S12, peak_number=0):
        """Add event features."""

        t = S12[peak_number][0]
        E = S12[peak_number][1]

        epeak = np.max(E)
        i_t = loc_elem_1d(E, epeak)
        tpeak = t[i_t]

        self[peak_number] = S12Features(tmin  = t[0],
                                        tmax  = t[-1],
                                        tpeak = tpeak,
                                        etot  = np.sum(E),
                                        epeak = epeak)
