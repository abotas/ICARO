import tables as tb
import numpy  as np

def sample_events(n_sample_events, rwf_path, all_events=False):
    ev_sample = []
    with tb.open_file(rwf_path, 'r+') as f_rwfs:
        events = [f_rwfs.root.Run.events[i][0] for i in range(len(f_rwfs.root.Run.events[:]))]
        if all_events: return events
        ev_sample = np.random.choice(events[1:-1], size=n_sample_events-2, replace=False)
        ev_sample = np.append(np.insert(ev_sample, 0, events[0]), events[-1])
        ev_sample.sort()
        return ev_sample