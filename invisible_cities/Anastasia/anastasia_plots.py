import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from mpl_toolkits.mplot3d import Axes3D

def plot_nexus_event_hits(hits):
    fig = plt.figure()
    fig.set_figheight(35.0)
    fig.set_figwidth(40.0)
    c = hits[:,3] / hits[:,3].max()
    ax = fig.add_subplot(211, projection='3d')
    s1=ax.scatter(hits[:,0], hits[:,1], hits[:,2], c=c, cmap=plt.get_cmap('rainbow'),
                   vmin=0,
                   vmax=c.max(),
                   s=50,
                   edgecolors='None')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
  
    
def plot_box(ax, hrb):
    ax.add_patch(Rectangle(
        (hrb.x_min - hrb.x_pitch / 2.,  hrb.y_min - hrb.y_pitch / 2.), # (x,y)
         hrb.x_max - hrb.x_min + hrb.x_pitch, # width
         hrb.y_max - hrb.y_min + hrb.y_pitch, # height
         fill=None, alpha=1, ls='dotted'))
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    
    
def plot_SiPM_map(ax, plane, resp, minc=.2, normalize=True):
    """
    Plots an SiPM map
    xarr is a NEW sipm map, yarr the pair of coordinates the map corresponds to
    """
    if normalize:
        probs = (resp - np.min(resp))
        if probs.max() > 0: probs /= probs.max()
    else: probs = resp

    for i, x in enumerate(plane.x_pos):
        for j, y in enumerate(plane.y_pos):
            r = Ellipse(xy=(x, y), width=2., height=2.)
            r.set_facecolor('0')
            r.set_alpha(probs[i, j]*(1-minc) + minc)
            ax.add_artist(r)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

def scatter_particles_in_time_bin(particles, 
                                  zp, 
                                  z_pitch, 
                                  t_el, 
                                  magnitude=None, 
                                  c='b', s=2, a=.3, label=None):
    """ scatter particles that fall in map at zp. particles must have x,y,t coordinates """
    #if type(magnitude) == type(None): magnitude = np.ones((len(particles)))
    if magnitude is None: magnitude = np.ones((len(particles)))
    for p, m in zip(particles, magnitude):
        if (p[2] < zp + z_pitch and 
            p[2] > zp - t_el): 
            plt.scatter(p[0], p[1], color=c, s=s, alpha=a*m, label=label)  
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
            
            