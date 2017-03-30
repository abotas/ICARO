import numpy as np
from matplotlib import pyplot as plt
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
    plt.show()
