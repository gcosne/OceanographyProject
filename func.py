# Usefull functions for notebooks
__author__ ='gmaze@ifremer.fr'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.cluster import KMeans
import numpy as np
import os
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import func

from scipy.linalg import qr, solve, lstsq
from scipy.stats import multivariate_normal

from numpy import *
from IPython.display import Image
from IPython.core.display import HTML
def discrete_colorbar(N, cmap, ticklabels='default'):
    """Add a colorbar with a discrete colormap.

        N: number of colors.
        cmap: colormap instance, eg. cm.jet.
    """
    # source: https://stackoverflow.com/questions/18704353/correcting-matplotlib-colorbar-ticks
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, N + 0.5)
    colorbar = plt.colorbar(mappable, shrink=0.5)
    colorbar.set_ticks(np.linspace(0, N, N))
    if 'default' in ticklabels:
        ticklabels = range(N)
    colorbar.set_ticklabels(ticklabels)
    return colorbar

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.
    """
    # source: https://stackoverflow.com/questions/18704353/correcting-matplotlib-colorbar-ticks
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
                      for i in range(N + 1)]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, N)

def plot_one_profile(surface, zeta, PCM, ds, itim, iobs):

    # And plot it:
    x = ds['lon'].isel(time=itim)
    y = ds['lat'].isel(time=itim)

    ilat = int(ds['obs_ilat'].isel(time=itim).isel(n_obs=iobs).values)-1 # Because obs_ilon is 1-based
    ilon = int(ds['obs_ilon'].isel(time=itim).isel(n_obs=iobs).values)-1 # Because obs_ilon is 1-based

    obs_label = ds['labels'].isel(time=itim).isel(longitude=ilon).isel(latitude=ilat).values
    obs_posteriors = ds['posteriors'].isel(time=itim).isel(longitude=ilon).isel(latitude=ilat).values

    # Compute robustness of the classification:
    K = 4
    robust = (np.max(obs_posteriors, axis=0) - 1./K)*K/(K-1.)
    Plist = [0, 0.33, 0.66, 0.9, .99, 1.01];
    rowl0 = ('Unlikely','As likely as not','Likely','Very Likely','Virtually certain')
    robust_id = np.digitize(robust, Plist)-1

    def show_loc():
        plt.plot(ds['obs_lon'].isel(time=itim).isel(n_obs=iobs),
                 ds['obs_lat'].isel(time=itim).isel(n_obs=iobs), 'kp', markersize=10)

    fig = plt.figure(figsize=(25, 10))
    grid = plt.GridSpec(2, 4, wspace=0.4, hspace=0.3)

    # Sea Surface Temperature Map:
    ax = plt.subplot(grid[0, 0])
    SST = ds['temperature'].isel(time=itim).isel(depth=0)
    K = 68
    cmap = cmap_discretize(plt.cm.spectral, K)
    plt.pcolormesh(x,y,SST, cmap=cmap, vmin=0,vmax=30)
    show_loc()
    plt.colorbar()
    ax.set_title('SST')

    # Sea Surface Height:
    ax = plt.subplot(grid[0, 1])
    this_time = ds['tim'].isel(time=itim).values
    itim_surface = np.argwhere(surface['tim'].values==this_time)[0][0]
    K = 68
    cmap = cmap_discretize(plt.cm.bwr, K)
    plt.pcolormesh(surface['lon'], surface['lat'], surface['SLA'].isel(time=itim_surface), 
                   cmap=cmap, vmin=-1,vmax=1)
    show_loc()
    plt.colorbar()
    ax.set_title('SLA')  

    # Profiles Classification results:
    ax = plt.subplot(grid[1, 0])
    LABELS = ds['labels'].isel(time=itim)
    K = 4
    cmap = cmap_discretize(plt.cm.viridis_r, K)
    plt.pcolormesh(x,y,LABELS, cmap=cmap, vmin=1, vmax=K)
    show_loc()
    discrete_colorbar(K, cmap, ticklabels=np.arange(1,K+1)).set_label("PCM class")
    ax.set_title('LABELS')

    # Surface vorticity
    ax = plt.subplot(grid[1, 1])
    this_time = ds['tim'].isel(time=itim).values
    itim_zeta = np.argwhere(zeta['tim'].values==this_time)[0][0]
    f = 2*2*np.pi/86400*np.sin(45*np.pi/180.)
    K = 68
    cmap = cmap_discretize(plt.cm.bwr, K)
    plt.pcolormesh(zeta['lon'], zeta['lat'], zeta['ZETA'].isel(time=itim_zeta)/f, 
                   cmap=cmap, vmin=-0.5, vmax=0.5)
    show_loc()
    plt.colorbar()
    ax.set_title('VORTICITY/F') 

    # Class mean structure
    ax = plt.subplot(grid[0:, 2])
    K = 4
    cmap = cmap_discretize(plt.cm.viridis_r, K)
    for k in range(K):
        plt.plot(PCM['MU'+str(int(k+1))],ds['dpt'].isel(time=itim), color=cmap(k), linewidth=2, label= ("Class %i")%(obs_label))
        plt.plot(PCM['MU'+str(int(k+1))]-PCM['SI'+str(int(k+1))],ds['dpt'].isel(time=itim), color=cmap(k), linewidth=1)
        plt.plot(PCM['MU'+str(int(k+1))]+PCM['SI'+str(int(k+1))],ds['dpt'].isel(time=itim), color=cmap(k), linewidth=1)
    ax.legend()
    ax.grid('on')
    ax.set_title('Class mean/std')

    # Profile:
    ax = plt.subplot(grid[0:, 3])
    TEMP = ds['temperature'].isel(time=itim).isel(longitude=ilon).isel(latitude=ilat)
    plt.plot(PCM['MU'+str(int(obs_label))],ds['dpt'].isel(time=itim), color=cmap(obs_label-1), linewidth=2)
    plt.plot(PCM['MU'+str(int(obs_label))]-PCM['SI'+str(int(obs_label))],ds['dpt'].isel(time=itim), color=cmap(obs_label-1), linewidth=1)
    plt.plot(PCM['MU'+str(int(obs_label))]+PCM['SI'+str(int(obs_label))],ds['dpt'].isel(time=itim), color=cmap(obs_label-1), linewidth=1)
    plt.plot(TEMP, ds['dpt'].isel(time=itim), color='r', linewidth=3)
    rob_txt = ("Robustness: %0.2f%%, i.e. \"%s\"")%(round(robust*100,2), rowl0[robust_id])
    ax.set_title( ("TEMPERATURE Profile\n Classified with label #%i\n%s")%(obs_label,rob_txt) )
    ax.grid('on')
    
    # Output
    return fig