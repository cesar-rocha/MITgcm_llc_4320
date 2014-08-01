# compute autocorrelation of velocity data at surface
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import scipy as sp
from scipy import signal
from scipy import io
import numpy as np
from datetime import datetime, timedelta
import glob   
import seawater.csiro as sw
import matplotlib.colors as mcolors
from netCDF4 import Dataset

plt.close('all')

## mapping
lonplot = (-85, -45)
latplot = (-67, -50)

m = Basemap(llcrnrlon=lonplot[0],llcrnrlat=latplot[0],urcrnrlon=lonplot[1],
        urcrnrlat=latplot[1],
            rsphere=(6378137.00,6356752.3142),
            resolution='i',area_thresh=1000.,projection='lcc',
            lat_1=-55,lon_0=-70)

plt.rcParams.update({'font.size': 24, 'legend.handlelength'  : 1.5
    , 'legend.markerscale': 14., 'legend.linewidth': 3.})

#
# settings
#

data = np.load('subsets/eta2video.npz')


#
# mean and relative vorticity
#
lon, lat,time = data['lon'],data['lat'], data['time']
kx = time.size
ix,jx = lon.shape
eta = np.ma.masked_array(data['eta'],np.abs(data['eta'])>4.)
etam = np.repeat(eta.mean(axis=2),kx).reshape(ix,jx,kx)

eta = eta - etam 

#
# plotting
#

xgm1,ygm1 = m(data['lon'],data['lat'])

dec=2

for i in range(time.size):

    fig=plt.figure(facecolor='w', figsize=(8.,8.5))

    sc=8.
    #Q = m.quiver(xgm1,ygm1,data1['u'].mean(axis=2),data1['v'].mean(axis=2),scale=sc,color='k',linewidth=.5)
    m.pcolormesh(xgm1,ygm1,eta[:,:,i], vmin=-1.3,vmax=1.3,cmap='Spectral_r')
    cb = plt.colorbar(extend='both',shrink=.5)
    cb.set_label(u'$\mathcal{SSH}$ [m]')
    m.fillcontinents(color='.60',lake_color='none')
    m.drawparallels(np.arange(latplot[0], latplot[1], 4),
        labels=[1, 0, 0, 0], linewidth=.5, 
        fontsize=18., fontweight='medium')
    m.drawmeridians(np.arange(lonplot[0], lonplot[1], 10),
        labels=[0, 0, 0, 1], dashes=[1,1], linewidth=.5, 
        fontsize=18., fontweight='medium')
    m.drawcoastlines()
    plt.title(str(time[i])[0:13])
    plt.savefig('figs2movie/'+str(time[i])[0:10]+str(time[i])[11:13],dpi=50)
    plt.close('all')



