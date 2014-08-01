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
lonplot = (-70, -52)
latplot = (-67, -50)

m = Basemap(llcrnrlon=lonplot[0],llcrnrlat=latplot[0],urcrnrlon=lonplot[1],
        urcrnrlat=latplot[1],
            rsphere=(6378137.00,6356752.3142),
            resolution='i',area_thresh=1000.,projection='lcc',
            lat_1=-55,lon_0=-70)

plt.rcParams.update({'font.size': 24, 'legend.handlelength'  : 1.5
    , 'legend.markerscale': 14., 'legend.linewidth': 3.})

cl1="#e6e6fa"
cl12="#483d8b"
cl2 = "#f08080"
cl22 = "#a52a2a"

#
# settings
#

data = np.load('subsets/UV2movie.npz')
aux = np.load('subsets/eta_919_1440.npz')
time = aux['time']

dec = 6
xg1,yg1 = m(data['lon'][0::dec,0::dec],data['lat'][0::dec,0::dec])
xg2,yg2 = m(data['lonz'][0::dec,0::dec],data['latz'][0::dec,0::dec])


zeta = data['zeta'] # divide by grid resol. 
zeta = np.ma.masked_array(zeta,np.abs(zeta)>2.5)


#
# plotting
#

for i in range(data['U'].shape[2]):
    # mean currents
    fig=plt.figure(facecolor='w', figsize=(8.,8.5))
    
    sc=8.
    
    m.pcolormesh(xg2,yg2,zeta[:,:,i], vmin=-2.5,vmax=2.5,cmap='bwr')
    cb = plt.colorbar(extend='both')
    cb.set_label(u'$\zeta/|f|$')
    Q = m.quiver(xg1[0::dec,0::dec],yg1[0::dec,0::dec],data['U'][0::dec,0::dec,i],data['V'][0::dec,0::dec,i],scale=sc,color='m',linewidth=.5)

    # quiver legend
    qk = plt.quiverkey(Q, .2, .93, 1., '1.0  m/s', fontproperties={'weight': 'bold','size': '18'}, labelcolor='k')

    m.fillcontinents(color='.60',lake_color='none')
    m.drawparallels(np.arange(latplot[0], latplot[1], 4),
        labels=[1, 0, 0, 0], linewidth=.5,
        fontsize=18., fontweight='medium')
    m.drawmeridians(np.arange(lonplot[0], lonplot[1], 7),
        labels=[0, 0, 0, 1], dashes=[1,1], linewidth=.5,
        fontsize=18., fontweight='medium')
    m.drawcoastlines()

    plt.title(str(time[i]))
    plt.savefig('figs2movie/'+str(time[i])[0:10]+str(time[i])[11:13],dpi=150)
    plt.close('all')
