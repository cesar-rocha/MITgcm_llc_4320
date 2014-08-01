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

data = np.load('subsets/surface_uvwts_0_90_919_768_.npz')
data5 = np.load('subsets/eta_919_1440.npz')


#
# mean and relative vorticity
#
lon, lat = data['lon'],data['lat']
time = data5['time']

dx = 1.14*1.e3   # [km]
dy = 1.14*1.e3
dt = 1.          # [hour]

# compute daily average

time = time[0:-21]
u = data['u'][:,:,0:-21]
v = data['v'][:,:,0:-21]

ix,jx,kx = u.shape

ud = np.zeros((ix,jx,kx/24))
vd = np.copy(ud)


k = 0
for i in range(0,time.size,24):
    ud[:,:,k] = u[:,:,i:i+24].mean(axis=2)
    vd[:,:,k] = v[:,:,i:i+24].mean(axis=2)
    if i == 0:
        timei = datetime(time[i].year,time[i].month,time[i].day)
    else:
        timei = np.append(timei,datetime(time[i].year,time[i].month,time[i].day))
    k = k+1

umy,umx,ut = sp.gradient(ud,dx,dy,dt)
vmy,vmx,vt = sp.gradient(vd,dx,dy,dt)
f = sw.cor(lat)
ix,jx,kx = ut.shape
f = np.repeat(f,kx).reshape(ix,jx,kx)
zeta = (vmx-umy)/np.abs(f)
zeta = np.ma.masked_array(zeta,np.abs(zeta)>10.)

#
# plotting
#

xgm1,ygm1 = m(data['lon'],data['lat'])

dec=2

for i in range(timei.size):

    fig=plt.figure(facecolor='w', figsize=(8.,8.5))

    sc=8.
    #Q = m.quiver(xgm1,ygm1,data1['u'].mean(axis=2),data1['v'].mean(axis=2),scale=sc,color='k',linewidth=.5)
    m.pcolormesh(xgm1,ygm1,zeta[:,:,i], vmin=-4.,vmax=4.,cmap='bwr')
    cb = plt.colorbar(extend='both')
    cb.set_label(u'$\zeta/|f|$')
    Q = m.quiver(xgm1,ygm1,ud[:,:,i],vd[:,:,i],scale=sc,color='m',linewidth=.5)

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
    plt.title(str(timei[i])[0:10])
    plt.savefig('figs2movie/'+str(timei[i])[0:10],dpi=80)
    plt.close('all')


