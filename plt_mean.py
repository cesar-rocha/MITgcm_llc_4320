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
data2 = np.load('subsets/0m_uvwts_high_90_919_768.npz')
data4 = io.loadmat('subsets/uv_snap.mat')
data5 = np.load('subsets/eta_0_1_919_1440_20_20.npz')

#
# mean and relative vorticity
#
um,vm = data['u'].mean(axis=2),data['v'].mean(axis=2)
lon, lat = data['lon'],data['lat']

dx = 1.14*1.e3
dy = 1.14*1.e3

umy,umx = sp.gradient(um,dx,dy)
vmy,vmx = sp.gradient(vm,dx,dy)
f = sw.cor(lat)
zeta = (vmx-umy)/np.abs(f)
zeta = np.ma.masked_array(zeta,np.abs(zeta)>10.)


#
# plotting
#

xgm1,ygm1 = m(data['lon'],data['lat'])

dec=2

# mean currents
fig=plt.figure(facecolor='w', figsize=(8.,8.5))

sc=8.
#Q = m.quiver(xgm1,ygm1,data1['u'].mean(axis=2),data1['v'].mean(axis=2),scale=sc,color='k',linewidth=.5)
m.pcolormesh(xgm1,ygm1,zeta, vmin=-4.,vmax=4.,cmap='bwr')
cb = plt.colorbar(extend='both')
cb.set_label(u'$\zeta/|f|$')
Q = m.quiver(xgm1,ygm1,um,vm,scale=sc,color='m',linewidth=.5)

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

plt.savefig('figs/mean_uv.png')

fig=plt.figure(facecolor='w', figsize=(8.,8.5))

sc=8.

m.pcolor(xgm1,ygm1,data['theta'].mean(axis=2))
cb = plt.colorbar(extend='both')

m.fillcontinents(color='.60',lake_color='none')
m.drawparallels(np.arange(latplot[0], latplot[1], 4),
    labels=[1, 0, 0, 0], linewidth=.5, 
    fontsize=18., fontweight='medium')
m.drawmeridians(np.arange(lonplot[0], lonplot[1], 7),
    labels=[0, 0, 0, 1], dashes=[1,1], linewidth=.5, 
    fontsize=18., fontweight='medium')
m.drawcoastlines()

fig=plt.figure(facecolor='w', figsize=(8.,8.5))

sc=8.
salt = np.ma.masked_array(data['salt'],data['salt']<33.4)
m.pcolor(xgm1,ygm1,salt.mean(axis=2))
cb = plt.colorbar(extend='both')

m.fillcontinents(color='.60',lake_color='none')
m.drawparallels(np.arange(latplot[0], latplot[1], 4),
    labels=[1, 0, 0, 0], linewidth=.5, 
    fontsize=18., fontweight='medium')
m.drawmeridians(np.arange(lonplot[0], lonplot[1], 7),
    labels=[0, 0, 0, 1], dashes=[1,1], linewidth=.5, 
    fontsize=18., fontweight='medium')
m.drawcoastlines()

fig=plt.figure(facecolor='w', figsize=(8.,8.5))

sc=8.

m.pcolor(xgm1,ygm1,data['w'].mean(axis=2))
cb = plt.colorbar(extend='both')

m.fillcontinents(color='.60',lake_color='none')
m.drawparallels(np.arange(latplot[0], latplot[1], 4),
    labels=[1, 0, 0, 0], linewidth=.5, 
    fontsize=18., fontweight='medium')
m.drawmeridians(np.arange(lonplot[0], lonplot[1], 7),
    labels=[0, 0, 0, 1], dashes=[1,1], linewidth=.5, 
    fontsize=18., fontweight='medium')
m.drawcoastlines()

## sections in which spectra is being computed (also compare with grid_eta)
#grid_eta = sp.io.loadmat('grid_eta.mat')
#lon_eta =  grid_eta['xg'][:,0::100]
#lat_eta =  grid_eta['yg'][:,0::100]

ilat = ((data2['lat'][:,3]>-62)&(data2['lat'][:,3]<-55))
xg2,yg2 = m(data2['lon'][ilat,:],data2['lat'][ilat,:])
#xg3,yg3 = m(lon_eta[ilat,:],lat_eta[ilat,:])

fig=plt.figure(facecolor='w', figsize=(8.,8.5))

m.plot(xg2,yg2,'k.')
#m.plot(xg3,yg3,'m.')

plt.text(173718.87602860748, 511542.78266188467,'1')
plt.text(982595.90111271245, 361277.09025495616,'9')

m.fillcontinents(color='.60',lake_color='none')
m.drawparallels(np.arange(latplot[0], latplot[1], 4),
    labels=[1, 0, 0, 0], linewidth=.5, 
    fontsize=18., fontweight='medium')
m.drawmeridians(np.arange(lonplot[0], lonplot[1], 7),
    labels=[0, 0, 0, 1], dashes=[1,1], linewidth=.5, 
    fontsize=18., fontweight='medium')
m.drawcoastlines()

plt.savefig('figs/transects2KE')


