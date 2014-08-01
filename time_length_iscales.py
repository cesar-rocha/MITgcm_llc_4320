
import matplotlib.pyplot as plt
import scipy.signal
import scipy as sp
import numpy as np
import glob   
import seawater.csiro as sw

import aux_func as my

## data
iz = 100    # vertical level [m]
fni = 'subsets/'+str(iz)+'m_uvwts_high_90_919_768.npz'
data = np.load(fni)

## settings
lat = data['lat'][:,4]
lon = data['lon'][:,4]
dist,ang = sw.dist(lon,lat)
dx = dist.mean()   # [km], about 1 km

# avoid boundaries and ice
ilat = ((lat>-62)&(lat<-55))
U = data['u'][ilat,:,:] 
V = data['v'][ilat,:,:] 

## auto-correlation in time
ix,jx,kx = U.shape
Au_t = np.zeros(U.shape)
Av_t = np.zeros(V.shape)

for i in range(ix):
    for j in range(jx):
        Au_t[i,j,:] = my.auto_corr(U[i,j,:])
        Av_t[i,j,:] = my.auto_corr(V[i,j,:])

Au_t = np.nanmean(Au_t,axis=0); Au_t = np.nanmean(Au_t,axis=0)
Av_t = np.nanmean(Av_t,axis=0); Av_t = np.nanmean(Av_t,axis=0)
lag_t = np.arange(Au_t.size)

## estimates of time scales (days)
#  (no significant changes if use Uf and Vf instead)
dt = 1. # [h]

# gauss-fit
Tu = my.fit_gauss(lag_t[0:1000],Au_t[0:1000])/24
Tv = my.fit_gauss(lag_t[0:1000],Av_t[0:1000])/24

# integral-time scale Wunsch 97 JPO
Tu2 = ( Au_t[0]**2 + 2*(Au_t[1:]**2).sum() )*(dt/Au_t[0]**2)/24
Tv2 = ( Av_t[0]**2 + 2*(Av_t[1:]**2).sum() )*(dt/Av_t[0]**2)/24

# integral time scale
Tu3 = Au_t[0:1001].sum()/24
Tv3 = Av_t[0:1001].sum()/24

## auto-correlation in (along-track) space 
dx = 1.1 # [km]
ix,jx,kx = U.shape
Au_s = np.zeros(U.shape)
Av_s = np.zeros(V.shape)

for j in range(jx):
    for k in range(kx):
        Au_s[:,j,k] = my.auto_corr(U[:,j,k])
        Av_s[:,j,k] = my.auto_corr(V[:,j,k])

Au_s = np.nanmean(Au_s,axis=1); Au_s = np.nanmean(Au_s,axis=1)
Av_s = np.nanmean(Av_s,axis=1); Av_s = np.nanmean(Av_s,axis=1)
lag_s = np.arange(Au_s.size)*dx

## estimates of (along-track) length scales [km]
#  (no significant changes if use Uf and Vf instead)

# gauss-fit
Lu = my.fit_gauss(lag_s[0:100],Au_s[0:100])
Lv = my.fit_gauss(lag_s[0:100],Av_s[0:100])

# integral-time scale Wunsch 97 JPO
Lu2 = ( Au_s[0]**2 + 2*(Au_s[1:]**2).sum() )*(dx/Au_s[0]**2)
Lv2 = ( Av_s[0]**2 + 2*(Av_s[1:]**2).sum() )*(dx/Av_s[0]**2)

# integral time scale
Lu3 = Au_t[0:57].sum()
Lv3 = Av_t[0:57].sum()


