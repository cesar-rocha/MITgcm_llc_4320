
import matplotlib.pyplot as plt
import scipy.signal
import scipy as sp
import numpy as np
import glob   
import seawater.csiro as sw

import aux_func_3dfields as my

## reorder 3d array A to mimic sampling model
def reorder(A,Ai):
    l,m,n = Ai.shape
    As = np.zeros(Ai.shape)
    kk = np.arange(n)
    for i in range(l):
        for j in range(m):
            kkaux = kk+i
            if i>0:
                for ia in range(i):
                    kkaux[n-ia-1] = ia
            As[i,j,:] = Ai[i,j,kkaux]
    return As



plt.close('all')

plt.rcParams.update({'font.size': 24, 'legend.handlelength'  : 1.5
    , 'legend.markerscale': 14., 'legend.linewidth': 3.})

color2 = '#6495ed'
color1 = '#ff6347'
color3 = '#8470ff'
color4 = '#3cb371'
lw1 = 2

## filter
# create filter (butterworth order 2, lowpass cutoff period 30 h)
b, a = sp.signal.butter(2, 1./30,btype='lowpass')

## data
iz = 100    # vertical level [m]
fni = 'subsets/'+str(iz)+'m_uvwts_high_90_919_768.npz'
data = np.load(fni)

## settings
lat = data['lat'][:,4]
lon = data['lon'][:,4]

# find and mask bad data
u = data['u']
v = data['v']

du = np.diff(u,axis=2).sum(axis=2) == 0   # points where variable doesn't
                                          #   change over time
dv = np.diff(v,axis=2).sum(axis=2) == 0

u[du,:] = np.nan
v[dv,:] = np.nan

# for projection onto regular grid
lati = np.linspace(lat.min(),lat.max(),lat.size)

# avoid boundaries and ice
ilat = ((lati>-62)&(lati<-55))

dist,ang = sw.dist(lon,lati)
dx = dist.mean()   # [km], about 1 km

# project onto regular grid: linear interp
interp_u = sp.interpolate.interp1d(lat,u,kind='linear',axis=0)
interp_v = sp.interpolate.interp1d(lat,v,kind='linear',axis=0)
U = interp_u(lati)[ilat,:,:]
V = interp_v(lati)[ilat,:,:]

Uf = sp.signal.filtfilt(b, a, my.rmean(U), axis=2)
Vf = sp.signal.filtfilt(b, a, my.rmean(V), axis=2)

## add noise to U and V
varu = np.nanvar(U)
varv = np.nanvar(V)
nl = .0
k,l,m = U.shape
noise_u = np.sqrt(nl*varu)*np.random.randn(k,l,m)
noise_v = np.sqrt(nl*varv)*np.random.randn(k,l,m)
U = U + noise_u
V = V + noise_v



# linear interp onto 4.25 minute grid
ix,jx,kx = U.shape
dts = 48./ix
t = np.arange(kx)
ti = np.arange(0,kx-1,dts)
ti = np.linspace(0,kx-1,13083)  # dts about 48/ix
interp_ut = sp.interpolate.interp1d(t,U,kind='linear',axis=2)
interp_vt = sp.interpolate.interp1d(t,V,kind='linear',axis=2)
Ui = interp_ut(ti)
Vi = interp_vt(ti)

Us = reorder(U,Ui)[:,:,0::7]
Vs = reorder(V,Vi)[:,:,0::7]

U = U[:,:,0:kx-ix+1]
V = V[:,:,0:kx-ix+1]

Uf = Uf[:,:,0:kx-ix+1]
Vf = Vf[:,:,0:kx-ix+1]
Us = Us[:,:,0:kx-ix+1]
Vs = Vs[:,:,0:kx-ix+1]
np.savez('UV_sampled',Us=Us, Vs=Vs)
#UV_sampled = np.load('UV_sampled.npz')
#Us = UV_sampled['Us']
#Vs = UV_sampled['Vs']

lat = lat[ilat]
lon = lon[ilat]

dist,_ = sw.dist(lon,lat)
dist = np.append(0,np.cumsum(dist))

dxt = 5.
Usb = my.block_ave(dist,Us,dxt)
Vsb = my.block_ave(dist,Vs,dxt)
isb,jsb,ksb = Usb.shape

lati = np.linspace(lat.min(),lat.max(),isb)





#
# start tmw by computing the spectrum of the block averaged quantities
#


## spectral window

# window for ft in space 
ix,jx,kx = U.shape    
window_s = np.repeat(np.hanning(ix),jx).reshape(ix,jx)
window_s = np.repeat(window_s,kx).reshape(ix,jx,kx)

ix_s,jx_s,kx_s = Us.shape    
window_ss = np.repeat(np.hanning(ix_s),jx_s).reshape(ix_s,jx_s)
window_ss = np.repeat(window_ss,kx_s).reshape(ix_s,jx_s,kx_s)

window_sb = np.repeat(np.hanning(isb),jsb).reshape(isb,jsb)
window_sb = np.repeat(window_sb,ksb).reshape(isb,jsb,ksb)

# window for ft in time
ix,jx,kx = U.shape
window_t = np.repeat(np.hanning(kx),jx).reshape(kx,jx).T
window_t = np.repeat(window_t,ix).reshape(jx,kx,ix)
window_t = np.transpose(window_t,axes=(2,0,1))

## spectral estimates

# in space (meridional wavenumber)
Eu,k,dk,kNy = my.spec_est(my.rmean(U)*window_s,dx)
Ev,_,_,_ = my.spec_est(my.rmean(V)*window_s,dx)
Eus,_,_,_ = my.spec_est(my.rmean(Us)*window_ss,dx)
Evs,_,_,_ = my.spec_est(my.rmean(Vs)*window_ss,dx)
Eusb,ksb,_,_ = my.spec_est(my.rmean(Usb)*window_sb,dxt)
Evsb,ksb,_,_ = my.spec_est(my.rmean(Vsb)*window_sb,dxt)

Euf,_,_,_ = my.spec_est(Uf*window_s,dx)
Evf,_,_,_ = my.spec_est(Vf*window_s,dx)

# in time
dt = 1. # [h]
Eu_t,f,df,fNy = my.spec_est_time(my.rmean(U)*window_t,dt)
Ev_t,f,df,fNy = my.spec_est_time(my.rmean(V)*window_t,dt)
Euf_t,f,df,fNy = my.spec_est_time(Uf*window_t,dt)
Evf_t,f,df,fNy = my.spec_est_time(Vf*window_t,dt)

## masking bad values
Eu = np.ma.masked_array(Eu,np.isnan(Eu))
Ev = np.ma.masked_array(Ev,np.isnan(Ev))
Euf = np.ma.masked_array(Euf,np.isnan(Euf))
Evf = np.ma.masked_array(Evf,np.isnan(Evf))
Eus = np.ma.masked_array(Eus,np.isnan(Eus))
Evs = np.ma.masked_array(Evs,np.isnan(Evs))
Eusb = np.ma.masked_array(Eusb,np.isnan(Eusb))
Evsb = np.ma.masked_array(Evsb,np.isnan(Evsb))
Eu_t = np.ma.masked_array(Eu_t,np.isnan(Eu_t))
Ev_t = np.ma.masked_array(Ev_t,np.isnan(Ev_t))
Euf_t = np.ma.masked_array(Euf_t,np.isnan(Euf_t))
Evf_t = np.ma.masked_array(Evf_t,np.isnan(Evf_t))

# Eu/Ev on 10-100km range
l = 1./k
fl = (l<100)&(l>10)             # [km]
r = (Eu[fl,:]/Ev[fl,:]).mean()
rf = (Euf[fl,:]/Evf[fl,:]).mean()

## plotting

# -2 and -3 slopes in the loglog space
ks2 = np.array([1.e-2,1.e-1])
Es2 = (ks2**(-2))

ks3 = np.array([1.e-2,1.e-1])
Es3 = 1.e-3*(ks3**(-3))

# meridional and zonal velocity spectra
fig = plt.figure(facecolor='w', figsize=(12.,12.))
ax1 = fig.add_subplot(111)
#ax1.fill_between(ki,Eu_l1,Eu_u1, color=color1, alpha=0.25)
#ax1.set_xscale('log'); ax1.set_yscale('log')

ix,jx = Eu.shape

for i in range(jx):
    if Eu[:,i].mask.sum() != ix:
        ax1.loglog(k,Eu[:,i]*(10**i), color=color1, linewidth=lw1)
        ax1.loglog(k,Ev[:,i]*(10**i), color=color2, linewidth=lw1)
        ax1.loglog(k,Eus[:,i]*(10**i), '--', color=color1, linewidth=lw1)
        ax1.loglog(k,Evs[:,i]*(10**i), '--', color=color2, linewidth=lw1)
        ax1.loglog(ksb,Eusb[:,i]*(10**i), '.', color=color1, linewidth=lw1)
        ax1.loglog(ksb,Evsb[:,i]*(10**i), '.', color=color2, linewidth=lw1)

# reference slopes
ax1.loglog(ks2,Es2,'--', color='k',linewidth=2.)
ax1.loglog(ks3,Es3,'--', color='k',linewidth=2.)

ax1.axis((1./(1000),1./2,1.e-7,1.e9))

plt.text(0.077107526925355302, 133.35214321633242,u'k$^{-2}$')
plt.text(0.084609544077146992, 1.3823722273578996,u'k$^{-3}$')

plt.xlabel('Wavenumber [cycles/km]')
plt.ylabel(u'Spectral density [(m$^{2}$ s$^{-2}$)/(cycles/km)] x $10^n$ ')
#plt.savefig('figs/spectra_uv_'+str(iz)+'m')


## (spatially) mean spectra
Eu = np.nanmean(Eu,axis=1)
Euf = np.nanmean(Euf,axis=1)
Eus = np.nanmean(Eus,axis=1)
Eusb = np.nanmean(Eusb,axis=1)
Ev = np.nanmean(Ev,axis=1)
Evf = np.nanmean(Evf,axis=1)
Evsb = np.nanmean(Evsb,axis=1)
Evs = np.nanmean(Evs,axis=1)



