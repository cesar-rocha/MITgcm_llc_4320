
import matplotlib.pyplot as plt
import scipy.signal
import scipy as sp      
import numpy as np
import glob   
import seawater.csiro as sw

import aux_func_3dfields as my

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
iz = 0    # vertical level [m]
fni = 'subsets/'+str(iz)+'m_uvwts_high_90_919_768_zonal.npz'
data = np.load(fni)

## settings
lat = data['lat'][10,:]
lats = data['lat'][:,4]
lon = data['lon'][10,:]

# find and mask bad data
u = data['u']
v = data['v']

#du = np.diff(u,axis=2).sum(axis=2) == 0   # points where variable doesn't
                                           #   change over time
#dv = np.diff(v,axis=2).sum(axis=2) == 0

#u[du,:] = np.nan
#v[dv,:] = np.nan

# for projection onto regular grid
loni = np.linspace(lon.min(),lon.max(),lon.size)

# avoid boundaries and ice
ilat = ((lats>-62)&(lats<-55))

dist,ang = sw.dist(lon,lat)
dx = dist.mean()   # [km], about 1 km

# project onto regular grid: linear interp
interp_u = sp.interpolate.interp1d(lon,u,kind='linear',axis=1)
interp_v = sp.interpolate.interp1d(lon,v,kind='linear',axis=1)

U = interp_u(loni)[ilat,:,:]
V = interp_v(loni)[ilat,:,:]

Uf = sp.signal.filtfilt(b, a, my.rmean(U), axis=2)
Vf = sp.signal.filtfilt(b, a, my.rmean(V), axis=2)

# sampled velocities

## spectral window

# window for ft in space 
ix,jx,kx = U.shape    
window_s = np.repeat(np.hanning(jx),ix).reshape(jx,ix).T
window_s = np.repeat(window_s,kx).reshape(ix,jx,kx)

# window for ft in time
ix,jx,kx = U.shape
window_t = np.repeat(np.hanning(kx),jx).reshape(kx,jx).T
window_t = np.repeat(window_t,ix).reshape(jx,kx,ix)
window_t = np.transpose(window_t,axes=(2,0,1))

## spectral estimates

# in space (zonal wavenumber)
Eu,k,dk,kNy = my.spec_est_zonal(my.rmean(U)*window_s,dx)
Ev,_,_,_ = my.spec_est_zonal(my.rmean(V)*window_s,dx)
Euf,_,_,_ = my.spec_est_zonal(Uf*window_s,dx)
Evf,_,_,_ = my.spec_est_zonal(Vf*window_s,dx)

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
Eu_t = np.ma.masked_array(Eu_t,np.isnan(Eu_t))
Ev_t = np.ma.masked_array(Ev_t,np.isnan(Ev_t))
Euf_t = np.ma.masked_array(Euf_t,np.isnan(Euf_t))
Evf_t = np.ma.masked_array(Evf_t,np.isnan(Evf_t))

# Eu/Ev on 10-100km range
l = 1./k
fl = (l<100)&(l>10)     # [km]
r = (Ev[:,fl]/Eu[:,fl]).mean()
rf = (Evf[:,fl]/Euf[:,fl]).mean()

## plotting

# -2 and -3 slopes in the loglog space
ks2 = np.array([1.e-2,1.e-1])
Es2 = (ks2**(-2))

ks3 = np.array([1.e-2,1.e-1])
Es3 = 1.e-3*(ks3**(-3))

# zonal and zonal velocity spectra
fig = plt.figure(facecolor='w', figsize=(12.,12.))
ax1 = fig.add_subplot(111)
#ax1.fill_between(ki,Eu_l1,Eu_u1, color=color1, alpha=0.25)
#ax1.set_xscale('log'); ax1.set_yscale('log')

ix,jx = Eu.shape

for i in range(2,ix,2):
    if Eu[i,:].mask.sum() != ix:
        ax1.loglog(k,Eu[i,:]*(10**(i/2.)), color=color1, linewidth=lw1)
        ax1.loglog(k,Ev[i,:]*(10**(i/2.)), color=color2, linewidth=lw1)

# reference slopes
ax1.loglog(ks2,Es2,'--', color='k',linewidth=2.)
ax1.loglog(ks3,Es3,'--', color='k',linewidth=2.)

ax1.axis((1./(1000),1./2,1.e-7,1.e11))

plt.text(0.077107526925355302, 133.35214321633242,u'k$^{-2}$')
plt.text(0.084609544077146992, 1.3823722273578996,u'k$^{-3}$')

plt.xlabel('Wavenumber [cycles/km]')
plt.ylabel(u'Spectral density [(m$^{2}$ s$^{-2}$)/(cycles/km)] x $10^n$ ')
plt.savefig('figs/spectra_uv_'+str(iz)+'m_zonal')

# zonal and zonal LOW PASS (30h) velocity spectra
fig = plt.figure(facecolor='w', figsize=(12.,12.))
ax1 = fig.add_subplot(111)
#ax1.fill_between(ki,Eu_l1,Eu_u1, color=color1, alpha=0.25)
#ax1.set_xscale('log'); ax1.set_yscale('log')

ix,jx = Eu.shape

for i in range(2,ix,2):
    if Euf[i,:].mask.sum() != ix:
        ax1.loglog(k,Euf[i,:]*(10**(i/2.)), color=color1, linewidth=lw1)
        ax1.loglog(k,Evf[i,:]*(10**(i/2.)), color=color2, linewidth=lw1)

# reference slopes
ax1.loglog(ks2,Es2,'--', color='k',linewidth=2.)
ax1.loglog(ks3,Es3,'--', color='k',linewidth=2.)

ax1.axis((1./(1000),1./2,1.e-7,1.e11))

plt.text(0.077107526925355302, 133.35214321633242,u'k$^{-2}$')
plt.text(0.084609544077146992, 1.3823722273578996,u'k$^{-3}$')

plt.xlabel('Wavenumber [cycles/km]')
plt.ylabel(u'Spectral density [(m$^{2}$ s$^{-2}$)/(cycles/km)] x $10^n$ ')
plt.savefig('figs/spectra_lowpass_uv_'+str(iz)+'m_zonal')

# Frequency spec. mean
Eu_t = Eu_t.mean(axis=0)
Ev_t = Ev_t.mean(axis=0)
Euf_t = Euf_t.mean(axis=0)
Evf_t = Evf_t.mean(axis=0)

# total
fig = plt.figure(facecolor='w', figsize=(12.,12.))
ax1 = fig.add_subplot(111)

ax1.loglog(f,Eu_t, color=color1, linewidth=lw1,alpha=.6)
ax1.loglog(f,Ev_t, color=color2, linewidth=lw1,alpha=.6)

plt.xlabel('Frequency [cycles/h]')
plt.ylabel(u'Spectral density [(m$^{2}$ s$^{-2}$)/(cycles/h)] x $10^n$ ')
plt.savefig('figs/freq_spectra_uv_'+str(iz)+'m_zonal')

# low-pass
fig = plt.figure(facecolor='w', figsize=(12.,12.))
ax1 = fig.add_subplot(111)

ax1.loglog(f,Euf_t, color=color1, linewidth=lw1,alpha=.6)
ax1.loglog(f,Evf_t, color=color2, linewidth=lw1,alpha=.6)

plt.xlabel('Frequency [cycles/h]')
plt.ylabel(u'Spectral density [(m$^{2}$ s$^{-2}$)/(cycles/h)] x $10^n$ ')
plt.savefig('figs/freq_spectra_lowpass_uv_'+str(iz)+'m_zonal')


## compute (spatially) mean spectra and estimate formal errors
Eu = Eu[:,0:].mean(axis=0)
Ev = Ev[:,0:].mean(axis=0)
Euf = Euf[:,0:].mean(axis=0)
Evf = Evf[:,0:].mean(axis=0)

## remove mask before computing errors and saving
Eu = np.array(Eu)
Ev = np.array(Ev)
Euf = np.array(Euf)
Evf = np.array(Evf)

sn = 90 # about 90 independent estimates of the spectrum
Eu_l, Eu_u,_,_= my.spec_error(Eu, sn, .95) 
Ev_l, Ev_u,_,_= my.spec_error(Ev, sn, .95) 
Euf_l, Euf_u,_,_= my.spec_error(Euf, sn, .95) 
Evf_l, Evf_u,_,_= my.spec_error(Evf, sn, .95) 

## saving for comparison against observations
fno = 'outputs/spectra_'+str(iz)+'m_zonal'
np.savez(fno, Eu=Eu, Ev=Ev, Eu_l=Eu_l,Eu_u=Eu_u, Ev_l=Ev_l,Ev_u=Ev_u, 
        k=k, r=r, rf = rf,Euf=Euf, Evf=Evf, Euf_l=Euf_l,Euf_u=Euf_u, Evf_l=Evf_l,Evf_u=Evf_u)

