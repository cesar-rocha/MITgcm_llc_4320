
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io as sp
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
# create filter (butterworth order 3, lowpass cutoff 10 Hz)
b, a = sp.signal.butter(2, 1./30,btype='lowpass')

## data
data = np.load('subsets/eta_919_1440.npz')
grid_eta = sp.io.loadmat('grid_eta.mat')

## settings
lat = grid_eta['yg'][:,4]
lon = grid_eta['xg'][:,4]

# find and mask bad data
eta = data['eta']
deta = np.diff(eta,axis=2).sum(axis=2) == 0 # points where variable doesn't
                                            #   change over time
                                         
eta[deta,:] = np.nan

# for projection onto regular grid
lati = np.linspace(lat.min(),lat.max(),lat.size)

# avoid boundaries and ice
ilat = ((lati>-62)&(lati<-55))

dist,ang = sw.dist(lon,lati)
dx = dist.mean()   # [km], about 1 km

# project onto regular grid: 
interp_eta = sp.interpolate.interp1d(lat,eta,kind='linear',axis=0)
ETA = interp_eta(lati)[ilat,:,:]

ETAf = sp.signal.filtfilt(b, a, my.rmean(ETA), axis=2)

## spectral window

# window for ft in space 
ix,jx,kx = ETA.shape    
window_s = np.repeat(np.hanning(ix),jx).reshape(ix,jx)
window_s = np.repeat(window_s,kx).reshape(ix,jx,kx)

# window for ft in time
ix,jx,kx = ETA.shape
window_t = np.repeat(np.hanning(kx),jx).reshape(kx,jx).T
window_t = np.repeat(window_t,ix).reshape(jx,kx,ix)
window_t = np.transpose(window_t,axes=(2,0,1))

# in space
Eeta,k,dk,kNy = my.spec_est(my.rmean(ETA)*window_s,dx)
Eetaf,_,_,_ = my.spec_est(ETAf*window_s,dx)

# in time
dt = 1. # [h]
Eeta_t,f,df,fNy = my.spec_est_time(my.rmean(ETA)*window_t,dt)
Eetaf_t,f,df,fNy = my.spec_est_time(ETAf*window_t,dt)

## masking bad values
Eeta = np.ma.masked_array(Eeta,np.isnan(Eeta))
Eetaf = np.ma.masked_array(Eetaf,np.isnan(Eetaf))
Eeta_t = np.ma.masked_array(Eeta_t,np.isnan(Eeta_t))
Eetaf_t = np.ma.masked_array(Eetaf_t,np.isnan(Eetaf_t))

## plotting

# -2 and -3 slopes in the loglog space
ks2 = np.array([1.e-2,1.e-1])
Es2 = 1.e8*(ks2**(-2))
ks4 = np.array([1.e-2,1.e-1])
Es4 = 1.e-1*(ks4**(-4))
ks5 = np.array([1.e-2,1.e-1])
Es5 = 1.e-10*(ks5**(-5))

fig = plt.figure(facecolor='w', figsize=(12.,12.))
ax1 = fig.add_subplot(111)

ix,jx = Eeta.shape

for i in range(0,jx,1):
    if i == 0:
        ax1.loglog(k,Eeta[:,i]*(10**(i)), color=color1, linewidth=lw1,
                label=u'Total')
        ax1.loglog(k,Eetaf[:,i]*(10**(i)) ,color=color2, linewidth=lw1,
                label=u'Low-pass (30 h)')
    else:
        ax1.loglog(k,Eeta[:,i]*(10**(i)), color=color1, linewidth=lw1)
        ax1.loglog(k,Eetaf[:,i]*(10**(i)) ,color=color2, linewidth=lw1)


# reference slopes
ax1.loglog(ks2,Es2,'--', color='k',linewidth=2.)
ax1.loglog(ks4,Es4,'--', color='k',linewidth=2.)
ax1.loglog(ks5,Es5,'--', color='k',linewidth=2.)

#ax1.axis((1./(1000),1./2,.4e-5,30))
plt.text(0.011178591777554035, 1.3741078901053396,u'k$^{-5}$')
plt.text(0.010378367193526972, 224679.00918126441,u'k$^{-4}$')
plt.text(0.01, 56415888336.127533,u'k$^{-2}$')

# longitude references
plt.text(0.42960024080617593, 4.928427058753194e-10,u'$72.9\degree W$')
plt.text(0.42960024080617593, 0.12261445346236022,u'$56.2\degree W$')
plt.text(0.42960024080617593, 20658756.020541538,u'$47.9\degree W$')

plt.xlabel('Wavenumber [cycles/km]')
plt.ylabel(u'Spectral density [m$^{2}$/(cycles/km)] x $10^n$')

lg = plt.legend(loc=3,title= u'$\mathcal{SSH}$ spectrum', prop={'size':22}, numpoints=1)
lg.draw_frame(False)
my.leg_width(lg,5.)

plt.savefig('figs/spectra_ssh')


# Frequency spec. filtered
fig = plt.figure(facecolor='w', figsize=(12.,15.))
ax1 = fig.add_subplot(111)
#ax1.fill_between(ki,Eu_l1,Eu_u1, color=color1, alpha=0.25)
#ax1.set_xscale('log'); ax1.set_yscale('log')

ix,jx = Eeta.shape

for i in range(0,jx,1):
    ax1.loglog(f,Eeta_t[i,:]*(10**i), color=color1, linewidth=lw1,alpha=.6)

#ax1.axis((1./(1000),1./2,1.e-7,1.e9))

plt.xlabel('Frequency [cycles/h]')
plt.ylabel(u'Spectral density [(m$^{2}$)/(cycles/h)] x $10^n$ ')
plt.savefig('figs/freq_spectra_ssh')

# Frequency spec. filtered
fig = plt.figure(facecolor='w', figsize=(12.,15.))
ax1 = fig.add_subplot(111)
#ax1.fill_between(ki,Eu_l1,Eu_u1, color=color1, alpha=0.25)
#ax1.set_xscale('log'); ax1.set_yscale('log')

ix,jx = Eeta.shape

for i in range(0,jx,1):
    ax1.loglog(f,Eetaf_t[i,:]*(10**i), color=color1, linewidth=lw1,alpha=.6)

#ax1.axis((1./(1000),1./2,1.e-7,1.e9))

plt.xlabel('Frequency [cycles/h]')
plt.ylabel(u'Spectral density [(m$^{2}$)/(cycles/h)] x $10^n$ ')
plt.savefig('figs/freq_spectra_lowpass_ssh')

## remove mask before computing errors and saving
Eeta = np.array(Eeta)
Eetaf = np.array(Eetaf)

## save to compare against KE spectra
fno = 'outputs/spectra_ssh'
np.savez(fno, Eeta=Eeta, Eetaf=Eetaf, k=k)

