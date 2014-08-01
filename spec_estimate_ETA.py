
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io as sp
import scipy as sp
import numpy as np
import glob   
import seawater.csiro as sw

import aux_func_3dfields as my


def spectral_slope(k,E,kmin,kmax,stdE):
    ''' compute spectral slope in log space in
        a wavenumber subrange [kmin,kmax]'''

    fr = np.where((k>=kmin)&(k<=kmax))

    ki = np.matrix(np.log10(k[fr])).T
    Ei = np.matrix(np.log10(np.real(E[fr]))).T
    dd = np.matrix(np.eye(ki.size)*(stdE**2))

    G = np.matrix(np.append(np.ones((ki.size,1)),ki,axis=1))
    Gg = ((G.T*G).I)*G.T
    m = Gg*Ei
    mm = np.array(Gg*dd*Gg.T)
    yfit = np.array(G*m)
    m = np.array(m)[1]
    mm = np.sqrt(np.array(Gg*dd*Gg.T)[1,1])

    return m, mm

plt.close('all')

plt.rcParams.update({'font.size': 24, 'legend.handlelength'  : 1.5
    , 'legend.markerscale': 1.5, 'legend.linewidth': 3.})

color2 = '#6495ed'
color1 = '#ff6347'
color3 = '#8470ff'
color4 = '#3cb371'
color5 = '#ffa500'
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
lons = data['lon'][200,:]
eta = data['eta']
                                         
# for projection onto regular grid
lati = np.linspace(lat.min(),lat.max(),lat.size)
loni = np.linspace(lons.min(),lons.max(),lons.size)


# avoid boundaries and ice
# first lat band
latlim = np.array([-58,-54])
lonlim = np.array([-75,-46])
ilat = ((lati>=latlim[0])&(lati<=latlim[1]))
ilon = ((loni>=lonlim[0])&(loni<=lonlim[1]))


dist,ang = sw.dist(lon,lati)
dx = dist.mean()   # [km], about 1 km

# project onto regular grid: 
interp_eta = sp.interpolate.interp1d(lat,eta,kind='linear',axis=0)

ETA = interp_eta(lati)[ilat,:,:]; ETA = ETA[:,ilon,:]

ETAd = sp.signal.detrend(ETA,axis=0,type='linear')
ETAf = sp.signal.filtfilt(b, a, ETAd, axis=2)

# mask bad data
deta = np.diff(ETA,axis=2).sum(axis=2) == 0 # points where variable doesn't
                                            #   change over time
ETA[deta,:] = np.nan
ETAd[deta,:] = np.nan
ETAf[deta,:] = np.nan

# mirror (to become periodic) in meridional direction
ETAfm = np.append(ETAf,np.flipud(ETAf),axis=0)

## spectral window

# window for ft in space 
ix,jx,kx = ETA.shape    
window_s = np.repeat(np.hanning(ix),jx).reshape(ix,jx)
window_s = np.repeat(window_s,kx).reshape(ix,jx,kx)

# in space
Eeta,k,dk,kNy = my.spec_est_meridional(my.rmean(ETA),dx)   # de-mean
Eetad,_,_,_ = my.spec_est_meridional(my.rmean(ETAd),dx)    # detrended 
Eetadf,_,_,_ = my.spec_est_meridional(my.rmean(ETAf),dx)   #  detrended + filt
Eetadfw,_,_,_ = my.spec_est_meridional(my.rmean(ETAf)*window_s,dx) # detrended + filt + wind.
Eetadfm,km,_,_ = my.spec_est_meridional(my.rmean(ETAfm),dx) # detrended + filt + mirror 

## masking bad values
fm = (km>=k[0])
Eetadfm = Eetadfm[fm]/2. 
km = km[fm]

## compute spectral slopes
stdE = (1/np.sqrt(8))
kr = 1./np.array([100,10])

loni = loni[ilon]
s = np.zeros(loni.size); su = np.zeros(loni.size)
sd = np.zeros(loni.size); sud = np.zeros(loni.size)
sdf = np.zeros(loni.size); sudf = np.zeros(loni.size)
sdfw = np.zeros(loni.size); sudfw = np.zeros(loni.size)
sdfm = np.zeros(loni.size); sudfm = np.zeros(loni.size)

for i in range(loni.size):
    s[i],su[i] = spectral_slope(k,Eeta[:,i],kr[0],kr[1],stdE)
    sd[i],sud[i] = spectral_slope(k,Eetad[:,i],kr[0],kr[1],stdE)
    sdf[i],sudf[i] = spectral_slope(k,Eetadf[:,i],kr[0],kr[1],stdE)
    sdfw[i],sudfw[i] = spectral_slope(k,Eetadfw[:,i],kr[0],kr[1],stdE)
    sdfm[i],sudfm[i] = spectral_slope(km,Eetadfm[:,i],kr[0],kr[1],stdE)

s = np.ma.masked_array(s,np.isnan(s));su = np.ma.masked_array(su,np.isnan(s))
sd = np.ma.masked_array(sd,np.isnan(sd));sud = np.ma.masked_array(sud,np.isnan(sd))
sdf = np.ma.masked_array(sdf,np.isnan(sdf));sudf = np.ma.masked_array(sudf,np.isnan(sdf))
sdfw = np.ma.masked_array(sdfw,np.isnan(sdfw));sudfw = np.ma.masked_array(sudfw,np.isnan(sdfw))
sdfm = np.ma.masked_array(sdfm,np.isnan(sdfm));sudfm = np.ma.masked_array(sudfm,np.isnan(sdfm))

Eeta = np.ma.masked_array(Eeta,np.isnan(Eeta))
Eetad = np.ma.masked_array(Eetad,np.isnan(Eetad))
Eetadf = np.ma.masked_array(Eetadf,np.isnan(Eetadf))
Eetadfw = np.ma.masked_array(Eetadfw,np.isnan(Eetadfw))
Eetadfm = np.ma.masked_array(Eetadfm,np.isnan(Eetadfm))


## plotting
tit = 'Lat: ['+str(latlim[0])+','+str(latlim[1]) +'], Lon: [;' + str(lonlim[0]) + ',' + str(lonlim[1]) + ']'
figname =  str(latlim[0])+'-'+str(latlim[1])+'_'+str(lonlim[0])+'-'+str(lonlim[1])


# -2 and -3 slopes in the loglog space
ks2 = np.array([1.e-2,1.e-1])
Es2 = 1.e8*(ks2**(-2))
ks4 = np.array([1.e-2,1.e-1])
Es4 = 1.e-1*(ks4**(-4))
ks5 = np.array([1.e-2,1.e-1])
Es5 = 1.e-10*(ks5**(-5))
Es3 = 1.e-3*(ks2**(-3))


fig = plt.figure(facecolor='w', figsize=(12.,12.))
ax1 = fig.add_subplot(111)

ix,jx = Eeta.shape

for i in range(0,jx,1):
    if i == 0:
        ax1.loglog(k,Eeta[:,i]*(10**(i)), color=color1, linewidth=lw1,
                label=u'unfiltered')
        ax1.loglog(k,Eetad[:,i]*(10**(i)), color=color2, linewidth=lw1,
                label=u'de-trended')
        ax1.loglog(k,Eetadf[:,i]*(10**(i)) ,color=color3, linewidth=lw1,
                label=u'de-trended + filtered')
        ax1.loglog(k,Eetadfw[:,i]*(10**(i)) ,color=color4, linewidth=lw1,
                label=u'de-trended + filtered + windowed')
        ax1.loglog(km,Eetadfm[:,i]*(10**(i)) ,color=color5, linewidth=lw1,
                label=u'de-trended + filtered + mirrored')

    else:
        ax1.loglog(k,Eeta[:,i]*(10**(i)), color=color1, linewidth=lw1)
        ax1.loglog(k,Eetad[:,i]*(10**(i)), color=color2, linewidth=lw1)
        ax1.loglog(k,Eetadf[:,i]*(10**(i)) ,color=color3, linewidth=lw1)
        ax1.loglog(k,Eetadfw[:,i]*(10**(i)) ,color=color4, linewidth=lw1)
        ax1.loglog(km,Eetadfm[:,i]*(10**(i)) ,color=color5, linewidth=lw1)

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
plt.title(tit)

lg = plt.legend(loc=3,title= u'$\mathcal{SSH}$ spectrum', prop={'size':22}, numpoints=1)
lg.draw_frame(False)
my.leg_width(lg,5.)

plt.savefig('figs/spectra_ssh' + figname)

fig = plt.figure(facecolor='w', figsize=(12.,9.))

plt.plot(loni,s,'o',color=color1,markersize=14.,label='unfiltered')
plt.plot(loni,sd,'o',color=color2,markersize=14.,label='de-trended')
plt.plot(loni,sdf,'o',color=color3,markersize=14.,label='de-trended + filtered')
plt.plot(loni,sdfw,'o',color=color4,markersize=14.,label='de-trended + filtered + windowed')
plt.plot(loni,sdfm,'o',color=color5,markersize=14.,label='de-trended + filtered + mirrored')
lg = plt.legend(loc=3,borderaxespad=0.,title= u'Spectral slope [10,100] km', prop={'size':20}, numpoints=1)
lg.draw_frame(False)

plt.xlabel('Longitude')
plt.ylabel('Spectral slope')
plt.title(tit)
my.leg_width(lg,5.)
plt.ylim(-6,-2)
plt.xlim(lonlim[0]-.5,lonlim[1]+.5)
plt.savefig('figs/spectral_slope_vs_lon' + figname)

## save to compare against KE spectra
#fno = 'outputs/spectra_ssh'
#np.savez(fno, Eeta=Eeta, Eetaf=Eetaf, k=k)




