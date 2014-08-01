
import matplotlib.pyplot as plt
import scipy.signal
import scipy as sp
import scipy.io as io
import numpy as np
import glob   
import seawater.csiro as sw

import aux_func_3dfields as my

## reorder 3d array A to mimic sampling model
def reorder(A,Ai):
    l,m,n = Ai.shape
    Ai1 = Ai[:,:,0:n/2]
    Ai2 = np.flipud(Ai[:,:,n/2:])
    #Ai1 = np.flipud(Ai[:,:,0:n/2])
    #Ai2 = Ai[:,:,n/2:]

    l,m,n = Ai1.shape
    As1 = np.zeros((l,m,(n-l+1))) 
    As2 = np.zeros((l,m,(n-l+1))) 
    k=np.arange(n-l+1)
    for i in range(l):
        As1[i,:,:] = Ai1[i,:,k+i].T
        As2[i,:,:] = Ai2[i,:,k+i].T
    return np.append(As1,As2,axis=2)

plt.close('all')

plt.rcParams.update({'font.size': 24, 'legend.handlelength'  : 1.5
    , 'legend.markerscale': 14., 'legend.linewidth': 3.})

color2 = '#6495ed'
color1 = '#ff6347'
color3 = '#8470ff'
color4 = '#3cb371'
lw1 = 2

# GM spectrum mean N=WOA(0.160 m)
GM = io.loadmat('/Users/crocha/Dropbox/research/comparisons/GarrettMunkMatlab-master/GM_southern-ocean.mat',squeeze_me=True,struct_as_record=False)

## filter
# create filter (butterworth order 2, lowpass cutoff period 30 h)
b, a = sp.signal.butter(2, 1./24,btype='lowpass')

## data
iz = 50    # vertical level [m]
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

#Uf = sp.signal.filtfilt(b, a, my.rmean(U), axis=2)
#Vf = sp.signal.filtfilt(b, a, my.rmean(V), axis=2)
# Uf and Vf as daily average to be consistent with direction-averaged spectrum
U = U[:,:,0:-21]
V = V[:,:,0:-21]
ix,jx,kx=U.shape
Uf = np.zeros((ix,jx,kx/24))
Vf = np.zeros((ix,jx,kx/24))

k = 0
for i in range(0,kx,24):
    Uf[:,:,k]= np.nanmean(U[:,:,i:i+24],axis=2)
    Vf[:,:,k]= np.nanmean(V[:,:,i:i+24],axis=2)
    k = k+1

# linear interp onto 4.25 minute grid
ix,jx,kx = U.shape
Tdp = 2.2*24
dts = Tdp/ix
t = np.arange(kx)
ti = np.arange(0,kx-1,dts)
#ti = np.linspace(0,kx-1,13083)  # dts about Tdp/ix, Tdp [h] is the time is takes to cross the passage 
interp_ut = sp.interpolate.interp1d(t,U,kind='linear',axis=2)
interp_vt = sp.interpolate.interp1d(t,V,kind='linear',axis=2)
Ui = interp_ut(ti)
Vi = interp_vt(ti)

Us = reorder(U,Ui)[:,:,0::7]
Vs = reorder(V,Vi)[:,:,0::7]

lat = lat[ilat]
lon = lon[ilat]

dist,_ = sw.dist(lon,lat)
dist = np.append(0,np.cumsum(dist))

# block average every 5 km (as in the ADCP)
dxt = 5. # [km]
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

ix,jx,kx = Uf.shape    
window_sf = np.repeat(np.hanning(ix),jx).reshape(ix,jx)
window_sf = np.repeat(window_sf,kx).reshape(ix,jx,kx)

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

ix,jx,kx = Uf.shape
window_tf = np.repeat(np.hanning(kx),jx).reshape(kx,jx).T
window_tf = np.repeat(window_tf,ix).reshape(jx,kx,ix)
window_tf = np.transpose(window_tf,axes=(2,0,1))




## spectral estimates

# in space (meridional wavenumber)
Eu,k,dk,kNy = my.spec_est_meridional(my.rmean(U)*window_s,dx)
Ev,_,_,_ = my.spec_est_meridional(my.rmean(V)*window_s,dx)
Eus,_,_,_ = my.spec_est_meridional(my.rmean(Us)*window_ss,dx)
Evs,_,_,_ = my.spec_est_meridional(my.rmean(Vs)*window_ss,dx)
Eusb,ksb,_,_ = my.spec_est_meridional(my.rmean(Usb)*window_sb,dxt)
Evsb,ksb,_,_ = my.spec_est_meridional(my.rmean(Vsb)*window_sb,dxt)

Euf,_,_,_ = my.spec_est_meridional(Uf*window_sf,dx)
Evf,_,_,_ = my.spec_est_meridional(Vf*window_sf,dx)

# in time
dt = 1. # [h]
Eu_t,f,df,fNy = my.spec_est_time(my.rmean(U)*window_t,dt)
Ev_t,f,df,fNy = my.spec_est_time(my.rmean(V)*window_t,dt)
Euf_t,f,df,fNy = my.spec_est_time(Uf*window_tf,dt)
Evf_t,f,df,fNy = my.spec_est_time(Vf*window_tf,dt)

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

## plotting

## (spatially) mean spectra
Eu = np.nanmean(Eu,axis=1)
Euf = np.nanmean(Euf,axis=1)
Eus = np.nanmean(Eus,axis=1)
Eusb = np.nanmean(Eusb,axis=1)
Ev = np.nanmean(Ev,axis=1)
Evf = np.nanmean(Evf,axis=1)
Evsb = np.nanmean(Evsb,axis=1)
Evs = np.nanmean(Evs,axis=1)

# Eu/Ev on 10-100km range
l = 1./k
fl = (l<150)&(l>40)              # [km]
r_m = (Eu[fl]/Ev[fl]).mean()
r_d = (Eu[fl]/Ev[fl]).std()/np.sqrt(fl.sum())  # standard error

rs = (Eus[fl]/Evs[fl]).mean()
rs_d = (Eus[fl]/Evs[fl]).std()/np.sqrt(fl.sum()) 

rf = (Euf[fl]/Evf[fl]).mean()
rf_d = (Euf[fl]/Evf[fl]).std()/np.sqrt(fl.sum()) 

l = 1./ksb
fl = (l<150)&(l>40)              # [km]
rsb = (Eusb[fl]/Evsb[fl]).mean()
rsb_d = (Eusb[fl]/Evsb[fl]).std()/np.sqrt(fl.sum())

# estimate spectral error
N = 200 # number of independent realizations
Eul, Euu, stdu = my.spec_error2(Eu, N)
Eufl, Eufu, stduf = my.spec_error2(Euf, N)
Eusl, Eusu,stdus = my.spec_error2(Eus, N)
Eusbl, Eusbu, stdusb = my.spec_error2(Eusb, N)
Evl, Evu,stdv = my.spec_error2(Ev, N)
Evfl, Evfu,stdvf = my.spec_error2(Evf, N)
Evsl, Evsu, stdvs = my.spec_error2(Evs, N)
Evsbl, Evsbu, stdvsb = my.spec_error2(Evsb, N)

# spectral slopes
kmin = 1./150  # cycles/km
kmax = 1./40   # cycles/km

slu,slua = my.spectral_slope(k,Eu,kmin,kmax,stdu)
sluf,slufa = my.spectral_slope(k,Euf,kmin,kmax,stduf)
slus,slusa = my.spectral_slope(k,Eus,kmin,kmax,stdus)
slusb,slusba = my.spectral_slope(ksb,Eusb,kmin,kmax,stdusb)
slv,slva = my.spectral_slope(k,Ev,kmin,kmax,stdv)
slvf,slvfa = my.spectral_slope(k,Evf,kmin,kmax,stdvf)
slvs,slvsa = my.spectral_slope(k,Evs,kmin,kmax,stdvs)
slvsb,slvsba = my.spectral_slope(ksb,Evsb,kmin,kmax,stdvsb)

# Plotting
ks = np.array([1.e-3,1])
Es2 = .2e-4*(ks**(-2))
Es3 = .3e-5*(ks**(-3))

fig = plt.figure(facecolor='w', figsize=(11.,12.))
ax1 = fig.add_subplot(111)

ax1.fill_between(k,Eul,Euu, color=color1, alpha=0.25)
ax1.fill_between(k,Evl,Evu, color=color2, alpha=0.25)
#ax1.fill_between(k,Eufl,Eufu, color=color3, alpha=0.25)
#ax1.fill_between(k,Evfl,Evfu, color=color4, alpha=0.25)
ax1.fill_between(k,Eusl,Eusu, color=color3, alpha=0.25)
ax1.fill_between(k,Evsl,Evsu, color=color4, alpha=0.25)
ax1.fill_between(ksb,Eusbl,Eusbu, color=color3, alpha=0.25)
ax1.fill_between(ksb,Evsbl,Evsbu, color=color4, alpha=0.25)
    
ax1.set_xscale('log'); ax1.set_yscale('log')

ax1.loglog(k,Eu, color=color1, linewidth=lw1,label='across-track')
ax1.loglog(k,Ev, color=color2, linewidth=lw1,label='along-track')
#ax1.loglog(k,Euf, color=color3, linewidth=lw1,label='across-track')
#ax1.loglog(k,Evf, color=color4, linewidth=lw1,label='along-track')

ax1.loglog(k,Eus,color=color3, linewidth=lw1,label='across-track, sampled')
ax1.loglog(k,Evs,color=color4, linewidth=lw1,label='along-track, sampled')

ax1.loglog(ksb,Eusb,'--', color=color3, linewidth=lw1,label='across-track, sampled + block-averaged')
ax1.loglog(ksb,Evsb,'--', color=color4, linewidth=lw1,label='along-track, sampled + block-averaged')

ax1.loglog(ks,Es2,'--', color='k',linewidth=2.,alpha=.5)
ax1.loglog(ks,Es3,'--', color='k',linewidth=2.,alpha=.5)

# GM
#ax1.loglog( GM['k'],GM['EGM']/2.,'--', color='k',linewidth=2.,alpha=.5)
ax1.axis((1./(1000),1./4,.4e-5,10))

plt.text(0.0011686481894527252, 5.4101984795026086,u'k$^{-2}$')
plt.text(0.0047869726184615827, 5.5118532543417871,u'k$^{-3}$')
#plt.text(0.002, 0.05,u'$\mathcal{GM}$')


#ax1.loglog([1/rd1,1/rd1],[1.e-3,.07],'--',linewidth=2.,color=('.5'))
#plt.text(0.048131084925677289, 0.032358609276653932,u'Rd$_1$')

plt.xlabel('Wavenumber [cycles/km]')
plt.ylabel(u'Spectral density [(m$^{2}$ s$^{-2}$)/(cycles/km)]')

lg = plt.legend(loc=3,title= u'0 m, 400 DOF', prop={'size':22}, numpoints=1)
lg.draw_frame(False)
my.leg_width(lg,5.)

plt.text(0.171, 6.7,'(a)')

tit = 'figs/spec_uv_total_'+np.str(iz)+'m'
plt.savefig(tit,bbox_inches='tight')


# filtered
fig = plt.figure(facecolor='w', figsize=(11.,12.))
ax1 = fig.add_subplot(111)

ax1.fill_between(k,Eufl,Eufu, color=color1, alpha=0.25)
ax1.fill_between(k,Evfl,Evfu, color=color2, alpha=0.25)

ax1.set_xscale('log'); ax1.set_yscale('log')

ax1.loglog(k,Euf, color=color1, linewidth=lw1,label='across-track')
ax1.loglog(k,Evf, color=color2, linewidth=lw1,label='along-track')

ax1.loglog(ks,Es2,'--', color='k',linewidth=2.,alpha=.5)
ax1.loglog(ks,Es3,'--', color='k',linewidth=2.,alpha=.5)

# GM
#ax1.loglog( GM['k'],GM['EGM']/2.,'--', color='k',linewidth=2.,alpha=.5)
ax1.axis((1./(1000),1./4,.4e-5,10))

plt.text(0.0011686481894527252, 5.4101984795026086,u'k$^{-2}$')
plt.text(0.0047869726184615827, 5.5118532543417871,u'k$^{-3}$')
#plt.text(0.002, 0.05,u'$\mathcal{GM}$')


#ax1.loglog([1/rd1,1/rd1],[1.e-3,.07],'--',linewidth=2.,color=('.5'))
#plt.text(0.048131084925677289, 0.032358609276653932,u'Rd$_1$')

plt.xlabel('Wavenumber [cycles/km]')
plt.ylabel(u'Spectral density [(m$^{2}$ s$^{-2}$)/(cycles/km)]')

lg = plt.legend(loc=3,title= u'0 m, 400 DOF', prop={'size':22}, numpoints=1)
lg.draw_frame(False)
my.leg_width(lg,5.)

plt.text(0.171, 6.7,'(b)')

tit = 'figs/spec_uv_filtered_'+np.str(iz)+'m'
plt.savefig(tit,bbox_inches='tight')

