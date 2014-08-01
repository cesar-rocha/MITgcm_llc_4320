
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

lw1 = 2

## filter
# create filter (butterworth order 2, lowpass cutoff period 30 h)
b, a = sp.signal.butter(2, 1./30,btype='lowpass')

## data
iz = 50    # vertical level [m]
fni = 'subsets/'+str(iz)+'m_uv_daily_full.npz'
data = np.load(fni)

## settings
lat = data['lat'][:,10]
lon = data['lon'][250,:]

# find and mask bad data
u = data['u']
v = data['v']

#du = np.diff(u,axis=2).sum(axis=2) == 0    # points where variable doesn't
                                           #   change over time
#dv = np.diff(v,axis=2).sum(axis=2) == 0

#u[du,:] = np.nan
#v[dv,:] = np.nan

# avoid boundaries and ice
ilat = ((lat>-62)&(lat<-55))
#dist,ang = sw.dist(lon,lati)
#dx = dist.mean()   # [km], about 1 km

u = u[ilat,:]
v = v[ilat,:]
lat = lat[ilat]

# for projection onto regular grid
lati = np.linspace(lat.min(),lat.max(),lat.size)
loni = np.linspace(lon.min(),lon.max(),lon.size)

ix,jx,kx = u.shape

ui = np.zeros(u.shape)
vi = np.zeros(v.shape)

for i in range(kx):
    interp_u = sp.interpolate.interp2d(lon,lat,u[:,:,i],kind='linear')
    ui[:,:,i] = interp_u(loni,lati)
    interp_v = sp.interpolate.interp2d(lon,lat,v[:,:,i],kind='linear')
    vi[:,:,i] = interp_v(loni,lati)

lon,lat = np.meshgrid(lon,lat)
loni,lati = np.meshgrid(loni,lati)

dist,ang = sw.dist(loni[100,:],lati[100,:])
dx = dist.mean()   # [km], about 1 km
dist,ang = sw.dist(loni[:,100],lati[:,100])
dy = dist.mean()   # [km], about 1 km

def spec_est2(A,d1,d2):

    """    computes 2D spectral estimate of A
           obs: the returned array is fftshifted
           and consistent with the f1,f2 arrays
           d1,d2 are the sampling rates in rows,columns   """

    l1,l2,L3 = A.shape
    df1 = 1./(l1*d1)
    df2 = 1./(l2*d2)
    f1Ny = 1./(2*d1)
    f2Ny = 1./(2*d2)

    f1 = np.arange(-f1Ny,f1Ny,df1)
    f2 = np.arange(-f2Ny,f2Ny,df2)

    an = np.fft.fft2(A,axes=(0,1))
    E = (an*an.conjugate()) / (df1*df2) / ((l1*l2)**2)
    E = np.fft.fftshift(E)
    E = E.mean(axis=2)

    return E,f1,f2,df1,df2,f1Ny,f2Ny


## spectral window in space
ix,jx,kx = ui.shape    
wx = np.matrix(np.hanning(ix))
wy =  np.matrix(np.hanning(jx))
window_s = np.repeat(np.array(wx.T*wy),kx).reshape(ix,jx,kx)

Eu,l,k,dl,dk,flNy,fkNy = spec_est2(ui*window_s,dy,dx)
Ev,l,k,dl,dk,flNy,fkNy = spec_est2(vi*window_s,dy,dx)

# mask the mean before plotting
#Eu[:,k.size/2] = np.nan
#Ev[:,k.size/2] = np.nan
#Eu[l.size/2,:] = np.nan
#Ev[l.size/2,:] = np.nan
#Eu = np.ma.masked_array(Eu,np.isnan(Eu))
#Ev = np.ma.masked_array(Ev,np.isnan(Ev))

## plotting
E = (Eu+Ev)/2

fig = plt.figure(facecolor='w', figsize=(12.,10.))
plt.contourf(k,l,np.log10(E), 25,cmap='Spectral_r')
cb = plt.colorbar()
cb.set_label(u'Spectral density  [(m$^2$s$^{-2}$)/(cycles/km)$^2$]')
plt.axis('equal')
plt.xlim(-.2,.2)
plt.ylim(-.2,.2)

plt.xlabel('Zonal wavenumber [cycles/km]')
plt.ylabel('Meridional wavenumber [cycles/km]')
plt.savefig('figs/2d_ke_spec')

## plotting two slices
Ex = E[l.size/2,k.size/2:]
Ey = E[l.size/2::,k.size/2]
kx = k[k.size/2:]
ly = l[l.size/2:]

# reference slopes
ks = np.array([1.e-3,1])
Es2 = .2e-4*(ks**(-2))
Es3 = .5e-6*(ks**(-3))

fig = plt.figure(facecolor='w', figsize=(12.,10.))
plt.loglog(kx,kx*Ex,label='Ex')
plt.loglog(ly,ly*Ey,label='Ey')
plt.loglog(ks,Es2,'--', color='k',linewidth=2.,alpha=.5)
plt.loglog(ks,Es3,'--', color='k',linewidth=2.,alpha=.5)
plt.text(0.0011686481894527252, 5.4101984795026086,u'k$^{-2}$')
plt.text(0.0047869726184615827, 5.5118532543417871,u'k$^{-3}$')
plt.savefig('figs/2slices')

## try to go POLAR
ki,li = np.meshgrid(k,l)
K = np.sqrt(ki**2+li**2)
K = np.ma.masked_array(K,K<1.e-10)

phi = np.math.atan2(dl,dk)
dK = dk*np.cos(phi)
Ki = np.arange(K.min(),K.max(),dK)
Ki2  = (Ki[1:]+Ki[0:-1])/2.
dK2 = dK/2.

Eiso = np.zeros(Ki2.size)

for i in range(Ki2.size):
    f =  (K>= Ki2[i]-dK2)&(K<=Ki2[i]+dK2)
    dtheta = (2*np.pi)/np.float(f.sum())
    Eiso[i] = ((E[f].sum()))*Ki2[i]*dtheta

# Try integrating in k or l
El = 2*(E.sum(axis=1)*dk)[l.size/2:]
Ek = 2*(E.sum(axis=0)*dl)[k.size/2:]

fig = plt.figure(facecolor='w', figsize=(12.,8.5))
plt.loglog(kx,Ek,label='Ek',linewidth=4.,alpha=.5)
plt.loglog(ly,El,label='El',linewidth=4.,alpha=.5)
plt.loglog(Ki2,Eiso,color='m',linewidth=4.,alpha=.5)
plt.loglog(ks,Es2,'--', color='k',linewidth=2.,alpha=.5)
plt.loglog(ks,Es3,'--', color='k',linewidth=2.,alpha=.5)
plt.text(0.0011686481894527252, 5.4101984795026086/2.,u'k$^{-2}$')
plt.text(0.0047869726184615827, 5.5118532543417871/2.,u'k$^{-3}$')
plt.axis((1./(1000),1.,.4e-5,10))
plt.ylabel('Spectral density  [(m$^2$s$^{-2}$)/(cycles/km)]')
plt.xlabel('Wavenumber  [cycles/km]')
plt.savefig('figs/EkEl')

# save for comparison
np.savez('outputs/Eiso_filtered',k=Ki2,E=Eiso)



