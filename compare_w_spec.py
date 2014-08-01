import matplotlib.pyplot as plt
import scipy.signal
import scipy as sp
import numpy as np
import glob   
import seawater.csiro as sw

import aux_func as my

plt.close('all')

plt.rcParams.update({'font.size': 24, 'legend.handlelength'  : 1.5
    , 'legend.markerscale': 14., 'legend.linewidth': 3.})

color2 = '#6495ed'
color1 = '#ff6347'
color3 = '#8470ff'
color4 = '#3cb371'
lw1 = 3

#kef = np.load('outputs/Eiso_filtered.npz')
w50 = np.load('outputs/Eiso_w_50m.npz')
w500 = np.load('outputs/Eiso_w_500m.npz')
w1000 = np.load('outputs/Eiso_w_1000m.npz')
w3500 = np.load('outputs/Eiso_w_3500m.npz')

Lw = 1./w50['Kw']
fw = ((Lw>=7.)&(Lw<=300))
Ew50 = w50['Eiso'][fw]
k50 = w50['Kw'][fw]
Lw = 1./w50['Kwf']
fw = ((Lw>=7.)&(Lw<=300))
Ewf50 = w50['Eisof'][fw]
kf50 = w50['Kwf'][fw]

Lw = 1./w500['Kw']
fw = ((Lw>=7.)&(Lw<=300))
Ew500 = w500['Eiso'][fw]
k500 = w500['Kw'][fw]
Lw = 1./w500['Kwf']
fw = ((Lw>=7.)&(Lw<=300))
Ewf500 = w500['Eisof'][fw]
kf500 = w500['Kwf'][fw]

Lw = 1./w1000['Kw']
fw = ((Lw>=7.)&(Lw<=300))
Ew1000 = w1000['Eiso'][fw]
k1000 = w1000['Kw'][fw]
Lw = 1./w1000['Kwf']
fw = ((Lw>=7.)&(Lw<=300))
Ewf1000 = w1000['Eisof'][fw]
kf1000 = w1000['Kwf'][fw]

Lw = 1./w3500['Kw']
fw = ((Lw>=7.)&(Lw<=300))
Ew3500 = w3500['Eiso'][fw]
k3500 = w3500['Kw'][fw]
Lw = 1./w3500['Kwf']
fw = ((Lw>=7.)&(Lw<=300))
Ewf3500 = w3500['Eisof'][fw]
kf3500 = w3500['Kwf'][fw]

# formal errors
sn = 11  # from decorrelation time scale
ci = 0.95

def spec_error(E,k,sn1,sn2,sn3):
    ''' spectral error with different dof at different ranges'''
    El = np.zeros(E.size);Eu = np.zeros(E.size);sni = np.zeros(E.size)
    Lw = 1./k
    sni[(Lw>=100.)]=sn1
    sni[((Lw<100.)&(Lw>=20.))]=sn2
    sni[(Lw<20.)]=sn3
    for i in range(E.size):
        El[i], Eu[i], cdf, pdf = my.spec_error(E[i],sni[i],ci)
    return El, Eu

Ew50l, Ew50u = spec_error(Ew50,k50,sn,4*sn,8*sn)
Ew50fl, Ewf50u = spec_error(Ewf50,kf50,sn,4*sn,8*sn)
Ew500l, Ew500u = spec_error(Ew500,k500,sn,4*sn,8*sn)
Ew500fl, Ewf500u = spec_error(Ewf500,kf500,sn,4*sn,8*sn)
Ew1000l, Ew1000u = spec_error(Ew1000,k1000,sn,4*sn,8*sn)
Ew1000fl, Ewf1000u = spec_error(Ewf1000,kf1000,sn,4*sn,8*sn)
Ew3500l, Ew3500u = spec_error(Ew3500,k3500,sn,4*sn,8*sn)
Ew3500fl, Ewf3500u = spec_error(Ewf3500,kf3500,sn,4*sn,8*sn)


fig = plt.figure(facecolor='w', figsize=(12.,12.))
ax1 = fig.add_subplot(111)

ax1.fill_between(k50,Ew50l,Ew50u, color='c', alpha=0.25)
ax1.fill_between(k500,Ew500l,Ew500u, color='m', alpha=0.25)
ax1.fill_between(k1000,Ew1000l,Ew1000u, color='k', alpha=0.25)
ax1.fill_between(k3500,Ew3500l,Ew3500u, color='b', alpha=0.25)

ax1.loglog(k50,Ew50,color='c', linewidth=lw1,label=u'50 m')
ax1.loglog(k500,Ew500,color='m', linewidth=lw1,label=u'500 m')
ax1.loglog(k1000,Ew1000,color='k', linewidth=lw1,label=u'1000 m')
ax1.loglog(k3500,Ew3500,color='b', linewidth=lw1,label=u'3500 m')

lg = plt.legend(loc=3,title= u'direction-averaged spectrum', prop={'size':22}, numpoints=1)
lg.draw_frame(False)
my.leg_width(lg,5.)
plt.ylabel('Spectral density  [m$^2$/(cycles/km)]')
plt.xlabel('Wavenumber  [cycles/km]')
plt.savefig('figs/isotropic_spectrum_w',bbox_inches='tight')

