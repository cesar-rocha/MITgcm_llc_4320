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
ke = np.load('outputs/Eiso_KE_0m.npz')
kend = np.load('outputs/Eiso_KE_nondiv_div.npz')
ssh  = np.load('outputs/Eiso_eta_small.npz')
w = np.load('outputs/Eiso_w.npz')

# constant
A = (((9.81)/sw.cor(-59.25))**2)
k = 2*np.pi*(ssh['K_w']/1.e3)/np.sqrt(2)   # [cycles/m]


Lw = 1./ke['K_w']
fw = ((Lw>=7.)&(Lw<=300))
E = ke['Eiso_w'][fw]
ke_kw = ke['K_w'][fw]

Lw = 1./ke['K_m']
fw = ((Lw>=7.)&(Lw<=300))
Em = ke['Eiso_m'][fw]
ke_km = ke['K_m'][fw]

Lw = 1./ke['K_wf']
fw = ((Lw>=7.)&(Lw<=300))
Ef = ke['Eiso_wf'][fw]
ke_kwf = ke['K_wf'][fw]

Lw = 1./ssh['K_w']
fe = ((Lw>=7.)&(Lw<=300))
Eeta = ssh['Ew'][fe]
ssh_kw = ssh['K_w'][fe]
kw = 2*np.pi*(ssh['K_w'][fe]/1.e3)/np.sqrt(2)   # [cycles/m]

Lw = 1./ssh['K_wf']
fe = ((Lw>=7.)&(Lw<=300))
Eetaf = ssh['Ewf'][fe]
kwf = 2*np.pi*(ssh['K_wf'][fe]/1.e3)/np.sqrt(2)   # [cycles/m]
ssh_kwf = ssh['K_wf'][fe]

Lw = 1./kend['K_nd']
fw = ((Lw>=7.)&(Lw<=300))
End = kend['Eiso_nd'][fw]
ke_kwnd = kend['K_nd'][fw]

Lw = 1./kend['K_d']
fw = ((Lw>=7.)&(Lw<=300))
Ed = kend['Eiso_d'][fw]
ke_kwd = kend['K_d'][fw]

Lw = 1./ssh['K_g']
fw = ((Lw>=7.)&(Lw<=300))
Eg = ssh['Eiso_g'][fw]
ke_kg = ssh['K_g'][fw]

Lw = 1./ssh['K_gf']
fw = ((Lw>=7.)&(Lw<=300))
Egf = ssh['Eiso_gf'][fw]
ke_kgf = ssh['K_gf'][fw]

Lw = 1./w['Kw']
fw = ((Lw>=7.)&(Lw<=300))
Ew = w['Eiso'][fw]
w_k = w['Kw'][fw]

Lw = 1./w['Kwf']
fw = ((Lw>=7.)&(Lw<=300))
Ewf = w['Eisof'][fw]
w_kf = w['Kwf'][fw]




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

El,Eu = spec_error(E,ke_kw,sn,4*sn,8*sn)
Egl,Egu = spec_error(Eg,ke_kg,sn,4*sn,8*sn)
Egfl,Egfu = spec_error(Egf,ke_kgf,sn,4*sn,8*sn)
Eml, Emu = spec_error(2.*Em,ke_km,sn,4*sn,8*sn)
Efl, Efu = spec_error(Ef,ke_kwf,sn,4*sn,8*sn)
Eetal, Eetau = spec_error(Eeta,ssh_kw,sn,4*sn,8*sn)
Eetafl, Eetafu = spec_error(Eetaf,ssh_kwf,sn,4*sn,8*sn)
Ewl, Ewu = spec_error(Ew,w_k,sn,4*sn,8*sn)
Ewfl, Ewfu = spec_error(Ewf,w_kf,sn,4*sn,8*sn)



# plotting
ks = np.array([1.e-3,1])
Es3 = .7e-6*(ks**(-3))
Es2 = .3e-4*(ks**(-2))

fig = plt.figure(facecolor='w', figsize=(12.,12.))
ax1 = fig.add_subplot(111)

ax1.fill_between(ke_kw,El,Eu, color=color1, alpha=0.25)
ax1.fill_between(ke_kwf,Efl,Efu, color=color2, alpha=0.25)
#ax1.fill_between(ke_km,Eml,Emu, color='m', alpha=0.25)
#ax1.fill_between(ssh_kw,A*(kw**2)*Eetal,A*(kw**2)*Eetau, color=color3, alpha=0.25)
#ax1.fill_between(ssh_kwf,A*(kwf**2)*Eetafl,A*(kwf**2)*Eetafu, color=color4, alpha=0.25)

ax1.fill_between(ke_kg,Egl,Egu, color=color3, alpha=0.25)
ax1.fill_between(ke_kgf,Egfl,Egfu, color=color4, alpha=0.25)

ax1.set_xscale('log'); ax1.set_yscale('log')

ax1.loglog(ke_kw,E,color=color1, linewidth=lw1,
        label=u'KE')

#ax1.loglog(ke_km,2*Em,color='m', linewidth=lw1,
#        label=u'KEm')

ax1.loglog(ke_kwf,Ef,color=color2, linewidth=lw1,
        label=u'KE, daily-averaged')

ax1.loglog(ke_kwnd,End,color='k', linewidth=lw1,
        label=u'KE, nondiv')

ax1.loglog(ke_kg,Eg,color=color3, linewidth=lw1,
        label=u'KE_g')


ax1.loglog(ke_kgf,Egf,color=color4, linewidth=lw1,
        label=u'KE_g, daily-averaged')


#ax1.loglog(ke_kwd,Ed,'--',color='k', linewidth=lw1,label=u'KE, nondiv')

ax1.loglog(ke_kwd,Ed,'--',color='k', linewidth=lw1,label=u'KE, div')

ax1.loglog(ke_kwd,Ed+End,'--',color='y', linewidth=lw1,label=u'KE, div+non-div')

ax1.loglog(w_k,1.e4*Ew,color='c', linewidth=lw1,
        label=u'$10^4$ x <w$^2$>')


#ax1.loglog(ssh_kw,A*(kw**2)*Eeta,color=color3, linewidth=lw1,
#       label=u'$\propto$ $\kappa^2<$ SSH$^2$$>$')

#ax1.loglog(ssh_kwf,A*(kwf**2)*Eetaf,color=color4, linewidth=lw1,
#        label=u'$\propto$ $\kappa^2<$ SSH$^2$$>$, daily-averaged')
plt.loglog(ks,Es3,'--',color='k',linewidth=2.,alpha=.5)
plt.loglog(ks,Es2,'--',color='k',linewidth=2.,alpha=.5)
plt.text(0.0056, 5.4,u'$\kappa^{-3}$')
plt.text(0.0025, 5.4,u'$\kappa^{-2}$')
plt.axis((1./(500),1./3,1.e-4,10))
lg = plt.legend(loc=3,title= u'direction-averaged spectrum', prop={'size':22}, numpoints=1)
lg.draw_frame(False)
my.leg_width(lg,5.)
plt.ylabel('Spectral density  [m$^2$/(cycles/km)]')
plt.xlabel('Wavenumber  [cycles/km]')
plt.savefig('figs/isotropic_spectrum_uv',bbox_inches='tight')




fig = plt.figure(facecolor='w', figsize=(12.,12.))
ax1 = fig.add_subplot(111)

ax1.fill_between(w_k,Ewl,Ewu, color='c', alpha=0.25)
#ax1.fill_between(w_kf,Ewfl,Ewfu, color='g', alpha=0.25)

ax1.loglog(w_k,Ew,color='c', linewidth=lw1,
        label=u'<w$^2$>')

#ax1.loglog(w_kf,Ewf,color='g', linewidth=lw1,
#        label=u'<w$^2$>, daily-average')


lg = plt.legend(loc=3,title= u'direction-averaged spectrum', prop={'size':22}, numpoints=1)
lg.draw_frame(False)
my.leg_width(lg,5.)
plt.ylabel('Spectral density  [m$^2$/(cycles/km)]')
plt.savefig('figs/isotropic_spectrum_w',bbox_inches='tight')



## compute the amount of "supertidal" energy
dk_ke = ke_kw[1]-ke_kw[0]
dk_kef = ke_kwf[1]-ke_kwf[0]
var_ke = E.sum()*dk_ke
var_kef = Ef.sum()*dk_kef
rf_ke = var_kef/var_ke
st_var_ke = 1-rf_ke


Lw = 1./ssh_kw  
fw = (Lw<=90)

dk_ssh = ssh_kw[1]-ssh_kw[0]
dk_sshf = ssh_kwf[1]-ssh_kwf[0]
var_ssh = Eeta[fw].sum()*dk_ssh
var_sshf = Eetaf[fw].sum()*dk_sshf
rf_ssh = var_sshf/var_ssh
st_var_ssh = 1-rf_ssh





