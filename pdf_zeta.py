import matplotlib.pyplot as plt
import numpy as np
import aux_func_3dfields as my

plt.close('all')

plt.rcParams.update({'font.size': 24, 'legend.handlelength'  : 1.5
    , 'legend.markerscale': 14., 'legend.linewidth': 3.})

color2 = '#6495ed'
color1 = '#ff6347'
color3 = '#8470ff'
color4 = '#3cb371'
lw1 = 3.

#
# settings
#

def gauss(x,mu,sig):
    g = (1./(np.sqrt(2*np.pi)*sig))*np.exp(-.5*(((x-mu)/sig)**2))
    return g

# compute moments
def moments(x,pdfx,n):
    dx = np.abs(x[1]-x[0])
    return ((x**n)*pdfx*dx).sum()

zeta = np.load('zeta_stats.npz')
bins = zeta['bin2'][:,0]
g0_zeta = gauss(bins,0,zeta['rms'][0])
g250_zeta = gauss(bins,0,zeta['rms'][30])

m1_0 = moments(bins,zeta['pdf'][:,0],1)
m2_0 = moments(bins,zeta['pdf'][:,0],2)
m3_0 = moments(bins,zeta['pdf'][:,0],3)
m4_0 = moments(bins,zeta['pdf'][:,0],4)

m1_250 = moments(bins,zeta['pdf'][:,33],1)
m2_250 = moments(bins,zeta['pdf'][:,33],2)
m3_250 = moments(bins,zeta['pdf'][:,33],3)
m4_250 = moments(bins,zeta['pdf'][:,33],4)

# compute skewness all depths
m3 = np.zeros(zeta['z'].size)
for i in range(m3.size):
    m3[i]=moments(bins,zeta['pdf'][:,i],3)


fig=plt.figure(facecolor='w', figsize=(10.,8.5))

plt.plot(bins,zeta['pdf'][:,0],color=color1,label='0 m, -1.22',linewidth=lw1,alpha=.6)
#plt.plot(zeta['bin2'],g0_zeta,'--',color=color1)

plt.plot(bins,zeta['pdf'][:,33],color=color2,label='300 m, -0.29',linewidth=lw1,alpha=.6)
#plt.plot(zeta['bin2'],g100_zeta,'--',color=color2)
plt.grid()
plt.xlabel(u'$\zeta/|f_0|$')
plt.ylabel(u'Probability density')
plt.xlim(-5,5)
lg = plt.legend(loc=1,title= u'Skewness', prop={'size':22}, numpoints=1)
lg.draw_frame(False)
my.leg_width(lg,5.)
plt.savefig('figs/pdf_zeta_0_250',bbox_inches='tight')

fig=plt.figure(facecolor='w', figsize=(10.,8.5))
plt.plot(m3,-zeta['z'],'ko',alpha=.6)
plt.ylim(2000,0)
plt.xlabel(u'Skewness of $\zeta/|f_0|$')
plt.ylabel('Depth [m]')
plt.grid()
plt.savefig('figs/skewness_zeta',bbox_inches='tight')

fig=plt.figure(facecolor='w', figsize=(10.,8.5))
plt.plot(zeta['rms'][0:81],-zeta['z'],'ko',alpha=.6)
plt.ylim(2000,0)
plt.xlabel(u'Rms of $\zeta/|f_0|$')
plt.ylabel('Depth [m]')
plt.grid()
plt.savefig('figs/rms_zeta',bbox_inches='tight')





