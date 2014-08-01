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

w = np.load('w_stats.npz')
bins = w['bin2'][:,0]
g0_w = gauss(bins,0,w['rms'][7])
g250_w = gauss(bins,0,w['rms'][30])

m1_0 = moments(bins,w['pdf'][:,7],1)
m2_0 = moments(bins,w['pdf'][:,7],2)
m3_0 = moments(bins,w['pdf'][:,7],3)
m4_0 = moments(bins,w['pdf'][:,7],4)

m1_250 = moments(bins,w['pdf'][:,30],1)
m2_250 = moments(bins,w['pdf'][:,30],2)
m3_250 = moments(bins,w['pdf'][:,30],3)
m4_250 = moments(bins,w['pdf'][:,30],4)

# compute skewness all depths
m3 = np.zeros(w['z'].size)
for i in range(m3.size):
    m3[i]=moments(bins,w['pdf'][:,i],3)


fig=plt.figure(facecolor='w', figsize=(10.,8.5))

plt.plot(bins,w['pdf'][:,7],color=color1,label='10 m, 0.0',linewidth=lw1,alpha=.6)
#plt.plot(w['bin2'],g0_w,'--',color=color1)

plt.plot(bins,w['pdf'][:,30],color=color2,label='250 m, 0.0',linewidth=lw1,alpha=.6)
#plt.plot(w['bin2'],g100_w,'--',color=color2)
plt.grid()
plt.xlabel(u'$w$ [m s$^{-1}$]')
plt.ylabel(u'Probability density [m$^{-1}$ s]')
plt.xlim(-0.001,0.001)
lg = plt.legend(loc=1,title= u'Skewness', prop={'size':22}, numpoints=1)
lg.draw_frame(False)
my.leg_width(lg,5.)
plt.savefig('figs/pdf_w_10_250',bbox_inches='tight')

fig=plt.figure(facecolor='w', figsize=(10.,8.5))
plt.plot(m3,-w['z'],'ko',alpha=.6)
plt.ylim(200,0)
plt.xlabel(u'Skewness of $w$')
plt.ylabel('Depth [m]')
plt.grid()
plt.xlim(-1.,1.)
plt.savefig('figs/skewness_w',bbox_inches='tight')


fig=plt.figure(facecolor='w', figsize=(10.,8.5))
plt.plot(w['rms'],-w['z'],'ko',alpha=.6)
plt.ylim(5000,0)
plt.xlabel(u'Rms of $w$ [m s$^{-1}$]')
plt.ylabel('Depth [m]')
plt.grid()
plt.savefig('figs/rms_w',bbox_inches='tight')





