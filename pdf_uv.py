import matplotlib.pyplot as plt
import numpy as np
import aux_func_3dfields as my

plt.close('all')

plt.rcParams.update({'font.size': 24, 'legend.handlelength'  : 1.5
    , 'legend.markerscale': 1.5, 'legend.linewidth': 3.})

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

v = np.load('vel_stats.npz')
bins = v['binu2'][:,0]
g0_v = gauss(bins,0,v['vrms'][0])
g300_v = gauss(bins,0,v['vrms'][33])
g0_u = gauss(bins,0,v['urms'][0])
g300_u = gauss(bins,0,v['urms'][33])

mv1_0 = moments(bins,v['pdfv'][:,0],1)
mv2_0 = moments(bins,v['pdfv'][:,0],2)
mv3_0 = moments(bins,v['pdfv'][:,0],3)
mv4_0 = moments(bins,v['pdfv'][:,0],4)
mv1_250 = moments(bins,v['pdfv'][:,33],1)
mv2_250 = moments(bins,v['pdfv'][:,33],2)
mv3_250 = moments(bins,v['pdfv'][:,33],3)
mv4_250 = moments(bins,v['pdfv'][:,33],4)

mu1_0 = moments(bins,v['pdfu'][:,0],1)
mu2_0 = moments(bins,v['pdfu'][:,0],2)
mu3_0 = moments(bins,v['pdfu'][:,0],3)
mu4_0 = moments(bins,v['pdfu'][:,0],4)
mu1_250 = moments(bins,v['pdfu'][:,33],1)
mu2_250 = moments(bins,v['pdfu'][:,33],2)
mu3_250 = moments(bins,v['pdfu'][:,33],3)
mu4_250 = moments(bins,v['pdfu'][:,33],4)

# compute skewness all depths
m3 = np.zeros(v['z'].size)
for i in range(m3.size):
    m3[i]=moments(bins,v['pdfv'][:,i],3)

fig=plt.figure(facecolor='w', figsize=(10.,8.5))

plt.plot(bins,v['pdfu'][:,0],color=color1,label='0 m',linewidth=lw1,alpha=.6)
plt.plot(bins,v['pdfv'][:,0],'--',color=color1,linewidth=lw1,alpha=.6)
#plt.plot(w['bin2'],g0_w,'--',color=color1)

plt.plot(bins,v['pdfu'][:,33],color=color2,label='300 m',linewidth=lw1,alpha=.6)
plt.plot(bins,v['pdfv'][:,33],'--',color=color2,linewidth=lw1,alpha=.6)
#plt.plot(w['bin2'],g100_w,'--',color=color2)
plt.grid()
plt.xlabel(u'$vel.$ [m s$^{-1}$]')
plt.ylabel(u'Probability density [m$^{-1}$ s]')
plt.xlim(-1.5,1.5)
lg = plt.legend(loc=1,title= u'', prop={'size':22}, numpoints=1)
lg.draw_frame(False)
my.leg_width(lg,5.)
plt.savefig('figs/pdf_u_0_250',bbox_inches='tight')

fig=plt.figure(facecolor='w', figsize=(10.,8.5))
plt.plot(m3,-v['z'],'ko',alpha=.6)
plt.ylim(200,0)
plt.xlabel(u'Skewness of $u$')
plt.ylabel('Depth [m]')
plt.grid()
plt.xlim(-1.,1.)
plt.savefig('figs/skewness_u',bbox_inches='tight')


fig=plt.figure(facecolor='w', figsize=(10.,8.5))
plt.plot(v['urms'],-v['z'],'ko',label='u',alpha=.6)
plt.plot(v['vrms'],-v['z'],'mo',label='v',alpha=.6)
plt.ylim(5000,0)
plt.xlabel(u'Rms [m s$^{-1}$]')
plt.ylabel('Depth [m]')
plt.grid()
lg = plt.legend(loc=4,title= u'', prop={'size':22}, numpoints=1)
lg.draw_frame(False)
my.leg_width(lg,5.)
plt.savefig('figs/rms_vel',bbox_inches='tight')





