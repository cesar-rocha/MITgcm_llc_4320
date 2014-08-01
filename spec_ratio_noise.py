def spec_est2(A,d1,d2,win=True):

    """    computes 2D spectral estimate of A
           obs: the returned array is fftshifted
           and consistent with the f1,f2 arrays
           d1,d2 are the sampling rates in rows,columns   """
    
    import numpy as np

    l1,l2,l3 = A.shape
    df1 = 1./(l1*d1)
    df2 = 1./(l2*d2)
    f1Ny = 1./(2*d1)
    f2Ny = 1./(2*d2)

    f1 = np.arange(-f1Ny,f1Ny,df1)
    f2 = np.arange(-f2Ny,f2Ny,df2)
    
    if win == True:
        wx = np.matrix(np.hanning(l1))
        wy =  np.matrix(np.hanning(l2))
        window_s = np.repeat(np.array(wx.T*wy),l3).reshape(l1,l2,l3)
    else:
        window_s = np.ones((l1,l2,l3))

    an = np.fft.fft2(A*window_s,axes=(0,1))
    E = (an*an.conjugate()) / (df1*df2) / ((l1*l2)**2)
    E = np.fft.fftshift(E)
    E = E.mean(axis=2)

    return np.real(E),f1,f2,df1,df2,f1Ny,f2Ny

def ps(u,v,dx,dy):

    """ decompose the vector field (u,v) into potential (up,vp)
        and solenoidal (us,vs) fields using 2D FT a la Smith JPO 2008 """

    ix,jx,kx = u.shape
    dl = 1./(ix*dy)
    dk = 1./(jx*dx)
    kNy = 1./(2*dx)
    lNy = 1./(2*dy)
    k = np.arange(-kNy,kNy,dk)
    k = np.fft.fftshift(k)
    l = np.arange(-lNy,lNy,dl)
    l = np.fft.fftshift(l)
    K,L = np.meshgrid(k,l)
    THETA = (np.arctan2(L,K))
    THETA = np.repeat(THETA,kx).reshape(ix,jx,kx)

    U = np.fft.fft2(u,axes=(0,1))
    V = np.fft.fft2(v,axes=(0,1))

    P = U*np.cos(THETA) + V*np.sin(THETA)
    S = -U*np.sin(THETA) + V*np.cos(THETA)

    # back to physical space
    up = np.real(np.fft.ifft2(P*np.cos(THETA),axes=(0,1)))
    vp = np.real(np.fft.ifft2(P*np.sin(THETA),axes=(0,1)))

    us = np.real(np.fft.ifft2(-S*np.sin(THETA),axes=(0,1)))
    vs = np.real(np.fft.ifft2(S*np.cos(THETA),axes=(0,1)))

    return up,vp,us,vs

if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal
    import scipy as sp
    import glob, os 
    import seawater.csiro as sw
    import aux_func_3dfields as my

    plt.close('all')
    plt.rcParams.update({'font.size': 24, 'legend.handlelength'  : 1.5
        , 'legend.markerscale': 14., 'legend.linewidth': 3.})

    color1 = '#ff6347'
    color2 = '#6495ed'
    color3 = '#8470ff'
    color4 = '#3cb371'

    lw1=3
    aph=.7

    # load obs-based 2d
    aux=np.load('outputs/Eiso.npz')
    Eobs = aux['E'][0::2]
    k = aux['k'][0::2]
    dk = k[2]-k[1]

    # create a spectral ramp based on spectrum of observed 
    #   motions in Drake Passage
    Esyn = (1./(k**3))
    Esyn = (Esyn*(Eobs.max()/Esyn.max())) 
    Esyn = Esyn*((Eobs/Esyn).mean())  # across-track

    k = np.append(-np.flipud(k),k)
    ki,li = np.meshgrid(k,k)
    K = np.sqrt((ki**2) + (li**2))

    Ki = np.sqrt(2.)*aux['k'][0::2]
    Ki2 = (Ki[1:]+Ki[0:-1])/2.
    dK = Ki[2]-Ki[1]

    # create isotropic spectral ramp
    Esyn2D = np.zeros(K.shape)
    for i in range(Ki.size):
        
        if i==(Ki.size):
            f =  (K>=(Ki[i-1]))
            Esyn2D[f] = Esyn[i]/Ki[i]/(2*np.pi)
        else:    
            f =  (K>=(Ki[i-1]))&(K<(Ki[i]))
            Esyn2D[f] = Esyn[i]/(Ki[i])/(2*np.pi)

    # assume Eu = Ev
    
    # create random phase
    n = 2
    ix,jx=Esyn2D.shape
    pha = 2*np.pi*(np.random.rand(ix*jx*n).reshape(ix,jx,n))  # assume phases are correlated
    ii = np.complex(0,1)
    pha = np.cos(pha) + ii*np.sin(pha)
    Esyn2D = np.repeat(Esyn2D,n).reshape(ix,jx,n)
    Esyn2D = np.sqrt(Esyn2D/1.e5)*pha

    # plot 2D spectral ramp
    fig = plt.figure(facecolor='w', figsize=(12.,10.))
    plt.contourf(k,k,np.log10(Esyn2D)[:,:,1], 25,cmap='Spectral_r')
    cb = plt.colorbar()
    cb.set_label(u'Spectral density  [(m$^2$s$^{-2}$)/(cycles/km)$^2$]')
    plt.axis('equal')
    plt.xlim(-.1,.1)
    plt.ylim(-.1,.1)

    plt.xlabel('Zonal wavenumber [cycles/km]')
    plt.ylabel('Meridional wavenumber [cycles/km]')
    plt.savefig('figs/2d_Esyn')        


    # back to fourier coefs
    an = np.fft.fftshift(Esyn2D)*((dk*dk)*((ix*jx)**2))

    # back to physical space (i.e., create synthetic u and v)
    U = np.fft.ifft2(an,axes=(0,1))
    u = np.real(U)
    v = np.imag(U)

    up,vp,us,vs = ps(u,v,1.,1.)


    # filter divergent part of the flow (cut-off about 20 km)
    nx = 20
    ny = 20
    x, y = np.mgrid[-nx/2:nx/2, -ny/2:ny/2]
    
    rx = 10.
    ry = 10.

    g = np.exp( -  ( (x/rx)**2 + (y/ry)**2 ) )
    g = g/g.sum()

    ix,jx,kx = up.shape
    upf = np.zeros(up.shape)
    vpf = np.zeros(vp.shape)

    for i in range(kx):
        upm = up[:,:,i].mean()
        upi = up[:,:,i]
        upf[:,:,i] = sp.signal.convolve2d(upi-upm,g, mode='same') + upm 
        vpm = vp[:,:,i].mean()
        vpi = vp[:,:,i]
        vpf[:,:,i] = sp.signal.convolve2d(vpi-vpm,g, mode='same') + vpm 

    # the divergent part associated with 'small scales'
    ud = up-upf
    vd = vp-vpf

    # make the flow slightly div.
    nd = 1.
    uu = nd*ud + us
    vv = nd*vd + vs

    # add white noise and divergent flow at small scales
    nn= 0.05
    Au = nn*us.std()
    Av = nn*vs.std()
    ix,jx,kx = us.shape
    nu = Au*(np.random.randn(ix*jx*kx)).reshape(ix,jx,kx)
    nv = Av*(np.random.randn(ix*jx*kx)).reshape(ix,jx,kx)

    # total flow
    Eut,kut,dku,kuNy = my.spec_est_meridional(u,1.)
    Evt,kvt,dkv,kvNy = my.spec_est_meridional(v,1.)
    Evt=Evt.mean(axis=1)
    Eut=Eut.mean(axis=1)

    # horizontally non-divergent
    Eu,ku,dku,kuNy = my.spec_est_meridional(us,1.)
    Ev,kv,dkv,kvNy = my.spec_est_meridional(vs,1.)
    Ev=Ev.mean(axis=1)
    Eu=Eu.mean(axis=1)

    # add divergent flow at small scales and random noise
    Eun,_,_,_ = my.spec_est_meridional(uu+nu,1.)
    Evn,_,_,_ = my.spec_est_meridional(vv+nv,1.)
    Evn=Evn.mean(axis=1)
    Eun=Eun.mean(axis=1)

    Ek = (np.sum(Esyn2D,axis=1)*dk)[k.size/2:]
    El = (np.sum(Esyn2D,axis=0)*dk)[k.size/2:]
    k = k[k.size/2:]

    # mask very low and very high wavenumbers
    L = 1./ku
    fm = ((L<=5)|(L>=200))
    Eu = np.ma.masked_array(Eu,fm)
    Ev = np.ma.masked_array(Ev,fm)
    Eun = np.ma.masked_array(Eun,fm)
    Evn = np.ma.masked_array(Evn,fm)

    # compute ratios in an arbitrary range
    f = ((L>=10)&(L<=100))
    rn = ((Eun/Evn)[f]).mean()
    r = ((Eu/Ev)[f]).mean()

    # plotting
    ks = np.array([1.e-3,1])
    Es3 = .7e-6*(ks**(-3))

    fig = plt.figure(facecolor='w', figsize=(12.,10.))
    plt.loglog(kut,Eut,color=color1,label='Zonal',linewidth=4.,alpha=.5)
    plt.loglog(kvt,Evt,color=color2,label='Meridional',linewidth=4.,alpha=.5)
    plt.loglog(ks,Es3,'--',color='k',linewidth=4.,alpha=.5)
    plt.text(0.012, 0.53,u'$\kappa^{-3}$')
    plt.axis((1./(400),1./2.5,1.e-6,1.))
    plt.ylabel('Spectral density  [m$^2$/(cycles/km)]')
    plt.xlabel('Wavenumber  [cycles/km]')
    lg = plt.legend(loc=1,title= u'', prop={'size':22}, numpoints=1)
    lg.draw_frame(False)
    my.leg_width(lg,5.)
    figtit = 'figs/EuEv_synthetic_total.png'
    plt.savefig(figtit,format='png', bbox_inches='tight')

    fig = plt.figure(facecolor='w', figsize=(12.,10.))
    plt.loglog(ku,Eu,color=color1,label='Zonal',linewidth=4.,alpha=.5)
    plt.loglog(kv,Ev,color=color2,label='Meridional',linewidth=4.,alpha=.5)
    plt.loglog(ku,Eun,'--',color=color1,linewidth=4.,alpha=.5)
    plt.loglog(kv,Evn,'--',color=color2,linewidth=4.,alpha=.5)
    plt.loglog(ks,Es3,'--',color='k',linewidth=4.,alpha=.5)
    plt.text(0.012, 0.53,u'$\kappa^{-3}$')
    plt.axis((1./(400),1./2.5,1.e-6,1.))
    plt.ylabel('Spectral density  [m$^2$/(cycles/km)]')
    plt.xlabel('Wavenumber  [cycles/km]')
    lg = plt.legend(loc=1,title= u'', prop={'size':22}, numpoints=1)
    lg.draw_frame(False)
    my.leg_width(lg,5.)
    figtit = 'figs/EuEv_synthetic.png'
    plt.savefig(figtit,format='png', bbox_inches='tight')



