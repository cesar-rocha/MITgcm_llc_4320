
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

    iz = 0     # vertical level [m]
    data_path = '/Users/crocha/Data/llc4320/uv/'+str(iz)+'m/*'
    grid_path= '/Users/crocha/Data/llc4320/uv/'

    grid = np.load(grid_path+'grid.npz')
    lons = grid['lon'][300,:]
    lats = grid['lat'][:,300]
    # projection onto regular grid
    lati = np.linspace(lats.min(),lats.max(),lats.size)
    loni = np.linspace(lons.min(),lons.max(),lons.size)
    loni,lati = np.meshgrid(loni,lati)

    dist,ang = sw.dist(loni[300,:],lati[300,:])
    dx = dist.mean()   # [km], about 1 km
    dist,ang = sw.dist(loni[:,300],lati[:,300])
    dy = dist.mean()   # [km], about 1 km

    files = sorted(glob.glob(data_path), key=os.path.getmtime) 
    
    Eund = np.zeros((lats.size,lons.size,np.array(files).size))
    Evnd = np.zeros((lats.size,lons.size,np.array(files).size))
    Eud = np.zeros((lats.size,lons.size,np.array(files).size))
    Evd = np.zeros((lats.size,lons.size,np.array(files).size))

    # save loni and lati in order to put SSH in the same grid
    lonuv,latuv = loni[300,:],lati[:,300]
    np.savez('grid_uv_i',lon=lonuv,lat=latuv,dx=dx,dy=dy)

    kk = 0
    for file in sorted(files):
        
        data = np.load(file)
        print kk

        ix,jx,kx = data['u'].shape
        ui = np.zeros((ix,jx,kx))
        vi = np.zeros((ix,jx,kx))

        for i in range(kx):
            interp_u = sp.interpolate.interp2d(lons,lats,data['u'][:,:,i],kind='linear')
            ui[:,:,i] = interp_u(loni[300,:],lati[:,300])
            interp_v = sp.interpolate.interp2d(lons,lats,data['v'][:,:,i],kind='linear')
            vi[:,:,i] = interp_v(loni[300,:],lati[:,300])

        # apply spectral window before decomposing the flow
        l1,l2,l3 = ui.shape
        wx = np.matrix(np.hanning(l1))
        wy =  np.matrix(np.hanning(l2))
        window_s = np.repeat(np.array(wx.T*wy),l3).reshape(l1,l2,l3)

        # decompose the flow
        #up,vp,us,vs = ps(ui*window_s,vi*window_s,dx,dy)
        up,vp,us,vs = ps(ui,vi,dx,dy)

        Eund[:,:,kk],lnd,knd,dlnd,dknd,flNynd,fkNynd = spec_est2(us,dy,dx,win=True)
        Evnd[:,:,kk],_,_,_,_,_,_ = spec_est2(vs,dy,dx,win=True)

        Eud[:,:,kk],ld,kd,dld,dkd,flNyd,fkNyd = spec_est2(up,dy,dx,win=True)
        Evd[:,:,kk],_,_,_,_,_,_ = spec_est2(vp,dy,dx,win=True)

        kk = kk + 1
        del data, ui,vi,ix,jx,kx,us,vs,up,vp

    Eund = Eund.mean(axis=2)
    Evnd = Evnd.mean(axis=2)
    End = (Eund+Evnd)/2

    Eud = Eud.mean(axis=2)
    Evd = Evd.mean(axis=2)
    Ed = (Eud+Evd)/2

    # isotropic spectral estimate
    ki,li = np.meshgrid(knd,lnd)
    K = np.sqrt(ki**2+li**2)
    K = np.ma.masked_array(K,K<1.e-10) 

    phi = np.math.atan2(dlnd,dknd)
    dK = dknd*np.cos(phi)
    Ki = np.arange(K.min(),K.max(),dK)
    K_nd  = (Ki[1:]+Ki[0:-1])/2.
    dK2 = dK/2.

    Eiso_nd = np.zeros(K_nd.size)

    for i in range(K_nd.size):
        f =  (K>=K_nd[i]-dK2)&(K<K_nd[i]+dK2)
        dtheta = (2*np.pi)/np.float(f.sum())
        Eiso_nd[i] = ((End[f].sum()))*K_nd[i]*dtheta

    # isotropic spectral estimate
    ki,li = np.meshgrid(kd,ld)
    K = np.sqrt(ki**2+li**2)
    K = np.ma.masked_array(K,K<1.e-10) 

    phi = np.math.atan2(dld,dkd)
    dK = dkd*np.cos(phi)
    Ki = np.arange(K.min(),K.max(),dK)
    K_d  = (Ki[1:]+Ki[0:-1])/2.
    dK2 = dK/2.

    Eiso_d = np.zeros(K_d.size)

    for i in range(K_d.size):
        f =  (K>=K_d[i]-dK2)&(K<K_d[i]+dK2)
        dtheta = (2*np.pi)/np.float(f.sum())
        Eiso_d[i] = ((Ed[f].sum()))*K_d[i]*dtheta

    fno='outputs/Eiso_KE_nondiv_div'
    np.savez(fno,Eiso_d=Eiso_d,K_d=K_d,Eiso_nd=Eiso_nd,K_nd=K_nd)

    # integrating in k or l
    Elm = 2*(Em.sum(axis=1)*dkm)[lm.size/2:]
    Ekm = 2*(Em.sum(axis=0)*dlm)[km.size/2:]
    kxm = kw[km.size/2:]
    lym = lw[lm.size/2:]

    # plotting
    ks = np.array([1.e-3,1])
    Es2 = .2e-4*(ks**(-2))
    Es3 = .5e-6*(ks**(-3))

    fig = plt.figure(facecolor='w', figsize=(12.,8.5))
    plt.loglog(kxw,Ekw,color='b',label='Ek',linewidth=4.,alpha=.5)
    plt.loglog(lyw,Elw,color='g',label='El',linewidth=4.,alpha=.5)
    plt.loglog(K_w,Eiso_w,color='m',linewidth=4.,alpha=.5)
    plt.loglog(ks,Es2,'--', color='k',linewidth=2.,alpha=.5)
    plt.loglog(ks,Es3,'--', color='k',linewidth=2.,alpha=.5)
    plt.text(0.0011686481894527252, 5.4101984795026086/2.,u'k$^{-2}$')
    plt.text(0.0047869726184615827, 5.5118532543417871/2.,u'k$^{-3}$')
    plt.axis((1./(1000),1.,.4e-5,10))
    plt.ylabel('Spectral density  [(m$^2$s$^{-2}$)/(cycles/km)]')
    plt.xlabel('Wavenumber  [cycles/km]')
    plt.savefig('figs/Eiso_uv_nondiv')
