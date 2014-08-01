
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

def spec_est2_2(A,d1,d2,win=True):

    """    computes 2D spectral estimate of A
           obs: the returned array is fftshifted
           and consistent with the f1,f2 arrays
           d1,d2 are the sampling rates in rows,columns   """
    
    import numpy as np

    l1,l2 = A.shape
    df1 = 1./(l1*d1)
    df2 = 1./(l2*d2)
    f1Ny = 1./(2*d1)
    f2Ny = 1./(2*d2)

    f1 = np.arange(-f1Ny,f1Ny-df1,df1)
    f2 = np.arange(-f2Ny,f2Ny,df2)
    
    if win == True:
        wx = np.matrix(np.hanning(l1))
        wy =  np.matrix(np.hanning(l2))
        window_s = np.array(wx.T*wy)
    else:
        window_s = np.ones((l1,l2))

    an = np.fft.fft2(A*window_s,axes=(0,1))
    E = (an*an.conjugate()) / (df1*df2) / ((l1*l2)**2)
    E = np.fft.fftshift(E)

    return np.real(E),f1,f2,df1,df2,f1Ny,f2Ny




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
    
    Euw = np.zeros((lats.size,lons.size,np.array(files).size))
    Evw = np.zeros((lats.size,lons.size,np.array(files).size))
    Euwf = np.zeros((lats.size,lons.size,np.array(files).size))
    Evwf = np.zeros((lats.size,lons.size,np.array(files).size))
    Eum = np.zeros((2*lats.size,2*lons.size,np.array(files).size))
    Evm = np.zeros((2*lats.size,2*lons.size,np.array(files).size))

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

        Euw[:,:,kk],lw,kw,dlw,dkw,flNyw,fkNyw = spec_est2(ui,dy,dx,win=True)
        Evw[:,:,kk],_,_,_,_,_,_ = spec_est2(vi,dy,dx,win=True)
        Euwf[:,:,kk],lwf,kwf,dlwf,dkwf,flNywf,fkNywf = spec_est2_2(ui.mean(axis=2),dy,dx,win=True)
        Evwf[:,:,kk],lwf,kwf,dlwf,dkwf,flNywf,fkNywf = spec_est2_2(vi.mean(axis=2),dy,dx,win=True)

        # mirror to become doubly-periodic
        ui2 = np.append(ui,np.flipud(ui),axis=0)
        ui2 = np.append(ui2,np.fliplr(ui2),axis=1)
        vi2 = np.append(vi,np.flipud(vi),axis=0)
        vi2 = np.append(vi2,np.fliplr(vi2),axis=1)

        Eum[:,:,kk],lm,km,dlm,dkm,flNym,fkNym = spec_est2(ui2,dy,dx,win=False)
        Evm[:,:,kk],_,_,_,_,_,_ = spec_est2(vi2,dy,dx,win=False)

        kk = kk + 1
        del data, ui,vi,ui2,vi2,ix,jx,kx

    Euw = Euw.mean(axis=2)
    Evw = Evw.mean(axis=2)
    Ew = (Euw+Evw)/2

    # isotropic spectral estimate
    ki,li = np.meshgrid(kw,lw)
    K = np.sqrt(ki**2+li**2)
    K = np.ma.masked_array(K,K<1.e-10) 

    phi = np.math.atan2(dlw,dkw)
    dK = dkw*np.cos(phi)
    Ki = np.arange(K.min(),K.max(),dK)
    K_w  = (Ki[1:]+Ki[0:-1])/2.
    dK2 = dK/2.

    Eiso_w = np.zeros(K_w.size)

    for i in range(K_w.size):
        f =  (K>=K_w[i]-dK2)&(K<K_w[i]+dK2)
        dtheta = (2*np.pi)/np.float(f.sum())
        Eiso_w[i] = ((Ew[f].sum()))*K_w[i]*dtheta

    Euwf = Euwf.mean(axis=2)
    Evwf = Evwf.mean(axis=2)
    Ewf = (Euwf+Evwf)/2

    # isotropic spectral estimate
    ki,li = np.meshgrid(kwf,lwf)
    K = np.sqrt(ki**2+li**2)
    K = np.ma.masked_array(K,K<1.e-10) 

    phi = np.math.atan2(dlwf,dkwf)
    dK = dkwf*np.cos(phi)
    Ki = np.arange(K.min(),K.max(),dK)
    K_wf  = (Ki[1:]+Ki[0:-1])/2.
    dK2 = dK/2.

    Eiso_wf = np.zeros(K_wf.size)

    for i in range(K_w.size):
        f =  (K>=K_wf[i]-dK2)&(K<K_wf[i]+dK2)
        dtheta = (2*np.pi)/np.float(f.sum())
        Eiso_wf[i] = ((Ewf[f].sum()))*K_wf[i]*dtheta


    Eum = Eum.mean(axis=2)/16. 
    Evm = Evm.mean(axis=2)/16.
    Em = (Eum+Evm)/2.

    # isotropic spectral estimate
    ki,li = np.meshgrid(km,lm)
    K = np.sqrt(ki**2+li**2)
    K = np.ma.masked_array(K,K<1.e-10) 

    phi = np.math.atan2(dlm,dkm)
    dK = dkm*np.cos(phi)
    Ki = np.arange(K.min(),K.max(),dK)
    K_m  = (Ki[1:]+Ki[0:-1])/2.
    dK2 = dK/2.

    Eiso_m = np.zeros(K_m.size)

    for i in range(K_w.size):
        f =  (K>=K_m[i]-dK2)&(K<K_m[i]+dK2)
        dtheta = (2*np.pi)/np.float(f.sum())
        Eiso_m[i] = ((Em[f].sum()))*K_m[i]*dtheta

    fno='outputs/Eiso_KE_'+str(iz)+'m'
    np.savez(fno,Eiso_w=Eiso_w,Eiso_wf=Eiso_wf,K_w=K_w,K_wf=K_wf,Eiso_m=Eiso_m,K_m=K_m)

    # integrating in k or l
    Elw = 2*(Ew.sum(axis=1)*dkw)[lw.size/2:]
    Ekw = 2*(Ew.sum(axis=0)*dlw)[kw.size/2:]
    kxw = kw[kw.size/2:]
    lyw = lw[lw.size/2:]
    Elm = 2*(Em.sum(axis=1)*dkm)[lm.size/2:]
    Ekm = 2*(Em.sum(axis=0)*dlm)[km.size/2:]
    kxm = km[km.size/2:]
    lym = lm[lm.size/2:]

    # plotting
    ks = np.array([1.e-3,1])
    Es2 = .2e-4*(ks**(-2))
    Es3 = .5e-6*(ks**(-3))

    fig = plt.figure(facecolor='w', figsize=(12.,8.5))
    plt.loglog(kxw,Ekw,color='b',label='Ek',linewidth=4.,alpha=.5)
    plt.loglog(lyw,Elw,color='g',label='El',linewidth=4.,alpha=.5)
    plt.loglog(K_w,Eiso_w,color='m',linewidth=4.,alpha=.5)
    plt.loglog(kxm,Ekm,'--',color='b',label='Ek',linewidth=4.,alpha=.5)
    plt.loglog(lym,Elm,'--',color='g',label='El',linewidth=4.,alpha=.5)
    plt.loglog(K_m,Eiso_m,'--',color='m',linewidth=4.,alpha=.5)
    plt.loglog(ks,Es2,'--', color='k',linewidth=2.,alpha=.5)
    plt.loglog(ks,Es3,'--', color='k',linewidth=2.,alpha=.5)
    plt.text(0.0011686481894527252, 5.4101984795026086/2.,u'k$^{-2}$')
    plt.text(0.0047869726184615827, 5.5118532543417871/2.,u'k$^{-3}$')
    plt.axis((1./(1000),1.,.4e-5,10))
    plt.ylabel('Spectral density  [(m$^2$s$^{-2}$)/(cycles/km)]')
    plt.xlabel('Wavenumber  [cycles/km]')
    plt.savefig('figs/Eiso_uv')
