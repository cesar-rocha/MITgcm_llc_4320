
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

    f1 = np.arange(-f1Ny,f1Ny-df1,df1)
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
    
    wx = np.matrix(np.hanning(l1))
    wy =  np.matrix(np.hanning(l2))
    window_s = np.array(wx.T*wy)

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
    data_path = '/Users/crocha/Data/llc4320/eta/2011*'
    grid_path= '/Users/crocha/Data/llc4320/eta/'

    grid = np.load(grid_path+'grid.npz')
    lons = grid['lon'][300,:]
    lats = grid['lat'][:,300]

    flats = (lats>-62.)
    flons = (lons<-49.)

    lons=lons[flons]
    lats=lats[flats]

    # projection onto regular grid
    lati = np.linspace(lats.min(),lats.max(),lats.size)
    loni = np.linspace(lons.min(),lons.max(),lons.size)
    loni,lati = np.meshgrid(loni,lati)

#    griduv = np.load('grid_uv_i.npz')
#    loni,lati = griduv['lon'],griduv['lat']
#    dx,dy = griduv['dx'],griduv['dy']

    dist,ang = sw.dist(loni[300,:],lati[300,:])
    dx = dist.mean()   # [km], about 1 km
    dist,ang = sw.dist(loni[:,300],lati[:,300])
    dy = dist.mean()   # [km], about 1 km

    files = sorted(glob.glob(data_path), key=os.path.getmtime) 
    
    Eetaw = np.zeros((lats.size,lons.size,np.array(files).size))
    Eetawf = np.zeros((lats.size,lons.size,np.array(files).size))

    Eugw = np.zeros((lats.size,lons.size,np.array(files).size))
    Eugwf = np.zeros((lats.size,lons.size,np.array(files).size))

    Evgw = np.zeros((lats.size,lons.size,np.array(files).size))
    Evgwf = np.zeros((lats.size,lons.size,np.array(files).size))

    # constant for geostrophic vel.
    f = np.repeat(sw.cor(lats),lons.size).reshape(lats.size,lons.size)
    C = 9.81/f

    kk = 0
    for file in sorted(files):
        
        data = np.load(file)
        print kk

        ix,jx,kx = data['eta'].shape
        etai = np.zeros((lats.size,lons.size,kx))
        ugi = np.zeros((lats.size,lons.size,kx))
        vgi = np.zeros((lats.size,lons.size,kx))

        # mask bad data
        ETA = data['eta']
        deta = np.diff(ETA,axis=2).sum(axis=2) == 0  # points where variable doesn't
                                                     #   change over time
        #ETA[deta] = np.nan
        #ETA = np.ma.masked_array(ETA,np.isnan(ETA))

        for i in range(kx):
            etaaux=ETA[:,flons,i]
            etaaux=etaaux[flats,:]
            interp_eta = sp.interpolate.interp2d(lons,lats,etaaux,kind='linear')
            etai[:,:,i] = interp_eta(loni[300,:],lati[:,300])
            ugi[:,:,i],b =  -C*sp.gradient(etai[:,:,i],dy*1.e3)
            a,vgi[:,:,i] =  C*sp.gradient(etai[:,:,i],dx*1.e3)

        Eetaw[:,:,kk],lw,kw,dlw,dkw,flNyw,fkNyw = spec_est2(etai,dy,dx)
        Eetawf[:,:,kk],lwf,kwf,dlwf,dkwf,flNywf,fkNywf = spec_est2_2(etai.mean(axis=2),dy,dx)

        Eugw[:,:,kk],l,k,dl,dk,flNy,fkNy = spec_est2(ugi,dy,dx)
        Eugwf[:,:,kk],_,_,_,_,_,_ = spec_est2_2(ugi.mean(axis=2),dy,dx)

        Evgw[:,:,kk],_,_,_,_,_,_ = spec_est2(vgi,dy,dx)
        Evgwf[:,:,kk],_,_,_,_,_,_ = spec_est2_2(vgi.mean(axis=2),dy,dx)

        kk = kk + 1
        del data,etai,ix,jx,kx,ETA,ugi,vgi,etaaux


    Ew = Eetaw.mean(axis=2)
    Ewf = Eetawf.mean(axis=2)

    Eu = Eugw.mean(axis=2)
    Euf = Eugwf.mean(axis=2)

    Ev = Evgw.mean(axis=2)
    Evf = Evgwf.mean(axis=2)

    Eg = (Eu+Ev)/2.
    Egf = (Euf+Evf)/2.

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

    ki,li = np.meshgrid(kwf,lwf)
    K = np.sqrt(ki**2+li**2)
    K = np.ma.masked_array(K,K<0.0023) 

    phi = np.math.atan2(dlwf,dkwf)
    dK = dkwf*np.cos(phi)
    Ki = np.arange(K.min(),K.max(),dK)
    K_wf  = (Ki[1:]+Ki[0:-1])/2.
    dK2 = dK/2.

    Eiso_wf = np.zeros(K_wf.size)

    for i in range(K_wf.size):
        f =  (K>=K_wf[i]-dK2)&(K<K_wf[i]+dK2)
        dtheta = (2*np.pi)/np.float(f.sum())
        Eiso_wf[i] = ((Ewf[f].sum()))*K_wf[i]*dtheta

    # isotropic spectral estimate
    ki,li = np.meshgrid(k,l)
    K = np.sqrt(ki**2+li**2)
    K = np.ma.masked_array(K,K<1.e-10) 

    phi = np.math.atan2(dl,dk)
    dK = dk*np.cos(phi)
    Ki = np.arange(K.min(),K.max(),dK)
    K_g  = (Ki[1:]+Ki[0:-1])/2.
    dK2 = dK/2.

    Eiso_g = np.zeros(K_g.size)

    for i in range(K_g.size):
        f =  (K>=K_g[i]-dK2)&(K<K_g[i]+dK2)
        dtheta = (2*np.pi)/np.float(f.sum())
        Eiso_g[i] = ((Eg[f].sum()))*K_g[i]*dtheta

    # isotropic spectral estimate
    ki,li = np.meshgrid(k,l)
    K = np.sqrt(ki**2+li**2)
    K = np.ma.masked_array(K,K<1.e-10) 

    phi = np.math.atan2(dl,dk)
    dK = dk*np.cos(phi)
    Ki = np.arange(K.min(),K.max(),dK)
    K_gf  = (Ki[1:]+Ki[0:-1])/2.
    dK2 = dK/2.

    Eiso_gf = np.zeros(K_g.size)

    for i in range(K_gf.size):
        f =  (K>=K_gf[i]-dK2)&(K<K_gf[i]+dK2)
        dtheta = (2*np.pi)/np.float(f.sum())
        Eiso_gf[i] = ((Egf[f].sum()))*K_gf[i]*dtheta


    # save for comparison
    fno = 'outputs/Eiso_eta_small'
    np.savez(fno,Ew=Eiso_w,Ewf=Eiso_wf,K_w = K_w,K_wf=K_wf, Eiso_g=Eiso_g,Eiso_gf=Eiso_gf,
            K_g=K_g,K_gf=K_gf)

    # integrating in k or l
    Elw = 2*(Ew.sum(axis=1)*dkw)[lw.size/2:]
    Ekw = 2*(Ew.sum(axis=0)*dlw)[kw.size/2:]
    kxw = kw[kw.size/2:]
    lyw = lw[lw.size/2:]
    Elwf = 2*(Ewf.sum(axis=1)*dkw)[lw.size/2:]
    Ekwf = 2*(Ewf.sum(axis=0)*dlw)[kw.size/2:]

    # mask low wavenumbers
    kref = 0.0022
    fw = (K_w > kref); K_w = K_w[fw]; Eiso_w = Eiso_w[fw]
    fw = (K_wf > kref); K_wf = K_wf[fw]; Eiso_wf = Eiso_wf[fw]
    fw = (kxw > kref); kxw = kxw[fw]; Ekw = Ekw[fw]
    fw = (lyw > kref); lyw = lyw[fw]; Elw = Elw[fw]




    # plotting
    ks = np.array([1.e-3,1])
    Es2 = .2e-7*(ks**(-3))
    Es3 = .5e-11*(ks**(-5))

    fig = plt.figure(facecolor='w', figsize=(12.,8.5))
    plt.loglog(kxw,Ekw,color='b',label='Ek',linewidth=4.,alpha=.5)
    plt.loglog(lyw,Elw,color='g',label='El',linewidth=4.,alpha=.5)
    plt.loglog(K_w,Eiso_w,color='m',linewidth=4.,alpha=.5)
    plt.loglog(K_wf,Eiso_wf,color='r',linewidth=4.,alpha=.5)
    plt.loglog(ks,Es2,'--', color='k',linewidth=2.,alpha=.5)
    plt.loglog(ks,Es3,'--', color='k',linewidth=2.,alpha=.5)
    plt.text(0.0011686481894527252, 5.4101984795026086/2.,u'k$^{-3}$')
    plt.text(0.0047869726184615827, 5.5118532543417871/2.,u'k$^{-5}$')
    plt.axis((1./(1000),1.,.4e-7,10))
    plt.ylabel('Spectral density  [m$^2$/(cycles/km)]')
    plt.xlabel('Wavenumber  [cycles/km]')
    plt.savefig('figs/Eiso_eta')


