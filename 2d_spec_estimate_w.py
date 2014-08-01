
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


    iz = 3500   # vertical level [m]
    data_path = '/Users/crocha/Data/llc4320/w/'+str(iz)+'m/*'
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
    
    E = np.zeros((lats.size,lons.size,np.array(files).size))
    Ef = np.zeros((lats.size,lons.size,np.array(files).size))

    kk = 0
    for file in sorted(files[0:-1]):
        
        data = np.load(file)
        print kk

        ix,jx,kx = data['w'].shape
        wi = np.zeros((ix,jx,kx))

        for i in range(kx):
            interp_w = sp.interpolate.interp2d(lons,lats,data['w'][:,:,i],kind='linear')
            wi[:,:,i] = interp_w(loni[300,:],lati[:,300])

        E[:,:,kk],l,k,dl,dk,flNy,fkNy = spec_est2(wi,dy,dx,win=True)
        Ef[:,:,kk],lf,kf,dlf,dkf,flNyf,fkNyf = spec_est2_2(wi.mean(axis=2),dy,dx,win=True)

        kk = kk + 1
        del data, wi 

    E = E.mean(axis=2)
    Ef = Ef.mean(axis=2)

    # isotropic spectral estimate
    ki,li = np.meshgrid(k,l)
    K = np.sqrt(ki**2+li**2)
    K = np.ma.masked_array(K,K<1.e-10) 

    phi = np.math.atan2(dl,dk)
    dK = dk*np.cos(phi)
    Ki = np.arange(K.min(),K.max(),dK)
    Kw  = (Ki[1:]+Ki[0:-1])/2.
    dK2 = dK/2.

    Eiso = np.zeros(Kw.size)

    for i in range(Kw.size):
        f =  (K>=Kw[i]-dK2)&(K<Kw[i]+dK2)
        dtheta = (2*np.pi)/np.float(f.sum())
        Eiso[i] = ((E[f].sum()))*Kw[i]*dtheta

    ki,li = np.meshgrid(kf,lf)
    K = np.sqrt(ki**2+li**2)
    K = np.ma.masked_array(K,K<1.e-10) 

    phi = np.math.atan2(dlf,dkf)
    dK = dkf*np.cos(phi)
    Ki = np.arange(K.min(),K.max(),dK)
    Kwf  = (Ki[1:]+Ki[0:-1])/2.
    dK2 = dK/2.

    Eisof = np.zeros(Kwf.size)

    for i in range(Kwf.size):
        f =  (K>=Kwf[i]-dK2)&(K<Kwf[i]+dK2)
        dtheta = (2*np.pi)/np.float(f.sum())
        Eisof[i] = ((Ef[f].sum()))*Kwf[i]*dtheta

    fno='outputs/Eiso_w_'+str(iz)+'m'
    np.savez(fno,Eiso=Eiso,Eisof=Eisof,Kw=Kw,Kwf=Kwf)

