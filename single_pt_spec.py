def spec(A,dt):
    ix,jx = A.shape
    df = 1./jx
    fNy = 1./(jx*dt)
    an = np.fft.fft(A,axis=1)
    an = an[:,1:jx/2]
    E = 2*(an*an.conj())/df/(jx**2)  
    f = np.arange(1,jx/2)*df
    return E.mean(axis=0),f,df,fNy 

if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy as sp
    import scipy.signal
    import seawater.csiro as sw
    import aux_func_3dfields as my

    iz = 50   # vertical level [m]
    dt = 1    # time step      [h]

    series = np.load('single_pt_uvw.npz')
    
    # chop the series into 4 slices to increase statistical reliability
    N = 4
    L = series['time'].size/N
    uN,vN,wN=series['uN'].reshape(N,L),series['vN'].reshape(N,L),series['wN'].reshape(N,L)
    uS,vS,wS=series['uS'].reshape(N,L),series['vS'].reshape(N,L),series['wS'].reshape(N,L)

    # spectral window
    window = np.repeat(np.hanning(L),N).reshape(N,L)

    EuN,f,df,fNy=spec(uN,dt)
    EvN,_,_,_=spec(vN,dt)
    EwN,_,_,_=spec(wN,dt)

    EuS,f,df,fSy=spec(uS,dt)
    EvS,_,_,_=spec(vS,dt)
    EwS,_,_,_=spec(wS,dt)


