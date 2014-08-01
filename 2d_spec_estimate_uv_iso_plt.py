

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

    # read and continue from here
    fni='outputs/2d_spec_uv.npz'
    spec = np.load(fni)

    Ew=spec['Ew'];Em=spec['Em'];kw=spec['kw'];km=spec['km'];lw=spec['lw'];lm=spec['lm'] 
    dkw=spec['dkw'];dlw=spec['dlw'];dkm=spec['dkm'];dlm=spec['dlm'] 

    Ew = np.ma.masked_array(Ew,Ew<1e-14)
    Em = np.ma.masked_array(Em,Em<1e-14)


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
        f =  (K>= K_w[i]-dK2)&(K<K_w[i]+dK2)
        dtheta = (2*np.pi)/np.float(f.sum())
        Eiso_w[i] = ((Ew[f].sum()))*K_w[i]*dtheta

    ki,li = np.meshgrid(km,lm)
    K = np.sqrt(ki**2+li**2)
    K = np.ma.masked_array(K,K<1.e-10) 
    phi = np.math.atan2(dlm,dkm)
    dK = dkm*np.cos(phi)
    Ki = np.arange(K.min(),K.max(),dK)
    K_m  = (Ki[1:]+Ki[0:-1])/2.
    dK2 = dK/2.
    Eiso_m = np.zeros(K_m.size)
    for i in range(K_m.size):
        f =  (K>= K_m[i]-dK2)&(K<K_m[i]+dK2)
        dtheta = (2*np.pi)/np.float(f.sum())
        Eiso_m[i] = ((Em[f].sum()))*K_m[i]*dtheta

    # integrating in k or l
    Elw = 2*(Ew.sum(axis=1)*dkw)[lw.size/2:]
    Ekw = 2*(Ew.sum(axis=0)*dlw)[kw.size/2:]
    kxw = kw[kw.size/2:]
    lyw = lw[lw.size/2:]
    Elm = 2*(Em.sum(axis=1)*dkm)[lm.size/2:]
    Ekm = 2*(Em.sum(axis=0)*dlm)[km.size/2:]
    kxm = km[km.size/2:]
    lym = lm[lm.size/2:]

    # restore original variance before mirroring
    Eiso_m = Eiso_m/8.
    Ekm = Ekm/8.
    Elm = Elm/8.    

    # save iso
    fno='outputs/Eiso_uv_hourly'
    np.savez(fno,Ew=Eiso_w,Em=Eiso_m,kw=K_w,km=K_m)

    # Eiso filtered
    aux=np.load('outputs/Eiso.npz')

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

   
    fig = plt.figure(facecolor='w', figsize=(12.,8.5))
    plt.loglog(K_w,Eiso_w,color='m',linewidth=4.,alpha=.5)
    plt.loglog(aux['k'],aux['E'],'--',color='m',linewidth=4.,alpha=.5)
    plt.loglog(ks,Es2,'--', color='k',linewidth=2.,alpha=.5)
    plt.loglog(ks,Es3,'--', color='k',linewidth=2.,alpha=.5)
    plt.text(0.0011686481894527252, 5.4101984795026086/2.,u'k$^{-2}$')
    plt.text(0.0047869726184615827, 5.5118532543417871/2.,u'k$^{-3}$')
    plt.axis((1./(1000),1.,.4e-5,10))
    plt.ylabel('Spectral density  [(m$^2$s$^{-2}$)/(cycles/km)]')
    plt.xlabel('Wavenumber  [cycles/km]')
    plt.savefig('figs/Eiso_uv_2')



