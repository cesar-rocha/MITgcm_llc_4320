# try and test PS decomposition a al Smith JPO 2008

def ps(u,v,dx,dy):

    ix,jx = u.shape
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
      
    U = np.fft.fft2(u)
    V = np.fft.fft2(v)

    P = U*np.cos(THETA) + V*np.sin(THETA)
    S = -U*np.sin(THETA) + V*np.cos(THETA)

    # back to physical space
    up = np.real(np.fft.ifft2(P*np.cos(THETA)))
    vp = np.real(np.fft.ifft2(P*np.sin(THETA)))

    us = np.real(np.fft.ifft2(-S*np.sin(THETA)))
    vs = np.real(np.fft.ifft2(S*np.cos(THETA)))

    return up,vp,us,vs



if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    import numpy as np


    plt.close('all')
    plt.rcParams.update({'font.size': 24, 'legend.handlelength'  : 1.5
        , 'legend.markerscale': 14., 'legend.linewidth': 3.})


    # create synthetic flows
    dx = 0.02
    dy = 0.02
    x = np.arange(-1.,1.+dx,dx)
    y = np.arange(-1.,1.+dy,dy)
    x,y = np.meshgrid(x,y)
    r = np.sqrt((x**2)+(y**2))
    theta = np.arctan2(y,x)

    # irrotational flow
    ui = np.cos(theta)
    vi = np.sin(theta)

    # non-div flow
    un = -np.sin(theta)*0
    vn = np.cos(theta)*0
    
    # total flow
    u = un + ui
    v = vn + vi
    
    # PS decomposition    
    u = u - u.mean()
    v = v - v.mean()

    up,vp,us,vs = ps(u,v,dx,dy)

    uu=up+us 
    vv=vp+vs

    # plotting
    dec = 3
    sc = 40.
    plt.figure()
    plt.quiver(x[0::dec,0::dec],y[0::dec,0::dec],u[0::dec,0::dec],v[0::dec,0::dec],color='b',
            scale=sc)
    plt.quiver(x[0::dec,0::dec],y[0::dec,0::dec],uu[0::dec,0::dec],vv[0::dec,0::dec],color='m',
            scale=sc)
    plt.title('Total')

    plt.figure()
    plt.quiver(x[0::dec,0::dec],y[0::dec,0::dec],ui[0::dec,0::dec],vi[0::dec,0::dec],color='b',
            scale=sc)
    plt.quiver(x[0::dec,0::dec],y[0::dec,0::dec],up[0::dec,0::dec],vs[0::dec,0::dec],color='m',
            scale=sc)
    plt.title('Irrotational')

    plt.figure()
    plt.quiver(x[0::dec,0::dec],y[0::dec,0::dec],un[0::dec,0::dec],vn[0::dec,0::dec],color='b',
            scale=sc)
    plt.quiver(x[0::dec,0::dec],y[0::dec,0::dec],us[0::dec,0::dec],vs[0::dec,0::dec],color='m',
            scale=sc)
    plt.title('Non-divergent')



