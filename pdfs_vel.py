
def gauss(x,mu,sig):
    ''' computes normal distribution given mean (mu) and std (sig) '''
    g = (1./(np.sqrt(2*np.pi)*sig))*np.exp(-.5*(((x-mu)/sig)**2))
    return g

def moments(x,pdfx,n):
    ''' computes the nth moments of x given its pdf '''
    dx = np.abs(x[1]-x[0])
    return ((x**n)*pdfx*dx).sum()


if __name__=='__main__':

    import matplotlib.pyplot as plt
    import numpy as np

    plt.close('all')

    plt.rcParams.update({'font.size': 24, 'legend.handlelength'  : 1.5
        , 'legend.markerscale': 14., 'legend.linewidth': 3.})

    color2 = '#6495ed'
    color1 = '#ff6347'
    color3 = '#8470ff'
    color4 = '#3cb371'
    lw1 = 2

    fni = 'outputs/0m_vel_stats.npz'
    vstats = np.load(fni)    

    

