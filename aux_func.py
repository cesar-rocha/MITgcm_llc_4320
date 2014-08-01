import numpy as np

# set the linewidth of each legend object
def leg_width(lg,fs):
    for legobj in lg.legendHandles:
        legobj.set_linewidth(fs)

def rmean(A):
    """removes time-mean of 3d fields; axis=2 is time"""
    ix,jx,kx = A.shape    
    Am = np.repeat(A.mean(axis=2),kx)
    Am = Am.reshape(ix,jx,kx)
    return A-Am


# autocorrelation
def auto_corr(x):
    a = np.correlate(x,x,mode='full')
    a = a[x.size-1:]
    a = a/a[0]
    return a

def fit_gauss(x,y):
    y = np.matrix(np.log(y)).T
    A1 =  np .matrix(np.ones((x.size,1)))
    A2 = np.matrix(-(x**2)).T
    A = A2

    xmax = 650
    we = np.float(xmax) - x
    we =  (we/(xmax))**2
    We = np.matrix(np.diag(we))

    Gg = ((A.T*We*A).I)*A.T
    c = Gg*y

    Lfit = np.sqrt(1/c[-1])

    yfit = np.exp( A*c )

    return Lfit

def spec_error(E,sn,ci):

    # computes confidence interval for spectral 
    #   estimate E
    # sn is the number of spectral realizations (dof/2)
    # ci = .95 for 95 % confidence interval 
    # returns lower (El) and upper (Eu) bounds on E
    #   as well as pdf and cdf used to estimate errors 

    ## params
    dbin = .001
    yN = np.arange(0,5.+dbin,dbin)
    dof = 2*sn # DOF = 2 x # of spectral estimates

    ## PDF for E/E0, where E (E0) is the estimate (true) 
    ##  process spectrum (basically a chi^2 distribution)
    C = dof / ( (2**sn) * np.math.gamma(sn) )  # constant
    pdf_yN = C * ( (dof*yN)**(sn-1) ) * np.exp( -(sn*yN) ) # chi^2(E/E0)

    ## CDF
    cdf_yN = np.cumsum(pdf_yN*dbin)  # trapezoidal-like integration

    ## compute confidence limits

    # lower
    el = ci
    fl = np.where( np.abs(cdf_yN - el) ==  np.abs(cdf_yN - el).min())
    El = E/yN[fl]

    # upper 
    eu = 1 - ci
    fu = np.where( np.abs(cdf_yN - eu) ==  np.abs(cdf_yN - eu).min())
    Eu = E/yN[fu]

    return El, Eu, cdf_yN, pdf_yN
