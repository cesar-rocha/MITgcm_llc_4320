
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import scipy as sp
    import glob, os 
    from netCDF4 import Dataset
   
    # mapping
    lonplot = (-70, -50)
    latplot = (-65.5, -52)

    m = Basemap(llcrnrlon=lonplot[0],llcrnrlat=latplot[0],urcrnrlon=lonplot[1],
            urcrnrlat=latplot[1],
                rsphere=(6378137.00,6356752.3142),
                resolution='i',area_thresh=1000.,projection='lcc',
                lat_1=-55,lon_0=-70)

    plt.rcParams.update({'font.size': 20, 'legend.handlelength'  : 1.5
        , 'legend.markerscale': 14., 'legend.linewidth': 3.})



    plt.close('all')
    plt.rcParams.update({'font.size': 24, 'legend.handlelength'  : 1.5
        , 'legend.markerscale': 14., 'legend.linewidth': 3.})

    plt.close('all')

    etopopath='/Users/crocha/Data/topo/etopo1.cdf'
    etopo = Dataset(etopopath)
    lont = np.array(etopo.variables['ETOPO01_X'][:])
    latt = np.array(etopo.variables['ETOPO01_Y'][:])
    topo = np.array(etopo.variables['ROSE'][(latt > latplot[0]) & (latt < latplot[1]),(lont > lonplot[0]) & (lont < lonplot[1]+7)])

    lont = lont[(lont > lonplot[0]) & (lont < lonplot[1]+7)]
    latt = latt[(latt > latplot[0]) & (latt < latplot[1])]

    topo= np.ma.masked_array(topo, topo>0)
    lont,latt = np.meshgrid(lont,latt)
    xgt,ygt=m(lont,latt)

    iz = 1000   # vertical level [m]
    data_path = '/Users/crocha/Data/llc4320/w/'+str(iz)+'m/*'
    grid_path= '/Users/crocha/Data/llc4320/uv/'

    grid = np.load(grid_path+'grid.npz')
    xgm,ygm = m(grid['lon'],grid['lat'])

    # time
    aux = np.load('subsets/eta2video.npz')
    time =  aux['time']
    del aux

    files = sorted(glob.glob(data_path), key=os.path.getmtime) 
    dect=5

    sc=8.
    it = 50
    kt = 50*24

    for file in sorted(files[it:]):
        
        data = np.load(file)
        ix,jx,kx = data['w'].shape
        waux = data['w']

        for i in range(kx):
            fig=plt.figure(facecolor='w', figsize=(12.,12.))
            m.pcolor(xgm,ygm,waux[:,:,i]*100, vmin=-10.,vmax=10.,cmap='PuOr')
            cb = plt.colorbar(extend='both',shrink=.7)
            cb.set_label(u'Vertical velocity [cm s$^{-1}$]')
            cs = m.contour(xgt[1:-1:dect,1:-1:dect],ygt[1:-1:dect,1:-1:dect],-topo[1:-1:dect,1:-1:dect],
                    np.array([200,1000,2000]),colors='k',alpha=.4)
            plt.clabel(cs,inline=1,fontsize=10,fmt='%i')
            m.fillcontinents(color='.60',lake_color='none')
            m.drawparallels(np.arange(latplot[0]-5, latplot[1]+5, 2),
                labels=[1, 0, 0, 0], linewidth=.5, 
                fontsize=18., fontweight='medium')
            m.drawmeridians(np.arange(lonplot[0]-5, lonplot[1]+5, 5),
                labels=[0, 0, 0, 1], dashes=[1,1], linewidth=.5, 
                fontsize=18., fontweight='medium')
            m.drawcoastlines()
            plt.title(str(time[kt])[0:13]+' h')
            plt.savefig('figs2movie_w/'+str(time[kt])[0:10]+str(time[kt])[11:13],dpi=75,bbox_inches='tight')
            plt.close('all')
            kt = kt+1

        del data, waux


