
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import scipy.signal
    import scipy as sp
    import glob, os 
    import seawater.csiro as sw
    import aux_func_3dfields as my
    

    plt.close('all')
    plt.rcParams.update({'font.size': 24, 'legend.handlelength'  : 1.5
        , 'legend.markerscale': 14., 'legend.linewidth': 3.})

    ## mapping
    lonplot = (-70, -50)
    latplot = (-67, -50)

    m = Basemap(llcrnrlon=lonplot[0],llcrnrlat=latplot[0],urcrnrlon=lonplot[1],
            urcrnrlat=latplot[1],
                rsphere=(6378137.00,6356752.3142),
                resolution='i',area_thresh=1000.,projection='lcc',
                lat_1=-55,lon_0=-70)

    iz = 0     # vertical level [m]
    data_path = '/Users/crocha/Data/llc4320/uv/'+str(iz)+'m/'
    grid_path= '/Users/crocha/Data/llc4320/uv/'

    grid = np.load(grid_path+'grid.npz')
    lon = grid['lon']
    lat = grid['lat']
     
    lon,lat=m(lon,lat)

    file = data_path+'2011-11-27.npz'

    data = np.load(file)
    speed = np.sqrt((data['u'][:,:,7]**2 + data['v'][:,:,7]**2))
    speed = np.ma.masked_array(speed,speed>1.26)

    del data

    fig = plt.figure(facecolor='w', figsize=(12.,8.5))


   # plt.contourf(lon,lat,speed,150,cmap='afmhot',vmin=0,vmax=1.26,extend='max')

    plt.pcolor(lon,lat,speed,cmap='afmhot',vmin=0,vmax=1.26)
    cb = plt.colorbar(orientation='horizontal',shrink=.45,pad = 0.08,extend='max')
    cb.set_ticks(np.arange(0,1.6,.4))
    cb.set_label(u'Surface speed [m s$^{-1}$]')

    m.fillcontinents(color='.60',lake_color='none')
    m.drawparallels(np.arange(latplot[0], latplot[1], 4),
        labels=[1, 0, 0, 0], linewidth=.5, 
        fontsize=18., fontweight='medium')
    m.drawmeridians(np.arange(lonplot[0], lonplot[1]+4, 7),
        labels=[0, 0, 0, 1], dashes=[1,1], linewidth=.5, 
        fontsize=18., fontweight='medium')
    m.drawcoastlines()
    plt.text(1278212.26, 1615373.07, '(b)')
    plt.savefig('figs/study_region',dpi=300,bbox_inches='tight')

    plt.close('all')
