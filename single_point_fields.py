
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy as sp
    import glob, os 
   
    # mapping
    lonplot = (-70, -50)
    latplot = (-65.5, -52)

    iz = 1000  # vertical level [m]
    grid_path= '/Users/crocha/Data/llc4320/uv/'

    grid = np.load(grid_path+'grid.npz')

    # time
    aux = np.load('subsets/eta2video.npz')
    time =  aux['time']
    del aux

    # select single points 
    iN,jN = 520,143  # (-56.6S,-63W)
    iS,jS = 120,143  # (-60.8S,-63W)

    data_path = '/Users/crocha/Data/llc4320/w/'+str(iz)+'m/*'
    files = sorted(glob.glob(data_path), key=os.path.getmtime) 
    k = 0
    for file in sorted(files):
        data = np.load(file)
        if k == 0:
            wN, wS = data['w'][iN,jN,:], data['w'][iS,jS,:]
        else:    
            wN, wS = np.append(wN,data['w'][iN,jN,:]),np.append(wS,data['w'][iS,jS,:])  
        k = k + 1
        del data

    data_path = '/Users/crocha/Data/llc4320/uv/'+str(iz)+'m/*'
    files = sorted(glob.glob(data_path), key=os.path.getmtime) 
    k = 0
    for file in sorted(files):
        data = np.load(file)
        if k == 0:
            uN, uS = data['u'][iN,jN,:], data['u'][iS,jS,:]
            vN, vS = data['v'][iN,jN,:], data['v'][iS,jS,:]
        else:    
            uN, uS = np.append(uN,data['u'][iN,jN,:]),np.append(uS,data['u'][iS,jS,:])  
            vN, vS = np.append(vN,data['v'][iN,jN,:]),np.append(vS,data['v'][iS,jS,:])  
        k = k + 1
        del data

time = time[0:uN.size]
np.savez('single_pt_uvw.npz', wN=wN,uN=uN,vN=vN,wS=wS,uS=uS,vS=vS,lonN=grid['lon'][iN,jN],
        latN=grid['lat'][iN,jN],lonS=grid['lon'][iS,jS],latS=grid['lat'][iS,jS],time=time)

