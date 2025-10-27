import numpy as np

def scale(X, scl=None, axis=None):
    
    if scl is None: # if no scaling factor provided, scale by max abs val
        scl = np.max(np.abs(X),axis=axis)
        
        if scl == 0.:
            scl = 1.0

    if axis==0 or axis is None:
        X = X/scl
    elif axis==1:
        X = X/np.vstack(scl)
    else:
        print('Invalid axis')
    return X,scl

def shift(X,shf=None,axis=None):
    if shf is None: # if no shift provided, shift by mean
        shf = np.mean(X,axis=axis)
    if axis == 0 or axis is None:
        X = X-shf
    elif axis == 1:
        X = X - np.vstack(shf)
    else:
        print('Invalid axis')
    return X,shf

def shiftthenscale(X,scl=None,shf=None,axis=None):
    X,shf = shift(X,shf,axis=axis)
    X,scl = scale(X,scl=scl,axis=axis)
    return X,shf,scl

def scalethenshift(X,scl=None,shf=None,axis=None):
    X,scl = scale(X,scl,axis=axis)
    X,shf = shift(X,shf,axis=axis)
    return X,scl,shf

def minus1to1(X,axis=None,bounds=None,rev = False):
    if rev:
        if len(bounds.shape)==2:
            minX = bounds[:,0]
            maxX = bounds[:,1]
        elif len(bounds.shape)==1:
            minX = bounds[0]
            maxX = bounds[1]
        X,shf,scl = shiftthenscale(X,shf=-(maxX+minX)/(maxX-minX),scl=2./(maxX-minX),axis=axis)
    else: 
        if bounds is None:
            minX = np.min(X,axis)
            maxX = np.max(X,axis)
        elif len(bounds.shape)==2:
            minX = bounds[:,0]
            maxX = bounds[:,1]
        elif len(bounds.shape)==1:
            minX = bounds[0]
            maxX = bounds[1]
        X,scl,shf = scalethenshift(X,scl=0.5*(maxX-minX),shf=(maxX+minX)/(maxX-minX),axis=axis)
    return X,scl,shf

def zeroto1(X,axis=None,bounds=None):
    if bounds is None:
        minX = np.min(X,axis)
        maxX = np.max(X,axis)
    elif len(bounds.shape)==2:
        minX = bounds[:,0]
        maxX = bounds[:,1]
    elif len(bounds.shape)==1:
        minX = bounds[0]
        maxX = bounds[1]
    X,scl,shf = scalethenshift(X,scl=maxX-minX,shf=minX/(maxX-minX),axis=axis)
    return X,scl,shf

def znorm(X,axis=None,stats=None):
    if stats is None:
        meanX = np.mean(X,axis)
        stdX  = np.std(X,axis)
    elif len(stats.shape)==2:
        meanX = stats[:,0]
        stdX  = stats[:,1]
    elif len(stats.shape)==1:
        meanX = stats[0]
        stdX = stats[1]
    X,shf,scl = shiftthenscale(X,shf = meanX,scl=stdX,axis=axis)
    return X,shf,scl

# test above
if __name__ == '__main__':
    A = np.random.rand(10,4)*10-5
    print(A)
    A2,scl,shf = minus1to1(A)
    print(A2)
