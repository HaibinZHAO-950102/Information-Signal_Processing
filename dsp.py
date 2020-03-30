import numpy as np

def sampling(Y, fa):
    t = Y[1,:].flatten()
    y = Y[0,:].flatten()
    t_max = np.max(t)
    y_sampling = []
    t_sampling = []
    for i in np.arange(0, t_max, 1 / fa):
        t1 = t - i
        t1 = np.abs(t1)
        index = np.where(t1 == np.min(t1))
        y_sampling.append(y[index])
        t_sampling.append(t[index])
    Y_sampling = np.array([[y_sampling],[t_sampling]])
    return Y_sampling

def dft(Y_sampling):
    t = Y_sampling[1,:].flatten()
    y = Y_sampling[0,:].flatten()
    N = len(t)
    ta = t[1] - t[0]
    fa = 1 / ta
    f = np.empty([N])
    Y = np.empty([N],dtype='D')
    for k in range(N):
        f[k] = 1 / ta / N * k
        Y[k] = 0
        for n in range (N):
            Y[k] = Y[k] + y[n] * np.exp(complex(0,-1)*2*np.pi*k*n/N)
    Y_dft = np.array([[Y],[f]])
    return Y_dft

def idft(Y_dft):
    f = Y_dft[1,:].flatten()
    a = Y_dft[0,:].flatten()
    N = len(f)
    t = np.empty([N])
    y = np.empty([N],dtype='D')
    fa = np.max(f)
    ta = 1 / fa
    for n in range(N):
        t[n] = ta * n
        y[n] = 0
        for k in range(N):
            y[n] = y[n] + a[k] * np.exp(complex(0,1)*2*np.pi*k*n/N)
    y = y / N
    Y_idft = np.array([[y],[t]])
    return Y_idft
