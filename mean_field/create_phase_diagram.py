import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from bp import bp_main
from numpy.lib.function_base import append
from optim import Optimizer

import argparse
parser = argparse.ArgumentParser()

# parser.add_argument("--molecule", default="all", help="")

args = parser.parse_args()

mpl.rcParams['figure.figsize'] = (7,6)


def create_mock_data(
    file_name,
    lp = 1,
    lm = 1,
    l0 = 1,
    Jp = -1,
    Jm = -1,
    Jpm = 3,
    J0 =  0,
    J0p = 0,
    J0m = 0,
    sx = 1, 
    sy = 1,
    lattice_connectivity = 4,
    N = 50,
):

    opt = Optimizer(
        lattice_connectivity=lattice_connectivity,
        sel='mock',
        vol_frac_scaling_x=sx,
        vol_frac_scaling_y=sy,
    )

    xp = np.linspace(0.01, 0.99, N)
    yp = np.linspace(0.01, 0.99, N)
    X, Y = np.meshgrid(xp, yp)
    X = X.flatten()
    Y = Y.flatten()
    ind = ((X + Y) < 1) & ((X + Y) > 0)
    
    x = np.ascontiguousarray(X[ind])
    y = np.ascontiguousarray(Y[ind])

    err, sep = opt.bp_wrapper(x, y, lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m)

    data = np.concatenate([
        x[:,np.newaxis], y[:,np.newaxis], 
        sep[:,np.newaxis], (2 * sep[:,np.newaxis]) - 1], axis=1)
    np.savetxt(file_name, data)

    Z = -np.ones_like(X)
    Z[ind] = sep
    X, Y = np.meshgrid(xp, yp)
    Z = Z.reshape(X.shape)    

    xx = []
    yy = []
    zz = []
    for i in np.arange(1, N-1):
        for j in np.arange(1, N-1):
            neighborhood = np.asarray([Z[i-1,j], Z[i+1,j], Z[i,j-1], Z[i,j+1]])
            notZ = abs(Z[i,j] - 1)
            if np.any(notZ == neighborhood):
                xx.append(X[i,j])
                yy.append(Y[i,j])
                zz.append(Z[i,j])

    plt.figure()
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='gray_r', vmax=3)

    xx = np.asarray(xx)
    yy = np.asarray(yy)
    zz = np.asarray(zz)

    plt.figure()
    plt.plot(xx[zz == 0], yy[zz == 0], 's', color='b', alpha=0.3)
    plt.plot(xx[zz == 1], yy[zz == 1], 's', color='r', alpha=0.3)

if __name__ == '__main__':

    par = []
    file_name = []

    file_name.append('mock_1_BP.txt')
    par.append([1, 1, 1, -1, -1, 3, 0, 0, 0, 1, 1])

    file_name.append('mock_2_BP.txt')
    par.append([1, 1, 1, 1, 1, -3, 0, 0, 0, 1, 1])

    file_name.append('mock_3_BP.txt')
    par.append([1, 1, 1, 0, 2, 0.5, 0, 0, 0, 1, 1])

    for (f, p) in zip(file_name, par):
        create_mock_data(f, *p)


plt.show()