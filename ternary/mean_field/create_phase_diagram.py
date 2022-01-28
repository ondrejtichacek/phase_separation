import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from bp import bp_main
from optim import Optimizer

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-N", type=int, default=50, help="")

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
    N = 100,
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
    ZZ = -np.ones_like(X)
    ind = ((X + opt.dx + Y) < 1) & ((X + Y + opt.dy) < 1) & (X > 0)  & (Y > 0)
    
    x = np.ascontiguousarray(X[ind])
    y = np.ascontiguousarray(Y[ind])

    err, sep = opt.bp_wrapper(x, y, lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m)
    print(err)

    data = np.concatenate([
        x[:,np.newaxis], y[:,np.newaxis], 
        sep[:,np.newaxis], (2 * sep[:,np.newaxis]) - 1], axis=1)
    np.savetxt(file_name, data)

    Z = -np.ones_like(ZZ)
    Z[ind] = sep
    X, Y = np.meshgrid(xp, yp)
    Z = Z.reshape(X.shape)    

    h1 = opt.plot_phase_diagram(X, Y, Z)

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

    xx = np.asarray(xx)
    yy = np.asarray(yy)
    zz = np.asarray(zz)

    plt.figure()
    plt.plot(xx[zz == 0], yy[zz == 0], 's', color='b', alpha=0.3)
    plt.plot(xx[zz == 1], yy[zz == 1], 's', color='r', alpha=0.3)

    # CW ----

    sep_cw = opt.CW_set(x, y, Jp, Jm, Jpm)
    print(sep_cw.shape)
    data = np.concatenate([
        x[:,np.newaxis], y[:,np.newaxis], 
        sep[:,np.newaxis], (2 * sep[:,np.newaxis]) - 1], axis=1)
    # np.savetxt(file_name, data)

    Z = -np.ones_like(ZZ)
    Z[ind] = sep_cw
    X, Y = np.meshgrid(xp, yp)
    Z = Z.reshape(X.shape)

    h2 = opt.plot_phase_diagram(X, Y, Z)

    # ---------

    return h1, h2
    

if __name__ == '__main__':

    par = []
    file_name = []

    file_name.append('mock_1_BP.txt')
    par.append([1, 1, 1, -1, -1, 3, 0, 0, 0, 1, 1])

    file_name.append('mock_2_BP.txt')
    par.append([1, 1, 1, 1, 1, -3, 0, 0, 0, 1, 1])

    file_name.append('mock_3_BP.txt')
    par.append([1, 1, 1, 0, 2, 0.5, 0, 0, 0, 1, 1])

    # file_name.append('atp_fit_best.txt')
    # par.append([9.95244, 6.88167, 1.0, -2.94081, -1.00765, 0.932429, -0.380839, -0.27919, -0.889506, 1, 1, 6])
    # file_name.append('adp_fit_best.txt')
    # par.append([0.472709, 4.70077, 1.0, -6.50968, -3.09633, 1.89146, -0.406973, -0.654119, 0.503798, 1, 1, 6])
    # file_name.append('amp_fit_best.txt')
    # par.append([5.08029, 0.747859, 1.0, -7.3834, -7.93062, 2.34327, -0.584678, -0.0971994, -0.148038, 1, 1, 6])

    for i, (f, p) in enumerate(zip(file_name, par)):
        h1, h2 = create_mock_data(f, *p, N=args.N)
        
        h1.savefig(f"phase_diagram_analytical_model_{i}_BP.pdf", format="pdf", bbox_inches="tight")
        h2.savefig(f"phase_diagram_analytical_model_{i}_CW.pdf", format="pdf", bbox_inches="tight")

    plt.show()
