import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from bp import bp_main
from common import timer
from numpy.lib.function_base import append
from optim import Optimizer

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--molecule", default="all", help="")
parser.add_argument("--maxiter", type=int, default=100, help="")
parser.add_argument("--plot", action="store_true", default=False, help="")
parser.add_argument("--test", action="store_true", default=False, help="")
parser.add_argument("--mock", action="store_true", default=False, help="")
parser.add_argument("--create-mock-data", action="store_true", default=False, help="")

args = parser.parse_args()

mpl.rcParams['figure.figsize'] = (7,6)

scale = {
    'amp': 232,
    'adp': 50,
    'atp': 20,
    'poly_arg': 20,
}

scale = {
    'mock': 1,
    'amp': 1,
    'adp': 1,
    'atp': 1,
}

exp_limit = {
    'atp': {
        'x': [0, 1],
        'y': [0, 1],
    },
    'adp': {
        'x': [0, 0.001],
        'y': [0.0005, 1],
    },
    'amp': {
        'x': [0, 0.0012],
        'y': [0, 1],
    },
}

exp_limit_plot = {
    'atp': {
        'x': [0, 1],
        'y': [0, 1],
    },
    'adp': {
        'x': [0, 1],
        'y': [0, 1],
    },
    'amp': {
        'x': [0, 1],
        'y': [0, 1],
    },
}

def main():    

    sel = args.molecule

    opt = Optimizer(
        lattice_connectivity=6,
        sel=sel,
        vol_frac_scaling_x=scale[sel],
        # vol_frac_scaling_y=scale['poly_arg'],
        vol_frac_scaling_y=scale[sel],
    )

    opt.load_exp_data(exp_limit_plot[sel])

    # p ... positive ... ARG
    # m ... negative ... ATP/ADP/AMP
    # 0 ... solvent ... water

    J = 10

    bounds = {
        'lp': (0, 10), # rel. to l0
        'lm': (0, 10), # rel. to l0
        'l0': (1, 1),  # reference
        'Jp': (-J, 0),
        'Jm': (-J, 0),
        'Jpm': (0, J),
        'J0': (-1, 1),
        'J0p': (-1, 1),
        'J0m': (-1, 1),
        'scale': (1, 100),
        'rel_scale_y': (0.1, 10),
    }

    opt.optimize_cy(bounds, maxiter=args.maxiter)

def main_mock():    

    sel = args.molecule

    opt = Optimizer(
        lattice_connectivity=4,
        sel='mock',
        vol_frac_scaling_x=1,
        vol_frac_scaling_y=1,
    )

    # opt.load_mock_data('../nino/phase_diag_0.txt')
    opt.load_mock_data('mock_data.txt')

    J1 = 4
    eps = 0.4
    J0 = 0

    bounds = {
        'lp': (1, 1+1e-10), # rel. to l0
        'lm': (1, 1+1e-10), # rel. to l0
        'l0': (1, 1+1e-10),  # reference
        'Jp': (-1-eps, -1+eps),
        'Jm': (-1-eps, -1+eps),
        'Jpm': (3-eps, 3+eps),
        'J0': (-J0, J0+1e-10),
        'J0p': (-J0, J0+1e-10),
        'J0m': (-J0, J0+1e-10),
        'scale': (1, 1+1e-10),
        'rel_scale_y': (1, 1+1e-10),
    }

    opt.optimize_cy(bounds, maxiter=args.maxiter)

@timer
def bp_wrapper(*args):
    err = bp_main(*args)
    if err < 0:
        raise(ValueError("Not converged"))
    return err

def test(
    sel, 
    # lp = 2,
    # lm = 1,
    # l0 = 1,
    # Jp = -0.2,
    # Jm = -20,
    # Jpm = 1.1*2,
    # J0 =  -0.2,
    # J0p = -0.2,
    # J0m = -0.1,
    lp = 1,
    lm = 1,
    l0 = 1,
    Jp = -1.3,
    Jm = -2,
    Jpm = 2.2,
    J0 =  0.1,
    J0p = 0.2,
    J0m = 0.6,
    opt_scale = 1,
    opt_rel_scale_y = 1,
):

    if sel == 'mock':
        lattice_connectivity = 4
    else:
        lattice_connectivity = 6

    opt = Optimizer(
        lattice_connectivity=lattice_connectivity,
        sel=sel,
        vol_frac_scaling_x=scale[sel] * opt_scale,
        # vol_frac_scaling_y=scale['poly_arg'],
        vol_frac_scaling_y=scale[sel] * opt_scale * opt_rel_scale_y,
    )

    if sel == 'mock':
        opt.load_mock_data('../nino/phase_diag_0.txt')
    else:
        opt.load_exp_data(exp_limit_plot[sel], extended=False)

    x = opt.x
    y = opt.y
    sep_exp = opt.sep

    n = x.size

    print(n)

    # x = np.random.rand(n) / 2
    # y = np.random.rand(n) / 2
    hx_xy = np.zeros(n)
    hy_xy = np.zeros(n)
    hx_xpy = np.zeros(n)
    hy_xpy = np.zeros(n)
    hx_xyp = np.zeros(n)
    hy_xyp = np.zeros(n)
    mp_ = np.zeros(n)
    mm_ = np.zeros(n)
    xp_ = np.zeros(n)
    xm_ = np.zeros(n)
    sep = np.ones(n, dtype=np.int32)
    is_on_boundary = np.ascontiguousarray(opt.is_on_boundary)

    err = bp_wrapper(
        lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m, 
        1, 1, 
        x, y,
        hx_xy, hy_xy,
        hx_xpy, hy_xpy,
        hx_xyp, hy_xyp,
        mp_, mm_, xp_, xm_,
        sep, sep_exp,
        is_on_boundary, opt.lattice_connectivity, n, 1, 1,
    )

    print(err)

    i1 = sep != 0
    i2 = sep == 0

    i3 = sep_exp != 0
    i4 = sep_exp == 0

    plt.figure()
    plt.plot(x[i1 & i3], y[i1 & i3], 's', color='b', alpha=0.3)
    plt.plot(x[i2 & i4], y[i2 & i4], 's', color='#999', alpha=0.3)
    plt.plot(x[i1 & i4], y[i1 & i4], 's', color='r', alpha=0.3)
    plt.plot(x[i2 & i3], y[i2 & i3], 's', color='#222', alpha=0.3)
    plt.xlabel(sel)
    plt.ylabel('polyARG')
    # plt.show()

    xp = np.linspace(0.001, 0.75, 128)
    yp = np.linspace(0.001, 0.75, 128)
    X, Y = np.meshgrid(xp, yp)
    X = X.flatten()
    Y = Y.flatten()
    ind = ((X + Y) < 1) & ((X + Y) > 0)
    print(ind.shape)
    x = np.ascontiguousarray(X[ind])
    y = np.ascontiguousarray(Y[ind])

    n = x.size

    print(n)

    # x = np.random.rand(n) / 2
    # y = np.random.rand(n) / 2
    hx_xy = np.zeros(n)
    hy_xy = np.zeros(n)
    hx_xpy = np.zeros(n)
    hy_xpy = np.zeros(n)
    hx_xyp = np.zeros(n)
    hy_xyp = np.zeros(n)
    mp_ = np.zeros(n)
    mm_ = np.zeros(n)
    xp_ = np.zeros(n)
    xm_ = np.zeros(n)
    sep_exp = np.ones(n, dtype=np.int32)
    sep = np.ones(n, dtype=np.int32)
    is_on_boundary = np.zeros(n, dtype=np.int32)

    err = bp_wrapper(
        lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m, 
        1, 1, 
        x, y,
        hx_xy, hy_xy,
        hx_xpy, hy_xpy,
        hx_xyp, hy_xyp,
        mp_, mm_, xp_, xm_,
        sep, sep_exp,
        is_on_boundary, opt.lattice_connectivity, n, 1, 1,
    )

    print(err)

    print(min(sep), max(sep))

    Z = -np.ones_like(X)
    Z[ind] = sep
    X, Y = np.meshgrid(xp, yp)
    Z = Z.reshape(X.shape)

    print(X.shape)

    plt.figure()
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='gray_r', vmax=3)
    plt.show()

def create_mock_data(
    lp = 1,
    lm = 1,
    l0 = 1,
    Jp = -1,
    Jm = -1,
    Jpm = 3,
    J0 =  0,
    J0p = 0,
    J0m = 0,
    lattice_connectivity = 4
):

    opt = Optimizer(
        lattice_connectivity=lattice_connectivity,
        sel='mock',
        vol_frac_scaling_x=1,
        vol_frac_scaling_y=1,
    )

    xp = np.linspace(0.001, 0.99, 128)
    yp = np.linspace(0.001, 0.99, 128)
    X, Y = np.meshgrid(xp, yp)
    X = X.flatten()
    Y = Y.flatten()
    ind = ((X + Y) < 1) & ((X + Y) > 0)
    
    x = np.ascontiguousarray(X[ind])
    y = np.ascontiguousarray(Y[ind])

    n = x.size

    hx_xy = np.zeros(n)
    hy_xy = np.zeros(n)
    hx_xpy = np.zeros(n)
    hy_xpy = np.zeros(n)
    hx_xyp = np.zeros(n)
    hy_xyp = np.zeros(n)
    mp_ = np.zeros(n)
    mm_ = np.zeros(n)
    xp_ = np.zeros(n)
    xm_ = np.zeros(n)
    sep_exp = np.ones(n, dtype=np.int32)
    sep = np.ones(n, dtype=np.int32)
    is_on_boundary = np.zeros(n, dtype=np.int32)

    err = bp_wrapper(
        lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m, 
        1, 1, 
        x, y,
        hx_xy, hy_xy,
        hx_xpy, hy_xpy,
        hx_xyp, hy_xyp,
        mp_, mm_, xp_, xm_,
        sep, sep_exp,
        is_on_boundary, opt.lattice_connectivity, n, 1, 1,
    )

    data = np.concatenate([x[:,np.newaxis], y[:,np.newaxis], sep[:,np.newaxis]], axis=1)
    np.savetxt('mock_data.txt', data)

    Z = -np.ones_like(X)
    Z[ind] = sep
    X, Y = np.meshgrid(xp, yp)
    Z = Z.reshape(X.shape)    

    xx = []
    yy = []
    zz = []
    for i in np.arange(1, 128-1):
        for j in np.arange(1, 128-1):
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
    plt.show()

if __name__ == '__main__':
    if args.test:
        test('adp')

    elif args.create_mock_data:
        create_mock_data()

    elif args.mock:        
        main_mock()

    else:

        if not args.plot:
            main()
        else:
            params = {}
            if args.molecule == "mock":
                pass
                par = []
                par.append([
                    1, 1, 1, -1, -1, 3, 0, 0, 0, 1, 1,
                    # 1,   1,   1, -2.12919, -8.21704,  10, 4.83072, 5.33092, 5.67371,   1,   1
                ])
                params['mock'] = par


            if args.molecule == "atp" or args.molecule == "all":
                pass
                par = []
                par.append([
                    # 0.373026, 1.25592,   1, -4.39179, -0.541745, 7.92896, -0.278993, 0.72404, -0.322766, 1.02934, 1
                    # 6.03091, 7.12607,   1, -1.46297, -9.83835, 2.87021, -0.984717, -0.843357, 0.643944, 99.9025, 4.67344
                    # 7.50997, 5.29602,   1, -0.887812, -7.12186, 2.08772, 0.712454, -0.0350979, 0.90165, 97.4518, 5.20762
                ])
                params['atp'] = par

            if args.molecule == "adp" or args.molecule == "all":
                pass
                par = []
                par.append([
                    # 3.05483, 6.91701,   1, -6.2345, -5.90474, 3.71985, -0.914625, -0.0814894, -0.397358, 34.6382, 1
                    # 7.59671, 4.39042,   1, -9.92094, -3.51605, 1.20798, 0.03642, -0.527211, -0.655306, 90.9876, 1
                    # 2.34725, 4.22956,   1, -0.0919252, -6.98841, 0.665866, 0.216858, -0.959242, 0.53815, 83.7615, 1
                    # 3.71504, 3.97799,   1, -4.27913, -3.12154, 4.88684, -0.143496, 0.20706, 0.640943, 64.4663, 0.106799
                    # 1.93002, 4.17366,   1, -3.36183, -8.36226, 3.39099, 0.787194, 0.516193, -0.0187999, 86.1679, 0.10456
                    4.34972, 6.10435,   1, -6.55052, -8.62514, 4.75724, -0.820719, -0.0102101, 0.27347, 84.3039, 0.107124
                ])
                
                params['adp'] = par

            if args.molecule == "amp" or args.molecule == "all":
                pass
                par = []
                par.append([
                    # 3.84697, 0.437053,   1, -7.13027, -3.27763, 8.36572, -0.704957, 0.784959, -0.455163, 1.05849, 1
                    # 1.91885, 9.47034,   1, -8.98172, -4.82571, 9.8982, -0.249436, -0.335888, 0.303387, 99.4064, 0.108451
                    7.11278, 6.24691,   1, -3.50006, -6.11359, 9.56895, 0.84808, 0.165324, 0.561215, 99.4149, 0.108503
                ])
                params['amp'] = par

            for mol in params:
                par = params[mol]
                for p in par:
                    test(mol, *p)