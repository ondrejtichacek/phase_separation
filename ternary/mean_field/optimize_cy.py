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
parser.add_argument("--nrepeat", type=int, default=1, help="")
parser.add_argument("--plot", action="store_true", default=False, help="")
parser.add_argument("--test", action="store_true", default=False, help="")
parser.add_argument("--mock", action="store_true", default=False, help="")
parser.add_argument("--mock-data-file", type=str, default='', help="")

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

    print(f"We will perform {args.nrepeat} optimizations")

    for i in np.arange(args.nrepeat):
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
            'scale_x': (100, 500),
            'scale_y': (100, 500),
            # 'rel_scale_y': (0.1, 10),
        }

        
        opt.optimize_cy(bounds, maxiter=args.maxiter)

def main_mock():
    
    print(f"We will perform {args.nrepeat} optimizations")

    for i in np.arange(args.nrepeat):

        sel = args.molecule

        opt = Optimizer(
            lattice_connectivity=4,
            sel=sel,
            vol_frac_scaling_x=1,
            vol_frac_scaling_y=1,
            use_cont_err=True,
        )

        print(args.mock_data_file)

        opt.load_mock_data(args.mock_data_file)

        J1 = 4
        eps = 0.4
        J0 = 0

        bounds = {
            'lp': (1, 1), # rel. to l0
            'lm': (1, 1), # rel. to l0
            'l0': (1, 1),  # reference
            'Jp': (-10, 10),
            'Jm': (-10, 10),
            'Jpm': (-10, 10),
            'J0': (0, 0),
            'J0p': (0, 0),
            'J0m': (0, 0),
            'scale_x': (1, 1),
            'scale_y': (1, 1),
        }

        opt.optimize_cy(bounds, maxiter=args.maxiter)

def test(
    sel, 
    lp = 1,
    lm = 1,
    l0 = 1,
    Jp = -1.3,
    Jm = -2,
    Jpm = 2.2,
    J0 =  0.1,
    J0p = 0.2,
    J0m = 0.6,
    opt_scale_x = 1,
    opt_scale_y = 1,
):

    if args.mock_data_file == '':
        lattice_connectivity = 6

        opt = Optimizer(
            lattice_connectivity=lattice_connectivity,
            sel=sel,
            vol_frac_scaling_x=scale[sel] * opt_scale_x,
            vol_frac_scaling_y=scale[sel] * opt_scale_y,
        )

    else:
        lattice_connectivity = 4

        opt = Optimizer(
            lattice_connectivity=lattice_connectivity,
            sel=sel,
            vol_frac_scaling_x=opt_scale_x,
            vol_frac_scaling_y=opt_scale_y,
        )

    if args.mock_data_file == '':
        opt.load_exp_data(exp_limit_plot[sel], extended=False)
    else:
        opt.load_mock_data(args.mock_data_file)
        
    x = opt.x
    y = opt.y
    sep_exp = opt.sep
    sep_exp_cont = opt.sep_cont

    err, sep = opt.bp_wrapper(x, y, lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m, 
        sep_exp, sep_exp_cont, opt.is_on_boundary)

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

    err, sep = opt.bp_wrapper(x, y, lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m)

    print(err)

    print(min(sep), max(sep))

    Z = -np.ones_like(X)
    Z[ind] = sep
    X, Y = np.meshgrid(xp, yp)
    Z = Z.reshape(X.shape)

    print(X.shape)

    plt.figure()
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='gray_r', vmax=3)
    # plt.show()

if __name__ == '__main__':
    if args.test:
        test('adp')

    elif args.mock:        
        main_mock()

    else:

        if not args.plot:
            main()
        else:
            params = {}
            if args.molecule == "mock_0":
                pass
                par = []
                par.append([1, 1, 1, -1, -1, 3, 0, 0, 0, 1, 1])
                params['mock_0'] = par

            if args.molecule == "mock_1":
                pass
                par = []
                par.append([1, 1, 1, -1, -1, 3, 0, 0, 0, 1, 1])
                # par.append([1,   1,   1, -2.37565, -9.79737, 3.42365,   0,   0,   0,   1,   1])
                params['mock_1'] = par

            if args.molecule == "mock_2":
                pass
                par = []
                par.append([1, 1, 1, 1, 1, -3, 0, 0, 0, 1, 1])
                # par.append([1,   1,   1, -5.80005, -0.535348, -8.05675,   0,   0,   0,   1,   1])
                # par.append([1,   1,   1, -7.12878, -5.35888, -8.37144,   0,   0,   0,   1,   1])
                par.append([1,   1,   1, -3.93954, 2.14487, -8.8923,   0,   0,   0,   1,   1])
                params['mock_2'] = par

            if args.molecule == "mock_3":
                pass
                par = []
                par.append([1, 1, 1, 0, 2, 0.5, 0, 0, 0, 1, 1])
                par.append([1, 1, 1, -7.5, 2, 2.2, 0, 0, 0, 1, 1]) # this has almosth the same phase-diagram
                # par.append([1,   1,   1, -6.63637, 2.18265, 1.22012,   0,   0,   0,   1,   1])
                # par.append([1,   1,   1, -7.36182, 2.18728, 1.16623,   0,   0,   0,   1,   1])
                params['mock_3'] = par

            if args.molecule == "atp" or args.molecule == "all":
                pass
                par = []
                # lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m, 
                # par.append([0.373026, 1.25592,   1, -4.39179, -0.541745, 7.92896, -0.278993, 0.72404, -0.322766, 1.02934, 1])
                # par.append([6.03091, 7.12607,   1, -1.46297, -9.83835, 2.87021, -0.984717, -0.843357, 0.643944, 99.9025, 4.67344])
                # par.append([7.50997, 5.29602,   1, -0.887812, -7.12186, 2.08772, 0.712454, -0.0350979, 0.90165, 97.4518, 5.20762])
                # par.append([7.2854, 2.45756,   1, -0.108328, -8.76307, 2.95043, -0.735305, -0.304302, 0.500513, 99.3928, 5.16023])
                par.append([7.4339, 2.54478,   1, -4.11252, -1.50244, 1.74199, 0.149457, 0.568983, -0.657058, 133.886, 377.618])
                par.append([9.95244, 6.88167, 1.0, -2.94081, -1.00765, 0.932429, -0.380839, -0.27919, -0.889506, 152.964, 359.972])
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
                    # 4.34972, 6.10435,   1, -6.55052, -8.62514, 4.75724, -0.820719, -0.0102101, 0.27347, 84.3039, 0.107124
                    3.47653, 8.99605,   1, -5.41778, -6.8823, 1.42014, 0.20018, -0.913092, 0.183591, 189.126, 66.6052
                ])
                par.append([0.472709, 4.70077, 1.0, -6.50968, -3.09633, 1.89146, -0.406973, -0.654119, 0.503798, 453.805, 225.624])
                
                params['adp'] = par

            if args.molecule == "amp" or args.molecule == "all":
                pass
                par = []
                par.append([
                    # 3.84697, 0.437053,   1, -7.13027, -3.27763, 8.36572, -0.704957, 0.784959, -0.455163, 1.05849, 1
                    # 1.91885, 9.47034,   1, -8.98172, -4.82571, 9.8982, -0.249436, -0.335888, 0.303387, 99.4064, 0.108451
                    # 7.11278, 6.24691,   1, -3.50006, -6.11359, 9.56895, 0.84808, 0.165324, 0.561215, 99.4149, 0.108503
                    # 2.02363, 1.51003,   1, -2.93271, -9.74068, 8.9148, 0.952906, -0.777616, -0.0830076, 98.6619, 0.108084
                    9.80007, 7.07477,   1, -8.09432, -6.59726, 2.45702, 0.504066, -0.985431, 0.57522, 199.339, 50.6425
                ])
                par.append([5.08029, 0.747859, 1.0, -7.3834, -7.93062, 2.34327, -0.584678, -0.0971994, -0.148038, 462.483, 374.262])
                params['amp'] = par

            for mol in params:
                par = params[mol]
                for p in par:
                    test(mol, *p)

plt.show()