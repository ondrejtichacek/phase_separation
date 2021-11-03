import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from bp import bp_main
from common import timer
from optim import Optimizer

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("molecule", help="")
parser.add_argument("--maxiter", type=int, default=100, help="")
parser.add_argument("--plot", action="store_true", default=False, help="")

args = parser.parse_args()

mpl.rcParams['figure.figsize'] = (7,6)

scale = {
    'amp': 232,
    'adp': 50,
    'atp': 20,
    # 'poly_arg': 20,
}

def main():    

    sel = args.molecule

    opt = Optimizer(
        sel=sel,
        vol_frac_scaling_x=scale[sel],
        # vol_frac_scaling_y=scale['poly_arg'],
        vol_frac_scaling_y=scale[sel],
    )

    opt.load_exp_data()

    # p ... positive ... ARG
    # m ... negative ... ATP/ADP/AMP
    # 0 ... solvent ... water

    J = 20

    bounds = {
        'lp': (0, 10), # rel. to l0
        'lm': (0, 10), # rel. to l0
        'l0': (1, 1),  # reference
        'Jp': (-J, 0),
        'Jm': (-J, 0),
        'Jpm': (0, J),
        'J0': (0, J),
        'J0p': (0, J),
        'J0m': (0, J),
    }

    opt.optimize_cy(bounds, maxiter=args.maxiter)


def test(
    sel, 
    lp = 1,
    lm = 1,
    l0 = 1,
    Jp = -1,
    Jm = -1,
    Jpm = 3,
    J0 = 0.1,
    J0p = 0.5,
    J0m = 0.2,
):

    opt = Optimizer(
        sel=sel,
        vol_frac_scaling_x=scale[sel],
        # vol_frac_scaling_y=scale['poly_arg'],
        vol_frac_scaling_y=scale[sel],
    )

    opt.load_exp_data()

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

    @timer
    def bp_wrapper(*args):
        return bp_main(*args)

    err = bp_wrapper(
        lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m, 
        x, y,
        hx_xy, hy_xy,
        hx_xpy, hy_xpy,
        hx_xyp, hy_xyp,
        mp_, mm_, xp_, xm_,
        sep, sep_exp,
        is_on_boundary, n
    )

    print(err)

    # i1 = sep != 0
    # i2 = sep == 0

    # plt.figure()
    # plt.plot(x[i1], y[i1], 's', color='r', alpha=0.3)
    # plt.plot(x[i2], y[i2], 's', color='#999', alpha=0.3)
    # plt.xlabel(sel)
    # plt.ylabel('polyARG')
    # #plt.show()

    # i1 = sep_exp != 0
    # i2 = sep_exp == 0

    # #plt.figure()
    # plt.plot(x[i1], y[i1], 's', color='b', alpha=0.3)
    # plt.plot(x[i2], y[i2], 's', color='#999', alpha=0.3)
    # plt.xlabel(sel)
    # plt.ylabel('polyARG')
    # plt.show()

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
    plt.show()

    xp = np.linspace(0.001, 1, 200)
    yp = np.linspace(0.001, 1, 200)
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

    @timer
    def bp_wrapper(*args):
        err = bp_main(*args)
        if err < 0:
            raise(ValueError("Not converged"))
        return err

    err = bp_wrapper(
        lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m, 
        x, y,
        hx_xy, hy_xy,
        hx_xpy, hy_xpy,
        hx_xyp, hy_xyp,
        mp_, mm_, xp_, xm_,
        sep, sep_exp,
        is_on_boundary, n
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
    

if __name__ == '__main__':
    if not args.plot:
        main()
    else:

        if args.molecule == "atp":
            pass
            par = []
            # test(
            #     'atp',
            #     0.60644092,   2.02152218,   1.        , -28.04578561,
            #    -25.45601227,  20.81318829,   0.5831698 ,   8.37261113,
            #      9.36110971
            # )
            # test(
            #     'atp',
            #     7.59660748e-02,  8.68736963e+00,  1.00000000e+00, -1.06064513e+01,
            #    -1.74019484e+01,  4.30497865e+00,  1.94293835e+01,  2.75468133e+01,
            #     9.11843816e-03
            # )
            # test(
            #     'atp',
            #     4.29133622,   4.13506414,   1.        ,
            #     -2.75974958, -27.76296863,  10.00089607, 
            #     8.41102626,   9.59353239, 4.92079208)
            # par = [
            #        0.89242431,   6.17641545, 1.        ,                   
            #     -9.34106654,  -25.135654  , 6.85812174, 
            #      29.20649741,   22.89043865, 9.27507924,
            # ]
                        # 4.13506414, 4.29133622,    1.        ,
            # -27.76296863, -2.75974958, 10.00089607, 
            # 8.41102626,   4.92079208, 9.59353239,
            # par = [5.967268501277159, 5.482414091116128, 1.0, 
            # -9.239379014100262, -27.496862291205026, 23.84342682257863, 
            # 28.662645932889866, 24.98230512512459, 23.631349971142807]
            par.append([
                3.2216783435433016, 8.532383036376764, 1.0,
                -30.80075498859751, -39.65689126829896, 70.06872770454012,
                36.35896896592931, 52.35004431313808, 50.18379684282572
            ])
            par.append([
                5.0872767041095965, 4.534972342088027, 1.0,
                -58.14889504948612, -62.13270322269149, 44.494819747364566, 
                94.60512249360126, 67.22348587054094, 67.98335677618871
            ])

        elif args.molecule == "adp":
            pass
            par = []
            # par.append([
            #     8.3215676 , 2.2923081,    1.        ,
            #     -1.2394423 , -2.76247692, 21.67804333,
            #     21.8467445 ,  12.73135856, 29.52351169,
            # ])
            par.append([1.84442664, 8.66104699,      1.        , 
                 -0.79464195, -11.58323324, 14.29676082,
                 12.54266087,   4.8571578, 20.68036733])
            par.append([1.84442664,   8.66104699,  1.        , 
                -0.79464195,  -11.58323324,  14.29676082,
                 12.54266087,  4.8571578, 20.68036733])

            par.append([0.14989495060633384, 7.7302113553128695, 1.0,
                 -0.7971531363293334, -13.110360504322951, 25.879187286995972,
                 3.694158184764616, 0.4519228151825043, 27.67757153055087])
            par.append([
                9.335627748627699, 3.551630759332944, 1.0,
                -12.935181652571138, -45.79489128218752, 53.235138840912846,
                38.73224218011392, 11.854822890996495, 79.31653941267307
            ])
        elif args.molecule == "amp":
            pass
            par = []
            par.append([
                3.5128614740444695, 4.095576494109553, 1.0, 
                -4.56390637784709, -23.552059818743686, 0.5736994268111175, 
                66.90260598124948, 30.819250045900937, 35.32940755299012
            ])
            par.append([8.0963607832593, 3.7067654351839607, 1.0,
             -19.74887434078677, -26.859897609912796, 1.4017760596582391, 
             84.30873993820705, 32.002524240501344, 52.35361181743063])

        for p in par:
            test(args.molecule, *p)