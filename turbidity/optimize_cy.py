import matplotlib.pyplot as plt
import numpy as np
from bp import bp_main

from optim import Optimizer

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("molecule", help="")

args = parser.parse_args()

scale = {
    'amp': 232,
    'adp': 50,
    'atp': 20,
    'poly_arg': 20,
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

    bounds = {
        'lp': (0, 10),
        'lm': (0, 10),
        'l0': (1, 1),
        'Jp': (-30, 0),
        'Jm': (-30, 0),
        'Jpm': (0, 30),
        'J0': (0, 30),
        'J0p': (0, 30),
        'J0m': (0, 30),
    }

    opt.optimize_cy(bounds, maxiter=100)


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

    # sel = 'amp'
    # sel = 'adp'
    # sel = 'atp'

    opt = Optimizer(
        sel=sel,
        vol_frac_scaling_x=scale[sel],
        vol_frac_scaling_y=scale['poly_arg'],
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

    # for i in range(n):
    #     x[i] = xx[i]
    #     y[i] = yy[i]

    err = bp_main(
        lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m, 
        x, y,
        hx_xy,
        hy_xy,
        hx_xpy,
        hy_xpy,
        hx_xyp,
        hy_xyp,
        mp_, mm_, xp_, xm_,
        sep,
        sep_exp,
        is_on_boundary,
        n,
        )

    print(err)

    i1 = sep != 0
    i2 = sep == 0

    plt.figure()
    plt.plot(x[i1], y[i1], 's')
    plt.plot(x[i2], y[i2], 's')
    plt.show()

    i1 = sep_exp != 0
    i2 = sep_exp == 0

    plt.figure()
    plt.plot(x[i1], y[i1], 's')
    plt.plot(x[i2], y[i2], 's')
    plt.show()

if __name__ == '__main__':
    # main()
    # test(1.3427142999104076, 4.834397995071055, 7.165882609064637, -13.8879476026868, -3.397763543728307, 8.613696431835756, 0.0, 3.8578599929869073, 1.2160291425758762)
    # test(0.03534053655146341, 4.12870963025583, 7.772578929162346, -8.109507006971366, -0.8462496690912147, 6.601439856942313, 0.0, 12.913832887901071, -1.1870391753824072)
    # test(0.6669275956773593, 9.028830574843594, 2.156316388827224, 1.834085729075321, -18.293449887182717, 3.573709802196281, 0.0, 8.83485531599549, -9.66742726423524)
    # test(0.617672762653565, 2.0839170707637726, 9.102981023364176, -5.259882943309946, 3.8552178387696845, 17.391008497954488, 0.0, 10.055549226223555, 4.600405623487971)
    # test(
    #     0.0943951 ,  2.36255804,  4.72634985, -4.38707623,  3.29483375,
    #    -2.16434615,  0.        , 10.95057411,  6.50080107
    # )
    # test(
    #     0.10814398,   3.43883932,   7.10541945,  17.40753283,
    #    -10.34348843,   1.95482393,   0.        ,  19.77449184,
    #      1.61894913
    # )
    # test(
    #     0.09327044,  7.44731216,  5.52221438, -5.41972529, 12.42907554,
    #     1.87661773,  0.        , 20.73854286, 13.49874229
    # )
    # test(
    #     0.43111956,   5.71757559,   4.78446164,  13.60170526,
    #    -13.60371612,  28.95391746,   0.        ,  19.09137199,
    #      7.86666062
    # )
    # test(
    #     0.07014874,   1.81376872,   4.63802223,  -7.87018426,
    #    -10.3582634 ,   3.4735479 ,   0.        ,  15.05304947,
    #     -5.56339495
    # )
    # # ATP 27.131782945736433
    # test(
    #     'atp',
    #     0.07015072,   7.07489837,   5.43583425,
    #     -23.9875863 , -10.2214982 ,   4.20174434,
    #     17.49648035,  24.45509315, 3.25529366
    # )
    # test(
    #     'atp',
    #     0.07147059,  7.41206443,  2.70655304,  3.12824289, 24.2155885 ,
    #    13.6975276 , 20.87782068, 18.41593364, 22.08146782
    # )
    # # ADP 3.962101636520241
    # test(
    #     'adp',
    #     0.43109974,   9.95798374,   7.82765335,  -9.79366477,
    #    -28.40124494,   0.95289702, -27.14318311,  -5.85067836,
    #    -22.33546241
    # )
    # test(
    #     'adp',
    #      0.40539653,   7.62329079,   4.84749126, -11.94038417,
    #      3.97931193,  25.22924649, -12.91147737,   6.30994274,
    #      3.97394686
    # )
    # # AMP 22.22222222222222
    # test(
    #     'amp',
    #     2.38773025,  7.72065442,  3.88500279, -0.4264304 , -2.95089022,
    #    26.74393907, 10.57298631, 26.6045048 ,  7.28890944
    # )
    # test(
    #     'amp',
    #     0.06431438,   2.00148103,   0.26630715,  -1.43729572,
    #     21.76586637,   8.25110032, -13.96775667,  29.25753533,
    #     26.92769403
    # )
    # ---------------------------------
    test(
        'atp',
        0.60644092,   2.02152218,   1.        , -28.04578561,
       -25.45601227,  20.81318829,   0.5831698 ,   8.37261113,
         9.36110971
    )