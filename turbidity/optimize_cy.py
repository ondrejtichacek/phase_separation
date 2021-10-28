import matplotlib.pyplot as plt
import numpy as np
from bp import bp_main

from optim import Optimizer

def main():

    opt = Optimizer(
        sel = 'atp',
        vol_frac_scaling_x = 10,
        vol_frac_scaling_y = 10,
    )

    opt.load_exp_data()

    bounds = {
        'lp': (0, 10),
        'lm': (0, 10),
        'l0': (0, 10),
        'Jp': (-20, 20),
        'Jm': (-20, 20),
        'Jpm': (-20, 20),
        'J0': (0, 0),
        'J0p': (-20, 20),
        'J0m': (-20, 20),
    }

    opt.optimize_cy(bounds, maxiter=1000)


def test(
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
        sel = 'atp',
        vol_frac_scaling_x = 10,
        vol_frac_scaling_y = 10,
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
    sep = np.ones(n, dtype=np.int32)
    is_on_boundary = np.zeros(n, dtype=np.int32)

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

if __name__ == '__main__':
    main()
    # test(1.3427142999104076, 4.834397995071055, 7.165882609064637, -13.8879476026868, -3.397763543728307, 8.613696431835756, 0.0, 3.8578599929869073, 1.2160291425758762)
    # test(0.03534053655146341, 4.12870963025583, 7.772578929162346, -8.109507006971366, -0.8462496690912147, 6.601439856942313, 0.0, 12.913832887901071, -1.1870391753824072)
    # test(0.6669275956773593, 9.028830574843594, 2.156316388827224, 1.834085729075321, -18.293449887182717, 3.573709802196281, 0.0, 8.83485531599549, -9.66742726423524)
    # test(0.617672762653565, 2.0839170707637726, 9.102981023364176, -5.259882943309946, 3.8552178387696845, 17.391008497954488, 0.0, 10.055549226223555, 4.600405623487971)