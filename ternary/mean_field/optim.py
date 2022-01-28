import os
from scipy.optimize import differential_evolution, dual_annealing

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

import numpy as np
import subprocess
from scipy.interpolate import NearestNDInterpolator

from common import timer

from bp import bp_main

class Optimizer():

    def __init__(self,
        lattice_connectivity,
        sel,
        vol_frac_scaling_x = 1,
        vol_frac_scaling_y = 1,
        use_cont_err = False,
        ):

        self.lattice_connectivity = lattice_connectivity

        self.use_cont_err = use_cont_err

        self.sel = sel

        self.vol_frac_scaling_x = vol_frac_scaling_x
        self.vol_frac_scaling_y = vol_frac_scaling_y

        self.logfile = f"log_{self.sel}.txt"

        # volume fractions dx and dy used for numerical calculations
        self.dx = 1e-6
        self.dy = 1e-6

        print(self.sel)

    def load_mock_data(self, fname):

        data = np.loadtxt(fname)

        x = data[:,0]
        y = data[:,1]
        sep = data[:,2]
        sep_cont = (1 + data[:,3]) / 2

        ind = ((x + 10*self.dx + y) < 1) & ((x + y + 10*self.dy) < 1) & (x > 0) & (y > 0)

        x = x[ind]
        y = y[ind]

        sep = sep[ind]

        self.x = np.ascontiguousarray(x, dtype=np.double)
        self.y = np.ascontiguousarray(y, dtype=np.double)
        self.sep = np.ascontiguousarray(sep, dtype=np.int32)
        self.sep_cont = np.ascontiguousarray(sep_cont, dtype=np.double)

        n = self.x.size

        min_x = np.min(self.x)
        min_y = np.min(self.y)
        max_x = np.max(self.x)
        max_y = np.max(self.y)

        tolx = 1e-5
        toly = 1e-5

        is_on_boundary = np.zeros(n, dtype=np.int32)

        for i in range(n):
            if ((self.x[i] < min_x + tolx)
                or (self.x[i] > max_x - tolx)
                or (self.y[i] < min_y + toly)
                or (self.y[i] > max_y - toly)):
                is_on_boundary[i] = 1
            else:
                is_on_boundary[i] = 0

        # is_on_boundary = np.zeros_like(is_on_boundary)

        self.is_on_boundary = is_on_boundary

    def load_exp_data_new(self, fname):

        data = np.loadtxt(fname)

        x = data[:,0]
        y = data[:,1]

        self.orig_x = np.copy(x)#/10)
        self.orig_y = np.copy(y)#/10)
        
        sep_cont = data[:,2]
        sep = sep_cont >= 0.5

        x /= x.max() * 2.01
        y /= y.max() * 2.01

        # ind = ((x + 10*self.dx + y) < 1) & ((x + y + 10*self.dy) < 1) & (x > 0) & (y > 0)

        # print(x)
        # print(ind)

        # x = x[ind]
        # y = y[ind]

        # sep = sep[ind]

        self.x = np.ascontiguousarray(x, dtype=np.double)
        self.y = np.ascontiguousarray(y, dtype=np.double)
        self.sep = np.ascontiguousarray(sep, dtype=np.int32)
        self.sep_cont = np.ascontiguousarray(sep_cont, dtype=np.double)

        n = self.x.size

        min_x = np.min(self.x)
        min_y = np.min(self.y)
        max_x = np.max(self.x)
        max_y = np.max(self.y)

        tolx = 1e-5
        toly = 1e-5

        is_on_boundary = np.zeros(n, dtype=np.int32)

        for i in range(n):
            if ((self.x[i] < min_x + tolx)
                or (self.x[i] > max_x - tolx)
                or (self.y[i] < min_y + toly)
                or (self.y[i] > max_y - toly)):
                is_on_boundary[i] = 1
            else:
                is_on_boundary[i] = 0

        # is_on_boundary = np.zeros_like(is_on_boundary)

        self.is_on_boundary = is_on_boundary

    def load_exp_data(self, limit=None, extended=True):

        f_sep = {}
        f_mix = {}
        for a in ['atp', 'adp', 'amp']:
            f_sep[a] = f'exp/ph_sep_{a}.dat'
            f_mix[a] = f'exp/mixed_{a}.dat'

        f_sep = f_sep[self.sel]
        f_mix = f_mix[self.sel]

        sep_exp = np.loadtxt(f_sep)
        mix_exp = np.loadtxt(f_mix)

        x = np.concatenate([sep_exp[:,0], mix_exp[:,0]])
        y = np.concatenate([sep_exp[:,1], mix_exp[:,1]])
        sep = np.concatenate([np.ones_like(sep_exp[:,0]), np.zeros_like(mix_exp[:,0])])

        if limit is not None:
            ind = ((limit['x'][0] < x) 
                    & (x < limit['x'][1]) 
                    & (limit['y'][0] < y) 
                    & (y < limit['y'][1]))

            x = x[ind]
            y = y[ind]

            sep = sep[ind]

        x *= self.vol_frac_scaling_x
        y *= self.vol_frac_scaling_y

        ind = ((x + 10*self.dx + y) < 1) & ((x + y + 10*self.dy) < 1) & (x > 0) & (y > 0)

        x = x[ind]
        y = y[ind]

        sep = sep[ind]

        if extended is True:
            y1 = np.linspace(np.min(y), 0.9, 40)
            x1 = np.full_like(y1, np.min(x))

            x2 = np.linspace(np.min(x), 0.9, 40)
            y2 = np.full_like(x2, np.min(y))

            x = np.concatenate([x, x1, x2])
            y = np.concatenate([y, y1, y2])

            sep_extra = np.zeros_like(x, dtype=np.int32)

            sep = np.concatenate([sep, sep_extra])

        self.x = np.ascontiguousarray(x, dtype=np.double)
        self.y = np.ascontiguousarray(y, dtype=np.double)
        self.sep = np.ascontiguousarray(sep, dtype=np.int32)
        self.sep_cont = np.ascontiguousarray(sep, dtype=np.double)
        n = self.x.size

        min_x = np.min(self.x)
        min_y = np.min(self.y)
        max_x = np.max(self.x)
        max_y = np.max(self.y)

        tolx = 1e-5 * self.vol_frac_scaling_x
        toly = 1e-5 * self.vol_frac_scaling_y

        is_on_boundary = np.zeros(n, dtype=np.int32)

        for i in range(n):
            if ((self.x[i] < min_x + tolx)
                or (self.x[i] > max_x - tolx)
                or (self.y[i] < min_y + toly)
                or (self.y[i] > max_y - toly)):
                is_on_boundary[i] = 1
            else:
                is_on_boundary[i] = 0

        self.is_on_boundary = is_on_boundary

        # plt.figure()
        # plt.plot(self.x[self.is_on_boundary == 0], self.y[self.is_on_boundary == 0], 'x')
        # plt.plot(self.x[self.is_on_boundary == 1], self.y[self.is_on_boundary == 1], 'x')
        # plt.show()

    def assert_volume_fractions(self, x, y):
        all_good = np.all(
            ((x + 10*self.dx + y) < 1)  & ((x + y + 10*self.dy) < 1)
             & (x > 0) & (y > 0))
        assert(all_good)

    def optimize_cy(self, params, maxiter=1):

        x = self.x
        y = self.y

        self.assert_volume_fractions(x, y)

        sep_exp = self.sep

        num_mix = np.count_nonzero(sep_exp == 0)
        num_sep = np.count_nonzero(sep_exp == 1)

        num_mix /= sep_exp.size
        num_sep /= sep_exp.size

        n = x.size

        t_double = np.float64
        t_bool = np.int32

        hx_xy = np.zeros(n, dtype=t_double)
        hy_xy = np.zeros(n, dtype=t_double)
        
        hx_xpy = np.zeros(n, dtype=t_double)
        hy_xpy = np.zeros(n, dtype=t_double)
        hx_xyp = np.zeros(n, dtype=t_double)
        hy_xyp = np.zeros(n, dtype=t_double)
        
        mp_ = np.zeros(n, dtype=t_double)
        mm_ = np.zeros(n, dtype=t_double)
        xp_ = np.zeros(n, dtype=t_double)
        xm_ = np.zeros(n, dtype=t_double)
        
        sep_model = np.ones(n, dtype=t_bool)

        use_cont_err = t_bool(self.use_cont_err)

        sep_exp_cont = np.ascontiguousarray(self.sep_cont, dtype=t_double)
        is_on_boundary = np.ascontiguousarray(self.is_on_boundary, dtype=t_bool)

        print(f"using cont. error {use_cont_err}")

        args = [
            x, y,
            hx_xy, hy_xy,
            hx_xpy, hy_xpy,
            hx_xyp, hy_xyp,
            mp_, mm_, xp_, xm_,
            sep_model, sep_exp,
            sep_exp_cont, use_cont_err,
            is_on_boundary,
            n, num_mix, num_sep,
        ]

        bounds = [params[key] for key in params]

        # nc = os.cpu_count() / 2

        popsize = 2*64
        popsize = 32
        popsize = 16

        maxeval = (maxiter + 1) * popsize * len(bounds)

        print(f"Maximum fun evals: (maxiter + 1) * popsize * len(x) = {maxeval}")
        print(f"  = {maxeval * 0.064 / 60} min")

        self.min_cost = np.inf
        self.it = 0

        # result = dual_annealing(self.fun_cy, bounds, args)

        result = differential_evolution(self.fun_cy, bounds, args,
            popsize=popsize,
            # polish=False,
            # recombination=0.2,
            # mutation=(0.5, 1.5)
            workers=1,
            # updating='deferred',
            maxiter=maxiter)

        print("best_x: " + " ".join([f"{x:3g}" for x in result.x]))
        print("best_x: " + ", ".join([f"{x:3g}" for x in result.x]))
        print(result)
        print(result.fun)

        self.result = result

        with open(self.logfile, 'a') as f:
            f.write(f"END: {result.fun} " + ", ".join([f"{x:3g}" for x in result.x]) + "\n")

    def fun_cy(self, P,
            x, y,
            hx_xy, hy_xy,
            hx_xpy, hy_xpy,
            hx_xyp, hy_xyp,
            mp_, mm_, xp_, xm_,
            sep, sep_exp, sep_exp_cont, use_cont_err,
            is_on_boundary, n, nmix, nsep):

        
        lp = P[0]
        lm = P[1]
        l0 = P[2]
        Jp = P[3]
        Jm = P[4]
        Jpm = P[5]
        J0 = P[6]
        J0p = P[7]
        J0m = P[8]
        scale_x = P[9]
        scale_y = P[10]

        cost = bp_main(
            lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m,
            scale_x, scale_y,
            self.dx, self.dy,
            x, y,
            hx_xy, hy_xy,
            hx_xpy, hy_xpy,
            hx_xyp, hy_xyp,
            mp_, mm_, xp_, xm_,
            sep, sep_exp, sep_exp_cont, use_cont_err,
            is_on_boundary,
            self.lattice_connectivity,
            n, nmix, nsep)

        self.it += 1

        if np.mod(self.it, 100) == 0:
            print(f"iteration {self.it}")

        if cost < self.min_cost:
            print(f"{cost:2g} : " + ", ".join([f"{x:3g}" for x in P]))
            self.min_cost = cost

            with open(self.logfile, 'a') as f:
                f.write(f"{cost:2g} : " + ", ".join([f"{x:3g}" for x in P]) + "\n")

        return cost

    @timer
    def bp_wrapper(self, x, y, lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m,
            sep_exp=None, sep_exp_cont=None, is_on_boundary=None):

        self.assert_volume_fractions(x, y)

        n = x.size

        # memory
        hx_xy = np.zeros(n)
        hy_xy = np.zeros(n)
        hx_xpy = np.zeros(n)
        hy_xpy = np.zeros(n)
        hx_xyp = np.zeros(n)
        hy_xyp = np.zeros(n)
        mp = np.zeros(n)
        mm = np.zeros(n)
        xp = np.zeros(n)
        xm = np.zeros(n)

        print(f"using cont. error {self.use_cont_err}")

        # dummy data
        if sep_exp is None:
            sep_exp = np.ones(n, dtype=np.int32)
        if sep_exp_cont is None:
            sep_exp_cont = np.ones(n, dtype=np.double)
        if is_on_boundary is None:
            is_on_boundary = np.zeros(n, dtype=np.int32)

        # return array
        sep = np.ones(n, dtype=np.int32)

        is_on_boundary = np.ascontiguousarray(is_on_boundary)

        err = bp_main(
            lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m, 
            1, 1,
            self.dx, self.dy,
            x, y,
            hx_xy, hy_xy,
            hx_xpy, hy_xpy,
            hx_xyp, hy_xyp,
            mp, mm, xp, xm,
            sep, sep_exp, sep_exp_cont, self.use_cont_err,
            is_on_boundary, self.lattice_connectivity, n, 1, 1,
        )

        return err, sep

    @timer
    def bp_test_speed(self, x, y, lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m,
            sep_exp=None, sep_exp_cont=None, is_on_boundary=None, calculate=True):

        self.assert_volume_fractions(x, y)

        n = x.size

        # memory
        hx_xy = np.zeros(n)
        hy_xy = np.zeros(n)
        hx_xpy = np.zeros(n)
        hy_xpy = np.zeros(n)
        hx_xyp = np.zeros(n)
        hy_xyp = np.zeros(n)
        mp = np.zeros(n)
        mm = np.zeros(n)
        xp = np.zeros(n)
        xm = np.zeros(n)

        # dummy data
        if sep_exp is None:
            sep_exp = np.ones(n, dtype=np.int32)
        if sep_exp_cont is None:
            sep_exp_cont = np.ones(n, dtype=np.double)
        if is_on_boundary is None:
            is_on_boundary = np.zeros(n, dtype=np.int32)

        # return array
        sep = np.ones(n, dtype=np.int32)

        is_on_boundary = np.ascontiguousarray(is_on_boundary)

        for i in np.arange(100):
            p = [lp + np.random.uniform(0,10),
                lm + np.random.uniform(0,10),
                l0 + np.random.uniform(0,10),
                Jp + np.random.uniform(-5,5),
                Jm + np.random.uniform(-5,5),
                Jpm + np.random.uniform(-5,5),
                J0 + np.random.uniform(-5,5),
                J0p + np.random.uniform(-5,5),
                J0m + np.random.uniform(-5,5)]

            if calculate:
                err = bp_main(
                    *p,
                    1, 1,
                    self.dx, self.dy,
                    x, y,
                    hx_xy, hy_xy,
                    hx_xpy, hy_xpy,
                    hx_xyp, hy_xyp,
                    mp, mm, xp, xm,
                    sep, sep_exp, sep_exp_cont, self.use_cont_err,
                    is_on_boundary, self.lattice_connectivity, n, 1, 1,
                )
            else:
                err = 0

        return err, sep

    def plot_phase_diagram(self, X, Y, Z):
        ZZ = np.copy(Z)
        ZZ[ZZ == -1] = np.nan

        hfig = plt.figure(figsize=(3,3))

        # colors = ['#CCC', 'crimson']
        # colors = ['#CCC', 'purple']
        colors = ['#CCC', 'blue']
        # colors = ['#CCC', 'green']
        ax = plt.gca()
        
        ax.pcolormesh(ZZ, #X, Y, ZZ, 
            #edgecolors='w', linewidth=0.1, 
            cmap=LinearSegmentedColormap.from_list('', colors))
        
        plt.xticks([0,ZZ.shape[0]], labels=[0,f"{X.flatten().max():.2f}"])
        plt.yticks([0,ZZ.shape[1]], labels=[0,f"{Y.flatten().max():.2f}"])

        ax.set_aspect('equal')
        legend_elements = [Patch(facecolor=color, edgecolor='w') for color in colors]
        
        ax.legend(handles=legend_elements,
                labels=[f"mixed", f"separated"])
        plt.xlabel('vol. frac. #1')
        plt.ylabel('vol. frac. #2')
        plt.tight_layout(pad=2)

        return hfig

    def CW_set(self, CP, CM, Jp, Jm, Jpm):

        sep = np.ones_like(CP)

        for i, (cp, cm) in enumerate(zip(CP, CM)):
            inv = 1 / (1 - cp - cm)
            tr = -2*(Jp + Jm) + 1/cp +1/cm + 2*inv
            det = (-2*Jp + 1/cp + inv) * (-2*Jm + 1/cm + inv) - (-Jpm + inv)*(-Jpm + inv)
            if (tr > 0 and det > 0):
                sep[i] = 0

        return sep