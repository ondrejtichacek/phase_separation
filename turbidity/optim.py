import os
from scipy.optimize import differential_evolution

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import subprocess
from scipy.interpolate import NearestNDInterpolator

from bp import bp_main

mpl.rcParams['figure.figsize'] = (14,8)

class Optimizer():

    def __init__(self,
        sel = 'atp',
        vol_frac_scaling_x = 10,
        vol_frac_scaling_y = 10,
        ):

        self.sel = sel


        print(self.sel)

        self.vol_frac_scaling_x = vol_frac_scaling_x
        self.vol_frac_scaling_y = vol_frac_scaling_y


    def load_exp_data(self):

        f_sep = {}
        f_mix = {}
        for a in ['atp', 'adp', 'amp']:
            f_sep[a] = f'exp/ph_sep_{a}.dat'
            f_mix[a] = f'exp/mixed_{a}.dat'

        f_sep = f_sep[self.sel]
        f_mix = f_mix[self.sel]

        self.sep_exp = np.loadtxt(f_sep)
        self.mix_exp = np.loadtxt(f_mix)

        self.sep_exp[:,0] *= self.vol_frac_scaling_x
        self.mix_exp[:,0] *= self.vol_frac_scaling_x

        self.sep_exp[:,1] *= self.vol_frac_scaling_y
        self.mix_exp[:,1] *= self.vol_frac_scaling_y

        x = np.concatenate([self.sep_exp[:,0], self.mix_exp[:,0]])
        y = np.concatenate([self.sep_exp[:,1], self.mix_exp[:,1]])
        sep = np.concatenate([np.ones_like(self.sep_exp[:,0]), np.zeros_like(self.mix_exp[:,1])])

        ind = ((x + y) < 1) & (x > 0)  & (y > 0)

        x = x[ind]
        y = y[ind]
        sep = sep[ind]

        self.x = np.ascontiguousarray(x, dtype=np.double)
        self.y = np.ascontiguousarray(y, dtype=np.double)
        self.sep = np.ascontiguousarray(sep, dtype=np.int32)

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


    def compile(self):
        out = subprocess.run(
                ["make"],
                capture_output=True)

        if out.returncode != 0:
            print(out.stdout.decode("utf-8"))
            print(out.stderr.decode("utf-8"))
            raise(ValueError("Something went wrong"))

    def optimize_cy(self, params, maxiter=1):

        x = self.x
        y = self.y
        sep_exp = self.sep

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
        sep = np.ones(n, dtype=np.int32)
        is_on_boundary = np.ascontiguousarray(self.is_on_boundary)

        args = [
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
            n
        ]

        bounds = [params[key] for key in params]

        nc = os.cpu_count() / 2

        result = differential_evolution(self.fun_cy, bounds, args,
            popsize=2*64,
            polish=False,
            maxiter=maxiter,
            workers=1,
            updating='deferred')

        print("best_x: " + " ".join([f"{x}" for x in result.x]))
        # print("best_x: " + ", ".join([f"{x}" for x in result.x]))
        print(result)
        print(result.fun)

        self.result = result

    def fun_cy(self, P,
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
            n):
        
        lp = P[0]
        lm = P[1]
        l0 = P[2]
        Jp = P[3]
        Jm = P[4]
        Jpm = P[5]
        J0 = P[6]
        J0p = P[7]
        J0m = P[8]

        cost = bp_main(
            lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m, 
            x, y,
            hx_xy, hy_xy,
            hx_xpy, hy_xpy,
            hx_xyp, hy_xyp,
            mp_, mm_, xp_, xm_,
            sep, sep_exp,
            is_on_boundary,
            n)

        return cost

    def fun(self, x, optimize, grid=False):
        
        lp = x[0]
        lm = x[1]
        l0 = x[2]
        Jp = x[3]
        Jm = x[4]
        Jpm = x[5]
        J0 = x[6]
        J0p = x[7]
        J0m = x[8]

        lim_x = 1
        lim_y = 1

        cmd = ["./turb_fit",
            self.f_sep,
            self.f_mix,
            f"{int(grid)}", f"{int(optimize)}",
            f"{lim_x}", f"{lim_y}",
            f"{self.vol_frac_scaling_x}", f"{self.vol_frac_scaling_y}",
            f"{lp}", f"{lm}", f"{l0}",
            f"{Jp}", f"{Jm}", f"{Jpm}",
            f"{J0}", f"{J0p}", f"{J0m}"]

        # print(" ".join(cmd))

        if optimize:
            timeout = 10
        else:
            timeout = None

        try:
            out = subprocess.run(cmd,
                timeout=timeout,
                capture_output=True)

            if out.returncode != 0:
                print(out.stdout.decode("utf-8"))
                print(out.stderr.decode("utf-8"))
                raise(ValueError("Something went wrong"))

            outstr = out.stdout.decode("utf-8");

            if optimize:
                cost = np.array(outstr)
                cost = cost.astype(float)
            else:
                cost = np.NaN

        except subprocess.TimeoutExpired as E:
            cost = np.inf
            return cost

        # all_costs.append(cost)

        return cost

    def optimize(self, params, maxiter=1):
        bounds = [params[key] for key in params]

        nc = os.cpu_count() / 2

        result = differential_evolution(self.fun, bounds, [True],
            popsize=int(np.max([12, 2*nc])),
            maxiter=maxiter,
            workers=nc,
            updating='deferred')

        print(" ".join([f"{x}" for x in result.x]))
        print(result.fun)

        self.result = result

    def plot_result(self):
        sep = np.loadtxt('sep')
        mix = np.loadtxt('mix')

        sep_exp = np.loadtxt(self.f_sep)
        mix_exp = np.loadtxt(self.f_mix)

        sep_exp[:,0] *= self.vol_frac_scaling_x
        mix_exp[:,0] *= self.vol_frac_scaling_x

        sep_exp[:,1] *= self.vol_frac_scaling_y
        mix_exp[:,1] *= self.vol_frac_scaling_y

        plt.figure()
        if sep.size > 0:
            plt.plot(sep[:,0], sep[:,1], 's')
        if mix.size > 0:
            plt.plot(mix[:,0], mix[:,1], 's')

        plt.figure()
        plt.plot(sep_exp[:,0], sep_exp[:,1], 's')
        plt.plot(mix_exp[:,0], mix_exp[:,1], 's')

    def plot_result_int(self, xm, ym):

        res = np.loadtxt('result.txt')

        x = res[:,0]
        y = res[:,1]
        z = res[:,2]

        sep_exp = np.loadtxt(self.f_sep)
        mix_exp = np.loadtxt(self.f_mix)

        sep_exp[:,0] *= self.vol_frac_scaling_x
        mix_exp[:,0] *= self.vol_frac_scaling_x

        sep_exp[:,1] *= self.vol_frac_scaling_y
        mix_exp[:,1] *= self.vol_frac_scaling_y

        interp_model = NearestNDInterpolator(list(zip(x, y)), z)

        x = np.concatenate([sep_exp[:,0], mix_exp[:,0]])
        y = np.concatenate([sep_exp[:,1], mix_exp[:,1]])
        z = np.concatenate([np.zeros_like(sep_exp[:,1]), np.ones_like(mix_exp[:,1])])

        interp_exp = NearestNDInterpolator(list(zip(x, y)), z)

        plt.figure()

        xp = np.linspace(0, xm, 200)
        yp = np.linspace(0, ym, 200)
        X, Y = np.meshgrid(xp, yp)

        Z = 1-interp_exp(X, Y)
        # plt.pcolormesh(X, Y, Z, shading='auto', cmap='bwr', vmin=-1, vmax=1, alpha=0.75)
        plt.pcolormesh(X, Y, Z, shading='auto', cmap='binary', vmin=0, vmax=3)

        Z = 1-interp_model(X, Y)
        Z[Z == 0] = np.nan
        plt.pcolormesh(X, Y, Z, shading='auto', cmap='bwr', vmin=-1, vmax=1, alpha=0.5)
        # plt.plot(sep[:,0], sep[:,1], "wx", label="sep")
        plt.legend()
        plt.show()

    def plot_result_grid(self):
        res = np.loadtxt('result.txt')

        x = res[:,0]
        y = res[:,1]
        z = res[:,2]

        z = np.reshape(z, (256, 256))

        plt.figure()
        plt.pcolormesh(z)
        plt.show()