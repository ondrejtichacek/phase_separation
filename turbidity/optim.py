import os
from scipy.optimize import differential_evolution

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import subprocess
from scipy.interpolate import NearestNDInterpolator

mpl.rcParams['figure.figsize'] = (14,8)

class Optimizer():

    def __init__(self,
        sel = 'atp',
        vol_frac_scaling_x = 10,
        vol_frac_scaling_y = 10,
        ):

        self.vol_frac_scaling_x = vol_frac_scaling_x
        self.vol_frac_scaling_y = vol_frac_scaling_y

        f_sep = {}
        f_mix = {}
        for a in ['atp', 'adp', 'amp']:
            f_sep[a] = f'exp/ph_sep_{a}.dat'
            f_mix[a] = f'exp/mixed_{a}.dat'

        self.f_sep = f_sep[sel]
        self.f_mix = f_mix[sel]

    def compile(self):
        out = subprocess.run(
                ["make"],
                capture_output=True)

        if out.returncode != 0:
            print(out.stdout.decode("utf-8"))
            print(out.stderr.decode("utf-8"))
            raise(ValueError("Something went wrong"))

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