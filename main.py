# conda create -n phase-sep
# conda activate phase-sep
# conda install -c conda-forge numpy scipy matplotlib ipykernel tqdm

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed
import itertools
import scipy.optimize

import subprocess
import shutil

def parameter_product(**kwargs):
    L = []
    for V in itertools.product(*kwargs.values()):
        kw = {}
        for i, key in enumerate(kwargs.keys()):
            kw[key] = V[i]
        L.append(kw)

    return L

def mock_simulate(**kwargs):
    s = f"sim with args {kwargs}"
    print(s)
    return s

def simulator(fun, params, parallel=True):
    res = []
    if parallel is True:
        res = Parallel(n_jobs=6, verbose=10) (delayed(fun)(**p) for p in params)
    else:
        for p in params:
            res.append(fun(**p))
    
    return res

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

class LatticePhaseReact():
    def __init__(
        self,
        volume_fraction=0.3,
        beta_range=np.linspace(0, 3, 30+1),
        interaction_range=np.linspace(0, 1, 6),
        num_components=20,
        lattice_size=100,
        extended_neighborhood=True,
        log_concentration=False,
        wrkdir=None):

        self.interaction_range = interaction_range
        self.volume_fraction = volume_fraction
        self.beta_range = beta_range

        self.num_components = num_components
        self.lattice_size = lattice_size
        self.extended_neighborhood = extended_neighborhood

        self.log_concentration = log_concentration

        self.condensation_interaction_version = 'default'
        self.reaction_interaction_version = 'default'

        if wrkdir is None:
            wrkdir = os.getcwd()
        
        self.wrkdir = Path(wrkdir)
        self.res_dir = self.wrkdir / "results"

        # shutil.rmtree(self.res_dir)
        self.res_dir.mkdir(exist_ok=True)

    def get_defining_parameters(self, part):

        defining_parameters = [
            ('volume_fraction', 'Vf'),
            ('lattice_size', 'd'),
            ('num_components', 'n'),
            ('condensation_interaction_version', 'vC'),
        ]

        if part == 'reaction':
            defining_parameters.append(
                ('reaction_interaction_version', 'vR'))

        return defining_parameters

    def get_char_string(self, part):
        s = ""
        for (p, n) in self.get_defining_parameters(part):
            s += f"{n}_{getattr(self, p)}"
        return s

    def get_dir_condensation(self):
        return self.res_dir / (f'condensation_' + self.get_char_string('condensation'))

    def get_dir_reaction(self, interaction):
        return self.res_dir / (f'reaction_' + f'interaction_{interaction:.3f}_' + self.get_char_string('reaction'))

    def make(self, wrkdir=None):
        if wrkdir is None:
            wrkdir = self.wrkdir
        
        os.chdir(wrkdir)

        CFLAGS = [
            "-march=x86-64",
            "-mtune=native",
            "-O2",
            "-pipe",
            "-fno-plt",
            "-fexceptions",
	        "-mavx2",
            "-Wp,-D_FORTIFY_SOURCE=2",
            "-Wformat",
            "-Werror=format-security",
            "-fstack-clash-protection",
            "-fcf-protection",
        ]

        DFLAGS = [
            f"-D NUM_COMPONENTS={self.num_components}",
            f"-D LATTICE_SIZE={self.lattice_size}",
        ]

        if self.log_concentration is True:
            DFLAGS.append(f"-D LOG_CONCENTRATION=1")
        else:
            DFLAGS.append(f"-D LOG_CONCENTRATION=0")
        
        if self.extended_neighborhood is True:
            DFLAGS.append(f"-D EXTENDED_NEIGHBORHOOD=1")
        else:
            DFLAGS.append(f"-D EXTENDED_NEIGHBORHOOD=0")

        out = subprocess.run(
            ["g++"] + CFLAGS + DFLAGS + [self.wrkdir / "react_phase_sep.cpp"],
            capture_output=True)

        if out.returncode > 0:
            print(out)
            raise(ValueError("Something went wrong"))

    def simulate_condensation(self, num_sim_cond=1000):

        sim_dir = self.get_dir_condensation()
        
        if sim_dir.exists():
            shutil.rmtree(sim_dir)
        sim_dir.mkdir(exist_ok=True)
        
        os.chdir(sim_dir)

        self.save_condensate_interaction_matrix()

        self.make(wrkdir=sim_dir)

        out = subprocess.run(["./a.out",
            f"0",
            f"{self.volume_fraction}",
            f"{self.beta_range[0]}",
            f"{self.beta_range[-1]}",
            f"{self.beta_range.size}",
            f"{num_sim_cond}",
            f"0"],
            capture_output=True)

        if out.returncode > 0:
            print(out)
            raise(ValueError("Something went wrong"))

    def perform_reaction_simulation(self, interaction, num_sim_react):

        cond_sim_dir = self.get_dir_condensation()
        sim_dir = self.get_dir_reaction(interaction)
        
        if sim_dir.exists():
            shutil.rmtree(sim_dir)
        sim_dir.mkdir(exist_ok=True)
        
        shutil.copytree(cond_sim_dir, sim_dir, dirs_exist_ok=True)

        os.chdir(sim_dir)

        out = subprocess.run(["./a.out",
            f"{interaction}",
            f"{self.volume_fraction}",
            f"{self.beta_range[0]}",
            f"{self.beta_range[-1]}",
            f"{self.beta_range.size}",
            f"0",
            f"{num_sim_react}"],
            capture_output=True)

        if out.returncode > 0:
            print(out)

    def simulate_reaction(self, num_sim_react=1000, parallel=True):

        self.save_reaction_interaction_matrix()

        if parallel is False:
            for interaction in self.interaction_range:
                self.perform_reaction_simulation(interaction, num_sim_react)
        else:
            pfun = lambda x: self.perform_reaction_simulation(x, num_sim_react)
            Parallel(n_jobs=6) (delayed(pfun)(ir) for ir in self.interaction_range)

    def generate_reaction_interaction(self, ver="ones", r=0.4, s=0.2):

        self.reaction_interaction_version = ver

        q = self.num_components

        if ver == "ones":
            I = np.ones((q+1, q+1), dtype=float)
            I[0,:] = 0
            I[:,0] = 0

        elif ver == "rand1":
            I = np.empty((q+1, q+1), dtype=float)

            for i in range(1, q+1):
                # interaction with solute (=0 default)
                I[0,i] = 0;
                I[i,0] = 0;

            for i in range(1, q+1):
                for j in range(1, q+1):
                    I[i,j] = np.random.uniform(0, 2)

            for i in range(1, q+1):
                for j in range(i, q+1):
                    I[i,j] = np.random.uniform(1, 2)

            for i in range(1, q+1):
                for j in range(1, i):
                    I[i,j] = np.random.uniform(0, 1)

            for i in range(1, q+1):
                for j in range(i, q+1):
                    if (i == j):
                        I[i,j] = 0
                    if (i + 1 == j):
                        I[i,j] = 2
        
        elif ver == "rand2":
            I = np.random.uniform(0+r, 2-r, size=(q+1, q+1))
            
            I[0,:] = np.random.uniform(0, s, size=(q+1))
            I[:,0] = np.random.uniform(0, s, size=(q+1))

        self.I = I

    def generate_condensate_interaction(self, ver="ones", r=0.4, s=0.2):

        self.condensation_interaction_version = ver

        q = self.num_components

        if ver == "ones":
            J = np.ones((q+1, q+1), dtype=float)
            J[0,:] = 0
            J[:,0] = 0

        elif ver == "rand1":
            J = np.empty((q+1, q+1), dtype=float)

            for i in range(1, q+1):
                # interaction with solute (=0 default)
                J[0,i] = 0;
                J[i,0] = 0;

            for i in range(1, q+1):
                for j in range(i, q+1):
                    if (i == j):
                        J[i,j] = 1 # self interaction  (=1 default)
                    elif (j >= i + 1):
                        J[i,j] = np.random.uniform(0, 2)
                    J[j,i] = J[i,j]

        elif ver == "rand2":
            J = np.random.uniform(0+r, 2-r, size=(q+1, q+1))
            
            for i in range(1, q+1):
                for j in range(i, q+1):
                    J[j,i] = J[i,j]

            # interaction with solute
            J[0,:] = np.random.uniform(0, s, size=(q+1))
            J[:,0] = np.random.uniform(0, s, size=(q+1))

        # print(J)

        self.J = J


    def save_condensate_interaction_matrix(self):

        sim_dir = self.get_dir_condensation()
        np.savetxt(sim_dir / "J.csv", self.J, fmt='%.6f')

    def save_reaction_interaction_matrix(self):

        sim_dir = self.get_dir_condensation()
        np.savetxt(sim_dir / "I.csv", self.I, fmt='%.6f')

    def plot_condensation_interaction_matrix(self):

        plt.figure()
        sim_dir = self.get_dir_condensation()

        J = np.genfromtxt(sim_dir / f'J.csv')

        #J[J == 0] = np.NaN
        plt.imshow(J, cmap=plt.cm.Spectral)
        plt.title(f"J -- interaction")
        plt.xlabel(f"Component (enzyme)")
        plt.ylabel(f"Component (enzyme)")
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("interaction strength")

    def plot_reaction_interaction_matrix(self):
        
        plt.figure()
        sim_dir = self.get_dir_condensation()

        I = np.genfromtxt(sim_dir / f'I.csv')

        #I[I == 0] = np.NaN
        plt.imshow(I, cmap=plt.cm.Spectral)
        plt.title(f"I -- interaction")
        plt.xlabel(f"Component (enzyme)")
        plt.ylabel(f"Solute (reactant)")
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("interaction strength")

    def get_time_to_react(self):
        time_to_react = []
        time_to_react_err = []

        for i, interaction in enumerate(self.interaction_range):

            sim_dir = self.get_dir_reaction(interaction)

            data = np.genfromtxt(sim_dir / f'reaction_tempo.csv')
            
            y = np.mean(data, axis=1)
            e = np.std(data, axis=1) / np.sqrt(data.shape[1])

            time_to_react.append(y)
            time_to_react_err.append(e)

        return time_to_react, time_to_react_err

    def plot_time_to_react(self):

        colors = plt.cm.Spectral(np.linspace(0, 1, self.interaction_range.size))

        plt.figure()

        for i, interaction in enumerate(self.interaction_range):

            sim_dir = self.get_dir_reaction(interaction)

            data = np.genfromtxt(sim_dir / f'reaction_tempo.csv')
            
            y = np.mean(data, axis=1)
            e = np.std(data, axis=1) / np.sqrt(data.shape[1])

            x = self.beta_range[:y.size]

            # print(x.shape)
            # print(y.shape)

            plt.plot(x, y, color=colors[i])
            plt.fill_between(x, y-e, y+e, alpha=0.2, edgecolor=colors[i], facecolor=colors[i])
            plt.xlabel("beta ~ organization")
            plt.ylabel("time")
            # plt.xlim((0,4))
            # plt.ylim((0,30000))

        plt.title("reaction time mean +- std error of mean")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral, norm=matplotlib.colors.Normalize(vmin=min(self.interaction_range), vmax=max(self.interaction_range)))
        cbar = plt.colorbar(sm)
        cbar.ax.set_ylabel("interaction")

        plt.figure()

        for i, interaction in enumerate(self.interaction_range):

            sim_dir = self.get_dir_reaction(interaction)

            data = np.genfromtxt(sim_dir / f'reaction_tempo.csv')

            y = np.std(data, axis=1)
            x = self.beta_range[:y.size]

            plt.plot(x, y, color=colors[i])
            plt.xlabel("beta ~ organization")
            plt.ylabel("std time")
            # plt.xlim((0,4))

        plt.title("reaction time standard deviation")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral, norm=matplotlib.colors.Normalize(vmin=min(self.interaction_range), vmax=max(self.interaction_range)))
        cbar = plt.colorbar(sm)
        cbar.ax.set_ylabel("interaction")

    @property
    def condensation_energy(self):

        sim_dir = self.get_dir_condensation()

        E = np.zeros(self.beta_range.size)

        data = np.genfromtxt(sim_dir / f'cond_energy.csv')

        for j, beta in enumerate(self.beta_range):

            E[j] = np.mean(data[j,:])
            
        return E

    def plot_condensation_convergence(self):

        colors = plt.cm.Spectral(np.linspace(0, 1, self.beta_range.size))

        sim_dir = self.get_dir_condensation()

        plt.figure()

        E = np.zeros(self.beta_range.size)

        data = np.genfromtxt(sim_dir / f'cond_energy.csv')
        #print(data.shape)

        for j, beta in enumerate(self.beta_range):

            E[j] = np.mean(data[j,:])

            plt.plot(data[j,:], color=colors[j])
            
        plt.xlabel("step")
        plt.ylabel("energy")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral, norm=matplotlib.colors.Normalize(vmin=min(self.beta_range), vmax=max(self.beta_range)))
        cbar = plt.colorbar(sm)
        cbar.ax.set_ylabel("beta")

        plt.figure()
        plt.plot(self.beta_range, E)
        plt.xlabel("beta")
        plt.ylabel("mean energy")

    def plot_condensation(self):

        for j, beta in enumerate(self.beta_range):
            plt.figure()
            ax = plt.gca()
            self.plot_lattice(beta, ax)
            plt.title(f"beta = {beta}")

    def plot_lattice(self, beta, ax):

        sim_dir = self.get_dir_condensation()
        data = np.genfromtxt(sim_dir / f'lattice_{beta:.3f}.csv')
        data[data == 0] = np.NaN

        ax.imshow(data, cmap=plt.cm.Spectral, interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_reaction_flux(self, interaction, product):
        
        sim_dir = self.get_dir_reaction(interaction)

        for j, beta in enumerate(self.beta_range):

            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

            data = np.genfromtxt(sim_dir / f'lattice_{beta:.3f}.csv')

            data[data == 0] = np.NaN
            #data[data != product] = 0

            ax1.imshow(data, cmap=plt.cm.Spectral, interpolation='none', resample=False)
            # ax1.title(f"{beta}")

            data1 = np.genfromtxt(sim_dir / f'l_react_{beta:.3f}_{product}.csv')
            data2 = np.genfromtxt(sim_dir / f'l_react_{beta:.3f}_{product+1}.csv')

            data = data1
            # data = data2 - data1

            #data = data - np.median(data)
            #m = max([data.max(), -data.min()])
            d = data.copy();
            data[data == 0] = np.NaN

            ax2.imshow(data, 
                #alpha=0.5+0.5*d/d.max(),
                cmap=plt.cm.magma_r, interpolation='none', resample=False)#, vmin=-m, vmax=m)
            # ax2.title(f"{beta}")

    def condensation_temperature(self, plotflag=False, plot_kwargs={}):

        E = self.condensation_energy
        E -= E[0]

        def func(x, a, b, c, d, e):
            return a*x + d*(0.5 + np.arctan(b*(x-c))/np.pi) + e

        xdata = self.beta_range
        ydata = E / min(E)

        popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata,
            p0=(0.0, 3, 3, 0.5, 0),
            bounds=((-1, 0, 0, -10, -1), (1, 10, 10, 2, 1),))

        p = " ".join([f"{p:.2f}" for p in popt])

        inflection_point = popt[2]
        beta_critical = inflection_point

        if plotflag is True:
            plt.plot(xdata, ydata, **plot_kwargs)
            plt.plot(xdata, func(xdata, *popt), 
                ':', label=p, **plot_kwargs)
            plt.plot(beta_critical, func(beta_critical, *popt), 
                'o', **plot_kwargs)
            plt.legend()

        return beta_critical