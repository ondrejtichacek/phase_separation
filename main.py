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
import functools
import scipy.optimize
import hashlib
import pprint

import subprocess
import shutil

def test_parameter_product():
    PAR = {
        'num_components': np.array([2,4,6]),
        'chemical_potential': np.linspace(0.5, 3, 5),
    }
    params, ind = parameter_product(**PAR)

    res = simulator(mock_simulate, params)

    for i, n in enumerate(PAR['num_components']):
        for j, c in enumerate(PAR['chemical_potential']):
            assert n == res[ind(i,j)]['num_components']
            assert c == res[ind(i,j)]['chemical_potential']

def parameter_product(**kwargs):
    for V in kwargs.values():
        assert len(V) > 0
    
    dims = [V.size for V in kwargs.values()]
    #linear_index = lambda *multi_index: np.ravel_multi_index(multi_index, dims)

    params = []
    for V in itertools.product(*kwargs.values()):
        kw = {}
        for i, key in enumerate(kwargs.keys()):
            kw[key] = V[i]
        params.append(kw)

    return params, dims

def mock_simulate(**kwargs):
    s = f"sim with args {kwargs}"
    print(s)
    return kwargs

def simulator(fun, params, dims=None, parallel=True):
    res = []
    if parallel is True:
        res = Parallel(n_jobs=6, verbose=10) (delayed(fun)(**p) for p in params)
    else:
        for p in params:
            res.append(fun(**p))

    if dims is not None:
        res = np.reshape(res, dims)
    else:
        res = np.asarray(res)

    return res

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

def mu_mean_field(volume_fraction, beta):
    x = volume_fraction
    J = 1 # mean value
    mu = 1/beta * np.log((1-x)/x) + 4 * J * x
    return mu

class LatticePhaseReact():
    def __init__(
        self,
        
        interaction_range=np.linspace(0, 1, 6),
        num_components=20,
        lattice_size=100,
        volume_fraction=0.3,
        
        nt=100,
        chemical_potential=1.6,
        chemical_potential_pathway=None,
        alpha=1,
        alpha_pathway=None,
        beta=1,
        beta_pathway=None,
        
        extended_neighborhood=False,
        log_concentration=False,

        resdir=None,
        wrkdir=None):

        self.interaction_range = interaction_range
        self.volume_fraction = volume_fraction

        if alpha_pathway is None:
            self.alpha_pathway = np.full(nt, alpha)
        else:
            assert alpha_pathway.size == nt
            self.alpha_pathway = np.copy(alpha_pathway)

        if beta_pathway is None:
            self.beta_pathway = np.full(nt, beta)
        else:
            assert beta_pathway.size == nt
            self.beta_pathway = np.copy(beta_pathway)

        self.beta_pathway.flags.writeable= False

        if chemical_potential_pathway is None:
            self.chemical_potential_pathway = np.full(nt, chemical_potential)
        else:
            assert chemical_potential_pathway.size == nt
            self.chemical_potential_pathway = np.copy(chemical_potential_pathway)

        self.chemical_potential_pathway.flags.writeable = False

        # if beta_pathway is None:
        #     bc_estimate = beta_critical(self.volume_fraction)
        #     beta_pathway = np.linspace(0, 2*bc_estimate, beta_num)

        self.num_components = num_components
        self.lattice_size = lattice_size
        self.extended_neighborhood = extended_neighborhood

        self.log_concentration = log_concentration

        self.condensation_interaction_version = 'default'
        self.reaction_interaction_version = 'default'

        if wrkdir is None:
            wrkdir = os.getcwd()
        self.wrkdir = Path(wrkdir)

        if resdir is None:
            self.res_dir = self.wrkdir / "results"
        else:
            self.res_dir = Path(resdir) / "results"

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

    def get_patwhay_hash(self):

        hashfun = lambda a: hashlib.blake2b(a, digest_size=8, usedforsecurity=False)
        
        s = ""
        s += hashfun(self.beta_pathway.tostring()).hexdigest()
        s += hashfun(self.chemical_potential_pathway.tostring()).hexdigest()

        return s

    def get_char_string(self, part):
        s = ""
        for (p, n) in self.get_defining_parameters(part):
            s += f"{n}_{getattr(self, p)}"

        return s

    def get_dir_condensation(self):
        return (self.res_dir 
                / (f'condensation_' + self.get_char_string('condensation'))
                / self.get_patwhay_hash())

    def get_dir_reaction(self, interaction):
        return (self.res_dir
                / (f'reaction_' + f'interaction_{interaction:.8f}_' + self.get_char_string('reaction'))
                / self.get_patwhay_hash())

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

        if out.returncode != 0:
            print(out.stdout.decode("utf-8"))
            print(out.stderr.decode("utf-8"))
            raise(ValueError("Something went wrong"))

    def simulate_condensation(self, num_sim_cond=1000):

        sim_dir = self.get_dir_condensation()
        
        if sim_dir.exists():
            shutil.rmtree(sim_dir)
        sim_dir.mkdir(exist_ok=True, parents=True)
        
        os.chdir(sim_dir)
        print(sim_dir)

        self.save_condensate_interaction_matrix()

        self.make(wrkdir=sim_dir)

        self.save_sweep()

        cmd = ["./a.out",
            f"0",
            f"{self.volume_fraction}",
            f"{num_sim_cond}",
            f"0",]

        print(" ".join(cmd))

        out = subprocess.run(cmd,
            capture_output=True)

        if out.returncode != 0:
            print(out.stdout.decode("utf-8"))
            print(out.stderr.decode("utf-8"))
            raise(ValueError("Something went wrong"))

    def perform_reaction_simulation(self, interaction, num_sim_react):

        cond_sim_dir = self.get_dir_condensation()
        sim_dir = self.get_dir_reaction(interaction)
        
        if sim_dir.exists():
            shutil.rmtree(sim_dir)
        sim_dir.mkdir(exist_ok=True, parents=True)
        
        shutil.copytree(cond_sim_dir, sim_dir, dirs_exist_ok=True)

        os.chdir(sim_dir)

        out = subprocess.run(["./a.out",
            f"{interaction}",
            f"{self.volume_fraction}",
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
            Parallel(n_jobs=6, verbose=10) (delayed(pfun)(ir) for ir in self.interaction_range)

    def save_sweep(self):
        sim_dir = self.get_dir_condensation()

        np.savetxt(sim_dir / "sweep_alpha.csv", self.alpha_pathway, fmt='%f')
        np.savetxt(sim_dir / "sweep_beta.csv", self.beta_pathway, fmt='%f')
        np.savetxt(sim_dir / "sweep_mu.csv", self.chemical_potential_pathway, fmt='%f')

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

        elif ver == "rand3":
            I = np.random.uniform(r[0]-s[0], r[0]+s[0], size=(q+1, q+1))
            
            I[0,:] = np.random.uniform(r[1]-s[1], r[1]+s[1], size=(q+1))
            I[:,0] = np.random.uniform(r[1]-s[1], r[1]+s[1], size=(q+1))

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
        # I = self.I

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

            x = self.beta_pathway[:y.size]

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
            x = self.beta_pathway[:y.size]

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

        E = np.zeros(self.beta_pathway.size)

        data = np.genfromtxt(sim_dir / f'cond_energy.csv')

        for j, beta in enumerate(self.beta_pathway):

            E[j] = np.mean(data[j,:])
            
        return E

    def plot_condensation_convergence(self):

        colors = plt.cm.Spectral(np.linspace(0, 1, self.beta_pathway.size))

        sim_dir = self.get_dir_condensation()

        plt.figure()

        E = np.zeros(self.beta_pathway.size)

        data = np.genfromtxt(sim_dir / f'cond_energy.csv')
        #print(data.shape)

        for j, beta in enumerate(self.beta_pathway):

            E[j] = np.mean(data[j,:])

            plt.plot(data[j,:], color=colors[j])
            
        plt.xlabel("step")
        plt.ylabel("energy")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral, norm=matplotlib.colors.Normalize(vmin=min(self.beta_pathway), vmax=max(self.beta_pathway)))
        cbar = plt.colorbar(sm)
        cbar.ax.set_ylabel("beta")

        plt.figure()
        plt.plot(self.beta_pathway, E)
        plt.xlabel("beta")
        plt.ylabel("mean energy")

    def plot_condensation(self):

        for j, beta in enumerate(self.beta_pathway):
            plt.figure()
            ax = plt.gca()
            self.plot_lattice(beta, ax)
            plt.title(f"beta = {beta}")

    def get_lattice(self, t_index):
        sim_dir = self.get_dir_condensation()
        f_lattice = sim_dir / f'lattice_{t_index:d}.csv'
        try:
            data = np.genfromtxt(f_lattice)
        except FileNotFoundError as E:
           print(f_lattice)
           raise(E)
        return data

    def plot_lattice(self, beta, ax):

        data = self.get_lattice(beta)
        data[data == 0] = np.NaN

        ax.imshow(data, cmap=plt.cm.Spectral, interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])

    @functools.cache
    def get_substrate(self, product, t_index):
        sim_dir = self.get_dir_condensation()

        data = np.genfromtxt(sim_dir / f'substrate_{t_index:d}_{product}')

        return data

    @functools.cache
    def get_reaction_flux(self, interaction, product, beta):
        
        sim_dir = self.get_dir_reaction(interaction)

        data = np.genfromtxt(sim_dir / f'l_react_{beta:.8f}_{product}.csv')

        # data = data - np.median(data)
        # m = max([data.max(), -data.min()])

        return data

    def plot_reaction_flux(self, interaction, product, beta_pathway=None):
        
        sim_dir = self.get_dir_reaction(interaction)

        if beta_pathway is None:
            beta_pathway = self.beta_pathway

        for j, beta in enumerate(beta_pathway):

            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

            f_lattice = sim_dir / f'lattice_{j:d}.csv'
            try:
                data = np.genfromtxt(f_lattice)
            except FileNotFoundError as E:
                print(f_lattice)
                raise(E)

            data[data == 0] = np.NaN
            #data[data != product] = 0

            ax1.imshow(data, cmap=plt.cm.Spectral, interpolation='none', resample=False)
            # ax1.title(f"{beta}")

            data1 = self.get_reaction_flux(interaction, product, beta)
            data2 = self.get_reaction_flux(interaction, product + 1, beta)

            data = data1
            # data = data2 - data1

            #data = data - np.median(data)
            #m = max([data.max(), -data.min()])
            d = data.copy()
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

        xdata = self.beta_pathway
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
    
def beta_critical(x):
    X, beta = beta_critical_fun()
    ind = np.where(x < X)[0]
    ind = ind[0:2]
    beta_avg = np.mean(beta[ind])

    return beta_avg
    

def beta_critical_fun():
    """
    x/(1-x) = w*(w-1)/(exp(beta) - w)
    
    where w = (-b +- sqrt(b^2 - 4*exp(beta)))/2
    where b = exp(beta) * (1-1/k) + 1 + 1/k
    where k = 3
    """
    N = 100000

    beta_pathway = np.logspace(0, 1, num=N, base=10)
    # print(beta_pathway)

    k = 3

    x = np.full((N,2), np.NaN)

    for i, beta in enumerate(beta_pathway):
        c = np.exp(beta)
        b = c * (1 - 1/k) + 1 + 1/k

        if b*b - 4*c > 0:
            wp = (b + np.sqrt(b*b - 4*c))/2
            wm = (b - np.sqrt(b*b - 4*c))/2
    
            w = wm
            g = w*(w-1)/(c - w)

            x[i,0] = g / (1 + g)

            w = wp
            g = w*(w-1)/(c - w)

            x[i,1] = g / (1 + g)

        # x = w * (w-1) / (w**2 - np.exp(beta))

    X = np.concatenate([x[:,0], x[:,1]])
    beta = np.concatenate([beta_pathway, beta_pathway])

    ind = np.isnan(X) == False
    X = X[ind]
    beta = beta[ind]

    ind = np.argsort(X)
    X = X[ind]
    beta = beta[ind]

    return X, beta

colorscale = [
        # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
        [0, "rgb(255, 255, 255)"],
        [0.00001, "rgb(255, 255, 255)"],

        # Let values between 10-20% of the min and max of z
        # have color rgb(20, 20, 20)
        [0.00002, "rgb(20, 20, 20)"],
        [0.1, "rgb(20, 20, 20)"],

        # Values between 20-30% of the min and max of z
        # have color rgb(40, 40, 40)
        [0.2, "rgb(40, 40, 40)"],
        [0.3, "rgb(40, 40, 40)"],

        [0.3, "rgb(60, 60, 60)"],
        [0.4, "rgb(60, 60, 60)"],

        [0.4, "rgb(80, 80, 80)"],
        [0.5, "rgb(80, 80, 80)"],

        [0.5, "rgb(100, 100, 100)"],
        [0.6, "rgb(100, 100, 100)"],

        [0.6, "rgb(120, 120, 120)"],
        [0.7, "rgb(120, 120, 120)"],

        [0.7, "rgb(140, 140, 140)"],
        [0.8, "rgb(140, 140, 140)"],

        [0.8, "rgb(160, 160, 160)"],
        [0.9, "rgb(160, 160, 160)"],

        [0.9, "rgb(180, 180, 180)"],
        [1.0, "rgb(180, 180, 180)"]
    ]