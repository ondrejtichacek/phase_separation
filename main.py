# conda create -n phase-sep
# conda activate phase-sep
# conda install -c conda-forge numpy matplotlib ipykernel

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed

import subprocess
import shutil

class LatticePhaseReact():
    def __init__(
        self,
        interaction_range,
        volume_fraction,
        beta_range,
        wrkdir=None):

        self.interaction_range = interaction_range
        self.volume_fraction = volume_fraction
        self.beta_range = beta_range

        if wrkdir is None:
            wrkdir = os.getcwd()
        
        self.wrkdir = Path(wrkdir)
        self.res_dir = self.wrkdir / "results"

        # shutil.rmtree(self.res_dir)
        self.res_dir.mkdir(exist_ok=True)

        self.make()

    def make(self):
        os.chdir(self.wrkdir)

        CFLAGS=[
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

        out = subprocess.run(
            ["g++"] + CFLAGS + ["react_phase_sep.cpp"],
            capture_output=True)
            
        if out.returncode > 0:
            print(out)

    def simulate_condensation(self, num_sim_cond=1000):

        sim_dir = self.res_dir / f'condensation'
        
        if sim_dir.exists():
            shutil.rmtree(sim_dir)
        sim_dir.mkdir(exist_ok=True)
        
        os.chdir(sim_dir)

        out = subprocess.run(["../../a.out",
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

    def perform_reaction_simulation(self, interaction, num_sim_react):

        cond_sim_dir = self.res_dir / f'condensation'
        sim_dir = self.res_dir / f'interaction_{interaction:.1f}'
        
        if sim_dir.exists():
            shutil.rmtree(sim_dir)
        sim_dir.mkdir(exist_ok=True)
        
        shutil.copytree(cond_sim_dir, sim_dir, dirs_exist_ok=True)

        os.chdir(sim_dir)

        out = subprocess.run(["../../a.out",
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

        if parallel is False:
            for interaction in self.interaction_range:
                self.perform_reaction_simulation(interaction, num_sim_react)
        else:
            pfun = lambda x: self.perform_reaction_simulation(x, num_sim_react)
            Parallel(n_jobs=6) (delayed(pfun)(ir) for ir in self.interaction_range)

    def plot_interaction_matrices(self):

        plt.figure()
        sim_dir = self.res_dir / f'condensation'

        J = np.genfromtxt(sim_dir / f'J.csv')

        #J[J == 0] = np.NaN
        plt.imshow(J, cmap=plt.cm.Spectral)
        plt.title(f"J -- interaction")
        plt.xlabel(f"Component (enzyme)")
        plt.ylabel(f"Component (enzyme)")
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("interaction strength")

        plt.figure()

        I = np.genfromtxt(sim_dir / f'I.csv')

        #I[I == 0] = np.NaN
        plt.imshow(I, cmap=plt.cm.Spectral)
        plt.title(f"I -- interaction")
        plt.xlabel(f"Component (enzyme)")
        plt.ylabel(f"Solute (reactant)")
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("interaction strength")

    def plot_time_to_react(self):

        colors = plt.cm.Spectral(np.linspace(0, 1, self.interaction_range.size))

        plt.figure()

        for i, interaction in enumerate(self.interaction_range):

            sim_dir = self.res_dir / f'interaction_{interaction:.1f}'

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
            plt.ylim((0,30000))

        plt.title("reaction time mean +- std error of mean")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral, norm=matplotlib.colors.Normalize(vmin=min(self.interaction_range), vmax=max(self.interaction_range)))
        cbar = plt.colorbar(sm)
        cbar.ax.set_ylabel("interaction")

        plt.figure()

        for i, interaction in enumerate(self.interaction_range):

            sim_dir = self.res_dir / f'interaction_{interaction:.1f}'

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

    def plot_condensation_convergence(self):

        colors = plt.cm.Spectral(np.linspace(0, 1, self.beta_range.size))

        sim_dir = self.res_dir / f'condensation'

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
        sim_dir = self.res_dir / f'condensation'

        for j, beta in enumerate(self.beta_range):

            plt.figure()

            data = np.genfromtxt(sim_dir / f'lattice_{beta:.1f}.csv')

            data[data == 0] = np.NaN

            ind = (int(np.floor(j/2)), np.remainder(j,2))

            plt.imshow(data, cmap=plt.cm.Spectral)
            plt.title(f"{beta}")
            #axs[ind].title(f"beta = {beta}, int = {interaction}")

            #plt.plot(x, y, color=colors[i])
            #plt.fill_between(x, y-e, y+e, alpha=0.2, edgecolor=colors[i], facecolor=colors[i])
            #plt.xlabel("beta ~ organization")
            #plt.ylabel("time")

        #plt.title("reaction time mean +- std error of mean")