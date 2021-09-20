#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# conda create -n phase-sep
# conda activate phase-sep
# conda install -c conda-forge numpy matplotlib ipykernel

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from main import LatticePhaseReact

import subprocess
import shutil

matplotlib.rcParams['figure.figsize'] = (14,8)


# In[ ]:


interaction_range = np.linspace(0, 1.4, 6)
# interaction_range = np.linspace(0, 0, 1)
volume_fraction = 0.3

beta_range = np.linspace(0, 3, 30+1)#[1:]

wrkdir = "/home/ondrej/repo/phase_separation"

PR = LatticePhaseReact(
    interaction_range=interaction_range,
    volume_fraction=volume_fraction,
    beta_range=beta_range,
    wrkdir=wrkdir)


# In[ ]:


# CONDENSATION
PR.simulate_condensation(num_sim_cond=1000)


# In[ ]:


# REACTION
PR.simulate_reaction(num_sim_react=1000, parallel=True)


# In[ ]:


matplotlib.rcParams['figure.figsize'] = (7,4)
PR.plot_interaction_matrices()


# In[ ]:


matplotlib.rcParams['figure.figsize'] = (7,4)
PR.plot_condensation_convergence()


# In[ ]:


matplotlib.rcParams['figure.figsize'] = (14,8)
PR.plot_time_to_react()


# In[ ]:


PR.plot_condensation()

