#!/usr/bin/env python
# coding: utf-8

# In[182]:


# conda create -n phase-sep
# conda activate phase-sep
# conda install -c conda-forge numpy matplotlib ipykernel

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import matplotlib
from main import LatticePhaseReact


matplotlib.rcParams['figure.figsize'] = (14,8)


# In[183]:


interaction_range = np.linspace(0, 1.4, 6)
# interaction_range = np.linspace(0, 0, 1)
volume_fraction = 0.3

beta_range = np.linspace(0, 3, 30+1)#[1:]

wrkdir = "/home/ondrej/repo/phase_separation"

PR = LatticePhaseReact(
    interaction_range=interaction_range,
    volume_fraction=volume_fraction,
    beta_range=beta_range,
    num_components=6,
    lattice_size=100,
    wrkdir=wrkdir)


# In[184]:


# CONDENSATION
PR.generate_condensate_interaction('ones')
PR.simulate_condensation(num_sim_cond=1000)


# In[185]:


matplotlib.rcParams['figure.figsize'] = (7,4)
PR.plot_condensation_interaction_matrix()


# In[186]:


# REACTION
PR.generate_reaction_interaction('ones')
PR.simulate_reaction(num_sim_react=1000, parallel=True)


# In[187]:


matplotlib.rcParams['figure.figsize'] = (7,4)
PR.plot_reaction_interaction_matrix()


# In[188]:


matplotlib.rcParams['figure.figsize'] = (7,4)
PR.plot_condensation_convergence()


# In[189]:


matplotlib.rcParams['figure.figsize'] = (14,8)
PR.plot_time_to_react()


# In[190]:


matplotlib.rcParams['figure.figsize'] = (14,8)
PR.plot_condensation()

