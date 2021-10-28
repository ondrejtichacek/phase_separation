import os
import shutil
import numpy as np

# srcdir = '/home/ondrej/repo/phase_separation/turbidity/'
srcdir = '/home1/tichacek/phase_separation/turbidity/'

tmpdir = '/dev/shm/turbidity/'

try:
    shutil.rmtree(tmpdir)
except FileNotFoundError as E:
    pass

shutil.copytree(srcdir, tmpdir)

wrkdir = tmpdir

os.chdir(wrkdir)

from optim import Optimizer

params = {
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

for mol in ["atp", "adp", "amp"]:
    print(mol)

    opt = Optimizer(
        sel = 'atp',
        vol_frac_scaling_x = 10,
        vol_frac_scaling_y = 10,
    )
    
    opt.compile()
    opt.optimize(params, maxiter=1000)
    
    best_x = opt.result.x
    print(best_x)

