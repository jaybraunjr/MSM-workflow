import matplotlib.pyplot as plt
# import matplotlib as mpl
import pyemma
from pyemma.util.contexts import settings
import mdtraj as md
import itertools
from itertools import combinations
import pandas as pd
import numpy as np
from pyemma.coordinates import source


data = pyemma.load('cluster.h5')
msm = pyemma.load('msm.h5')
tica_data = pyemma.coordinates.load('tica_trajectories_test.h5')
# tica_output = tica_data.get_output()
tica_concatenated = np.concatenate(tica_data)
                  
dtrajs_concatenated = np.concatenate(data.dtrajs)
print(dtrajs_concatenated)
nstates=4                 
msm.pcca(nstates)
                  
metastable_traj = msm.metastable_assignments[dtrajs_concatenated]

fig, ax = plt.subplots(figsize=(5, 4))
_, _, misc = pyemma.plots.plot_state_map(
    *tica_concatenated[:, :2].T, metastable_traj, ax=ax)
ax.set_xlabel('IC 1')
ax.set_ylabel('IC 2')
misc['cbar'].set_ticklabels([r'$\mathcal{S}_%d$' % (i + 1)
                             for i in range(nstates)])
fig.tight_layout()
plt.savefig('states.png')

print('state\tπ\t\tG/kT')
for i, s in enumerate(msm.metastable_sets):
    p = msm.pi[s].sum()
    print('{}\t{:f}\t{:f}'.format(i + 1, p, -np.log(p)))

tr=('1_old.xtc','2_old.xtc')

pcca_samples = msm.sample_by_distributions(msm.metastable_distributions, 10)
torsions_source = pyemma.coordinates.source(tr, top='6.6_2.gro')
pyemma.coordinates.save_trajs(
    torsions_source,
    pcca_samples,
    outfiles=['./pcca{}_com_tyr.gro'.format(n + 1)
              for n in range(msm.n_metastable)])