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


#### Load tica output and whatnot

tica_data = pyemma.coordinates.load('tica_trajectories_4dim.h5')
# tica_output = tica_data.get_output()
tica_concatenated = np.concatenate(tica_data)
print(tica_concatenated)
print(np.shape(tica_concatenated))


cluster = pyemma.coordinates.cluster_kmeans(
    tica_concatenated, k=200, max_iter=200, stride=1, fixed_seed=1)
dtrajs_concatenated = np.concatenate(cluster.dtrajs)
cluster.save('cluster.h5', save_streaming_chain=True)

### Show fig
fig, ax = plt.subplots(figsize=(4, 4))
pyemma.plots.plot_density(
    *tica_concatenated[:, :4].T, ax=ax, cbar=False, alpha=0.3)
ax.scatter(*cluster.clustercenters[:, :4].T, s=5, c='C1')
ax.set_xlabel('IC 1')
ax.set_ylabel('IC 2')
fig.tight_layout()
plt.savefig('cluster.png')


### Estimate implied timescales
its = pyemma.msm.its(cluster.dtrajs, lags=100, nits=10, errors='bayes')
pyemma.plots.plot_implied_timescales(its, outfile='its.png', units='ns', dt=0.1);


### Lag dependent of previous tests. Have to choose carfully

msm = pyemma.msm.bayesian_markov_model(data.dtrajs, lag=40, dt_traj='0.2 ns')
print('fraction of states used = {:.2f}'.format(msm.active_state_fraction))
print('fraction of counts used = {:.2f}'.format(msm.active_count_fraction))



