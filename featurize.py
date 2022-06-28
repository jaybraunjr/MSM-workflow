
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

traj = md.load('whole1.xtc',top='6.6_2.gro')
feat = pyemma.coordinates.featurizer('6.6_2.gro')


### compute distance between 11 and 31
from pyemma.coordinates.data.featurization.misc import GroupCOMFeature
c_1 = GroupCOMFeature(feat.topology, [list(range(157,180))])
c_2 = GroupCOMFeature(feat.topology, [list(range(496,512))])


def feature_function_trp_gln(traj: traj):
    centers1 = c_1.transform(traj)  # yields ndarray
    centers2 = c_2.transform(traj)  # yields ndarray
#     print(centers1[:,-1])
    cen1 = centers1[:,-1]
    cen2 = centers2[:,-1]


    xyz = np.hstack((centers1, centers2))
    traj = md.Trajectory(xyz.reshape(-1, 2, 3), topology=None)
    # this has shape (n_frames, 1)
    print(traj)
    
    
    cont = md.compute_distances(traj,atom_pairs=[[0, 1]], periodic=False)
    print(np.shape(cont))
    return cont
    
### compute distance between 11 and 27
from pyemma.coordinates.data.featurization.misc import GroupCOMFeature
c_1a = GroupCOMFeature(feat.topology, [list(range(157,180))])
c_2a = GroupCOMFeature(feat.topology, [list(range(437,443))])
# c3 = GroupCOMFeature(feat.topology, [list(range(90,468))])

def feature_function_trp_gly(traj: traj):
    centers1 = c_1a.transform(traj)  # yields ndarray
    centers2 = c_2a.transform(traj)  # yields ndarray
#     print(centers1[:,-1])
    cen1 = centers1[:,-1]
    cen2 = centers2[:,-1]
#     centers3 = c3.transform(traj)  # yields ndarray

    xyz = np.hstack((centers1, centers2))
    traj = md.Trajectory(xyz.reshape(-1, 2, 3), topology=None)
    # this has shape (n_frames, 1)
    print(traj)
    
    
    cont = md.compute_distances(traj,atom_pairs=[[0, 1]], periodic=False)
    print(np.shape(cont))
    return cont

########### COM, accounting only for z-dim

from pyemma.coordinates.data.featurization.misc import GroupCOMFeature
c1 = GroupCOMFeature(feat.topology, [list(range(325,367))])
c2 = GroupCOMFeature(feat.topology, [list(range(628,33617))])
# c3 = GroupCOMFeature(feat.topology, [list(range(90,468))])

def feature_function_com(traj: traj):
    centers1 = c1.transform(traj)  # yields ndarray
    centers2 = c2.transform(traj)  # yields ndarray

#     print(centers1[:,-1])
    cen1 = centers1[:,-1]
    cen2 = centers2[:,-1]
    sub=(cen2-cen1)
    reshaped=(sub.reshape(-1,1))
    print(np.shape(reshaped))
    j=np.absolute(reshaped)
    k=j.astype('float32')
#     dt=j.dtype(np.float32)
    return k

### z-com of tyrosine on end (37)

c1a = GroupCOMFeature(feat.topology, [list(range(604,627))])
c2a = GroupCOMFeature(feat.topology, [list(range(628,33617))])
# c3 = GroupCOMFeature(feat.topology, [list(range(90,468))])

def feature_function_com2(traj: traj):
    centers1 = c1a.transform(traj)  # yields ndarray
    centers2 = c2a.transform(traj)  # yields ndarray

#     print(centers1[:,-1])
    cen1 = centers1[:,-1]
    cen2 = centers2[:,-1]
    sub=(cen2-cen1)
    reshaped=(sub.reshape(-1,1))
    print(np.shape(reshaped))
    j=np.absolute(reshaped)
    k=j.astype('float32')
#     dt=j.dtype(np.float32)
    return k


feat.add_custom_func(feature_function_trp_gln, dim=1, description='cont_trp_gln')
feat.add_custom_func(feature_function_trp_gly, dim=1, description='cont_trp_gly')
feat.add_custom_func(feature_function_com, dim=1, description='com_z')
feat.add_custom_func(feature_function_com2, dim=1, description='com_z2')


# reader = pyemma.coordinates.source(['whole1.xtc','whole2.xtc','whole3.xtc','whole4.xtc','whole5.xtc','whole6.xtc','whole7.xtc','whole8.xtc'], features=feat)

reader = pyemma.coordinates.source(['tot2.xtc'],features=feat)

tica = pyemma.coordinates.tica(reader, lag=5)
tica_output = tica.get_output()
tica_concatenated = np.concatenate(tica_output)
print(tica_concatenated)



### Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
pyemma.plots.plot_feature_histograms(
    tica_concatenated,
    ax=axes[0],
#     feature_labels=['IC1', 'IC2', 'IC3'],
    ylog=True, ignore_dim_warning=True)
pyemma.plots.plot_density(*tica_concatenated[:, :2].T, ax=axes[1], logscale=True)
axes[1].set_xlabel('IC 1')
axes[1].set_ylabel('IC 2')
fig.tight_layout()
plt.savefig('tica.png')

fig, axes = plt.subplots(6, 1, figsize=(12, 5), sharex=True)
x = 0.1 * np.arange(tica_output[0].shape[0])
for i, (ax, tic) in enumerate(zip(axes.flat, tica_output[0].T)):
    ax.plot(x, tic)
    ax.set_ylabel('IC {}'.format(i + 1))
axes[-1].set_xlabel('time / ns')
fig.tight_layout()
plt.savefig('ics.png')

tica.write_to_hdf5('tica_trajectories_4dim.h5')

