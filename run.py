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



from pyemma.coordinates.data.featurization.misc import GroupCOMFeature
c1 = GroupCOMFeature(feat.topology, [list(range(10,367))])
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




atoms=[]
for number in range(344,367):
    atoms.append(number)
    
def contacts(traj: traj):

    cutoff=0.3
    j=md.compute_neighbors(traj,cutoff=cutoff,query_indices=atoms)
    ls=[]
    for arr in j:
        lst=arr.tolist()
        sort = [x for x in lst if x>628 and x<36617]
        ls.append(len(sort))
    
    arr=np.array(ls)
    reshaped=(arr.reshape(-1,1))
    print(np.shape(reshaped))
    j=np.absolute(reshaped)
    k=j.astype('float32')
#     dt=j.dtype(np.float32)
    return k


#### TYR

atoms_tyr=[]
for number in range(604,627):
    atoms_tyr.append(number)

def contacts_tyr(traj: traj):

    cutoff=0.35
    j=md.compute_neighbors(traj,cutoff=cutoff,query_indices=atoms_tyr)
    ls=[]
    for arr in j:
        lst=arr.tolist()
        sort = [x for x in lst if x>628 and x<36617]
        ls.append(len(sort))
    
    arr=np.array(ls)
    reshaped=(arr.reshape(-1,1))
    print(np.shape(reshaped))
    j=np.absolute(reshaped)
    k=j.astype('float32')
#     dt=j.dtype(np.float32)
    return k


#### Phe

atoms_phe=[]
for number in range(1,6):
    atoms_phe.append(number)

def contacts_phe(traj: traj):

    cutoff=0.35
    j=md.compute_neighbors(traj,cutoff=cutoff,query_indices=atoms_phe)
    ls=[]
    for arr in j:
        lst=arr.tolist()
        sort = [x for x in lst if x>628 and x<36617]
        ls.append(len(sort))
    
    arr=np.array(ls)
    reshaped=(arr.reshape(-1,1))
    print(np.shape(reshaped))
    j=np.absolute(reshaped)
    k=j.astype('float32')

    return k

######### same but for water


atoms_sol=[]
for number in range(334,341):
    atoms_sol.append(number)

def contacts_sol(traj: traj):

    cutoff=0.35
    j=md.compute_neighbors(traj,cutoff=cutoff,query_indices=atoms_sol)
    ls=[]
    for arr in j:
        lst=arr.tolist()
        sort = [x for x in lst if x>108516 and x<234790]
        ls.append(len(sort))
    
    arr=np.array(ls)
    reshaped=(arr.reshape(-1,1))
    print(np.shape(reshaped))
    j=np.absolute(reshaped)
    k=j.astype('float32')
#     dt=j.dtype(np.float32)
    return k
    














### compute distance between 11 and 31
from pyemma.coordinates.data.featurization.misc import GroupCOMFeature
c_1 = GroupCOMFeature(feat.topology, [list(range(157,180))])
c_2 = GroupCOMFeature(feat.topology, [list(range(496,512))])
# c3 = GroupCOMFeature(feat.topology, [list(range(90,468))])

def feature_function_trp_gln(traj: traj):
    centers1 = c_1.transform(traj)  # yields ndarray
    centers2 = c_2.transform(traj)  # yields ndarray
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
    
### compute distance between 11 and 27
c_1a = GroupCOMFeature(feat.topology, [list(range(157,180))])
c_2a = GroupCOMFeature(feat.topology, [list(range(437,443))])


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
feat.add_custom_func(feature_function_com, dim=1, description='com_z')
feat.add_custom_func(contacts, dim=1, description='cont')
feat.add_custom_func(contacts_sol, dim=1, description='cont_sol')
feat.add_custom_func(contacts_tyr, dim=1, description='cont_tyr')
feat.add_custom_func(contacts_phe, dim=1, description='cont_phe')
feat.add_custom_func(feature_function_trp_gln, dim=1, description='cont_trp_gln')
feat.add_custom_func(feature_function_trp_gly, dim=1, description='cont_trp_gly')



# reader = pyemma.coordinates.source(['whole1.xtc','whole2.xtc','whole3.xtc','whole4.xtc','whole5.xtc','whole6.xtc','whole7.xtc','whole8.xtc'], features=feat)

# reader = pyemma.coordinates.source(['1_old.xtc','2_old.xtc','3_old.xtc'],features=feat)
reader = pyemma.coordinates.source(['tot2.xtc'],features=feat)
# reader = pyemma.coordinates.source(['whole1.xtc','whole2.xtc'],features=feat)

tica = pyemma.coordinates.tica(reader, lag=5)
tica_output = tica.get_output()
tica_concatenated = np.concatenate(tica_output)
print(tica_concatenated)

for i in range(len(tica_output)):
    with open(f'feat_output_{i:03d}.npy','wb') as handle:            # {i:03d} turns i into a 3 digit number starting with 0s
        np.save(handle, tica_output[i])

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

### Plot the individual IC's. This number will change depending on the input
fig, axes = plt.subplots(6, 1, figsize=(12, 5), sharex=True)
x = 0.1 * np.arange(tica_output[0].shape[0])
for i, (ax, tic) in enumerate(zip(axes.flat, tica_output[0].T)):
    ax.plot(x, tic)
    ax.set_ylabel('IC {}'.format(i + 1))
axes[-1].set_xlabel('time / ns')
fig.tight_layout()
plt.savefig('ics.png')


tica.write_to_hdf5('tica_trajectories_test.h5')




### cluster everything. This is very tunable

cluster = pyemma.coordinates.cluster_kmeans(
    tica_concatenated, k=200, max_iter=100, stride=1, fixed_seed=1)
dtrajs_concatenated = np.concatenate(cluster.dtrajs)
cluster.save('cluster.h5', save_streaming_chain=True, overwrite=True)

### Show fig
fig, ax = plt.subplots(figsize=(4, 4))
pyemma.plots.plot_density(
    *tica_concatenated[:, :2].T, ax=ax, cbar=False, alpha=0.3)
ax.scatter(*cluster.clustercenters[:, :2].T, s=5, c='C1')
ax.set_xlabel('IC 1')
ax.set_ylabel('IC 2')
fig.tight_layout()
plt.savefig('cluster.png')


#### look at msm timescales

msm = pyemma.msm.bayesian_markov_model(cluster.dtrajs, lag=40, dt_traj='0.2 ns')
print('fraction of states used = {:.2f}'.format(msm.active_state_fraction))
print('fraction of counts used = {:.2f}'.format(msm.active_count_fraction))

msm.save('msm.h5',overwrite=True)

### Estimate implied timescales
its = pyemma.msm.its(cluster.dtrajs, lags=100, nits=10, errors='bayes')
pyemma.plots.plot_implied_timescales(its, outfile='its.png', units='ns', dt=0.1);


### plot the spectral analysis. This should be done before so we know the number of states?
### need to come back to this

def its_separation_err(ts, ts_err):
    """
    Error propagation from ITS standard deviation to timescale separation.
    """
    return ts[:-1] / ts[1:] * np.sqrt(
        (ts_err[:-1] / ts[:-1])**2 + (ts_err[1:] / ts[1:])**2)


nits = 10

timescales_mean = msm.sample_mean('timescales', k=nits)
timescales_std = msm.sample_std('timescales', k=nits)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].errorbar(
    range(1, nits + 1),
    timescales_mean,
    yerr=timescales_std,
    fmt='.', markersize=10)
axes[1].errorbar(
    range(1, nits),
    timescales_mean[:-1] / timescales_mean[1:],
    yerr=its_separation_err(
        timescales_mean,
        timescales_std),
    fmt='.',
    markersize=10,
    color='C0')

for i, ax in enumerate(axes):
    ax.set_xticks(range(1, nits + 1))
    ax.grid(True, axis='x', linestyle=':')

axes[0].axhline(msm.lag * 0.1, lw=1.5, color='k')
axes[0].axhspan(0, msm.lag * 0.1, alpha=0.3, color='k')
axes[0].set_xlabel('implied timescale index')
axes[0].set_ylabel('implied timescales / ns')
axes[1].set_xticks(range(1, nits))
# axes[1].set_xticklabels(
#     ["{:d}/{:d}".format(k, k + 1) for k in range(1, nits + 2)],
#     rotation=45)
axes[1].set_xlabel('implied timescale indices')
axes[1].set_ylabel('timescale separation')
fig.tight_layout()
plt.savefig('spectral.png')



### CK test plotting and testing. This is very tunable

nstates = 4
cktest = msm.cktest(nstates, mlags=6)
pyemma.plots.plot_cktest(cktest, dt=0.1, units='ns');
plt.savefig('cktest.png')

### Macrostate analysis

msm.pcca(nstates)


fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
pyemma.plots.plot_contour(
    *tica_concatenated[:, :2].T,
    msm.pi[dtrajs_concatenated],
    ax=axes[0],
    mask=True,
    cbar_label='stationary distribution')
pyemma.plots.plot_free_energy(
    *tica_concatenated[:, :2].T,
    weights=np.concatenate(msm.trajectory_weights()),
    ax=axes[1],
    legacy=False)
for ax in axes.flat:
    ax.set_xlabel('IC 1')
axes[0].set_ylabel('IC 2')
axes[0].set_title('Stationary distribution', fontweight='bold')
axes[1].set_title('Reweighted free energy surface', fontweight='bold')
fig.tight_layout()
plt.savefig('macro_energy.png')

# fig, axes = plt.subplots(1, 4, figsize=(15, 3), sharex=True, sharey=True)
# for i, ax in enumerate(axes.flat):
#     pyemma.plots.plot_contour(
#         *tica_concatenated[:, :2].T,
#         msm.metastable_distributions[i][dtrajs_concatenated],
#         ax=ax,
#         cmap='afmhot_r',
#         mask=True,
#         cbar_label='metastable distribution {}'.format(i + 1))
#     ax.set_xlabel('IC 1')
# axes[0].set_ylabel('IC 2')
# fig.tight_layout()
# plt.savefig('macrodist.png')

### Get ths macrostates

metastable_traj = msm.metastable_assignments[dtrajs_concatenated]

fig, ax = plt.subplots(figsize=(5, 4))
_, _, misc = pyemma.plots.plot_state_map(
    *tica_concatenated[:, :2].T, metastable_traj, ax=ax)
ax.set_xlabel('IC 1')
ax.set_ylabel('IC 2')
misc['cbar'].set_ticklabels([r'$\mathcal{S}_%d$' % (i + 1)
                             for i in range(nstates)])
fig.tight_layout()
plt.savefig('macrostates.png')

### print out the mfpts

from itertools import product

mfpt = np.zeros((nstates, nstates))
for i, j in product(range(nstates), repeat=2):
    mfpt[i, j] = msm.mfpt(
        msm.metastable_sets[i],
        msm.metastable_sets[j])

from pandas import DataFrame
print('MFPT / ns:')
DataFrame(np.round(mfpt, decimals=2), index=range(1, nstates + 1), columns=range(1, nstates + 1))



### get the states in gro form for analysis
tr=['tot2.xtc']


pcca_samples = msm.sample_by_distributions(msm.metastable_distributions, 30)
torsions_source = pyemma.coordinates.source(tr, top='6.6_2.gro')
pyemma.coordinates.save_trajs(
    torsions_source,
    pcca_samples,
    outfiles=['./pcca{}_com.gro'.format(n + 1)
              for n in range(msm.n_metastable)])

### Print out the energies of the states

print('state\tπ\t\tG/kT')
for i, s in enumerate(msm.metastable_sets):
    p = msm.pi[s].sum()
    print('{}\t{:f}\t{:f}'.format(i + 1, p, -np.log(p)))
