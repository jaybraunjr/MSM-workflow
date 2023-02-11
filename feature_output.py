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

    cen1 = centers1[:,-1]
    cen2 = centers2[:,-1]
    sub=(cen2-cen1)
    reshaped=(sub.reshape(-1,1))
    print(np.shape(reshaped))
    j=np.absolute(reshaped)
    k=j.astype('float32')

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



reader = pyemma.coordinates.source(['1_old.xtc','2_old.xtc','3_old.xtc','6_old.xtc','1.new.whole.xtc','3.new.whole.xtc','5.new.whole.xtc','9.new.whole.xtc','7_old.xtc','rep4.xtc','rep5_tot.xtc'], features=feat)

# reader = pyemma.coordinates.source(['1_old.xtc','2_old.xtc','3_old.xtc'],features=feat)
# reader = pyemma.coordinates.source(['whole1.xtc'],features=feat)
# reader = pyemma.coordinates.source(['whole1.xtc','whole2.xtc'],features=feat)


output = reader.get_output()


for i in range(len(output)):
    with open(f'reader_output_new_{i:03d}.npy','wb') as handle:            # {i:03d} turns i into a 3 digit number starting with 0s
        np.save(handle, output[i])
