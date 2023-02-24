
import numpy as np
import MDAnalysis as mda
import mdtraj as md
import pyemma
from pyemma.coordinates.data.featurization.misc import GroupCOMFeature


import featurize.featurize as feat
import featurize.caller as caller

# traj_files = ['traj_dat/w1.xtc', 'traj_dat/w2.xtc','traj_dat/w3.xtc']  
# top_file = 'traj_dat/10.gro'


def make_nlist():
    nlist = []
    for item in traj_files:
        nlist.append(md.load(item,top=top_file))
    print(nlist)
    return nlist

def load_trajectories(traj_files, top_file):
    """Load molecular dynamics simulation trajectories."""
    traj = md.load(traj_files, top=top_file)
    return traj

def extract_features(traj_files, top_file):
    """Extract features from trajectories."""
    tr = top_file
    feat_ = caller.feature(tr)
    reader = pyemma.coordinates.source(traj_files, features=feat_)
    return reader

def calculate_contacts(lipid_selection, prot_selections, universes):
    contact_analysis = feat.cont()
    ls1 = []
    for universe in universes:
        for prot_sel in prot_selections:
            result = contact_analysis.contacts_MDA(universe, universe.select_atoms(prot_sel), universe.select_atoms(lipid_selection))
            ls1.append(result)
    return ls1

def calulate_dists(top_file, bil,a,b,c,d,e,f,g,h,i,j,k,l,m,n, nlist):
    bil = GroupCOMFeature(top_file.topology,bil)
    a = GroupCOMFeature(top_file.topology, a)
    b = GroupCOMFeature(top_file.topology, b)
    c = GroupCOMFeature(top_file.topology, c)
    d = GroupCOMFeature(top_file.topology, d)
    e = GroupCOMFeature(top_file.topology, e)
    f = GroupCOMFeature(top_file.topology, f)
    g = GroupCOMFeature(top_file.topology, g)
    h = GroupCOMFeature(top_file.topology, h)
    i = GroupCOMFeature(top_file.topology, i)
    j = GroupCOMFeature(top_file.topology, j)
    k = GroupCOMFeature(top_file.topology, k)
    l = GroupCOMFeature(top_file.topology, l)
    m = GroupCOMFeature(top_file.topology, m)
    n = GroupCOMFeature(top_file.topology, n)
    group_features_dict = {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'g': g, 'h': h, 'i': i, 'j': j, 'k': k, 'l': l, 'm': m, 'n': n}
    d_bil = feat.dist()
    ls2 = []
    for traj in nlist:
        for name, feature in group_features_dict.items():
            ls2.append(d_bil.dist_bil(traj,bil, feature))
    return ls2


def get_inputs(m, splits, feat_len):
    processors = [caller.ReturnInputs() for i in range(len(splits))]
    inputs = [processor.return_inputs(m, i, feat_len)[0] for i, processor in enumerate(processors)]
    return inputs

def get_fubar_output(splits, feat_len):
    processor = caller.ChunkProcessor(splits, feat_len)
    return processor.fubar(splits, feat_len)

def combine_results(inputs1, inputs2,splits1, splits2, feat_len1, feat_len2):

    result_list = []
    for i in range(len(splits1)):
        # arr1 = data_output[i]
        # print(np.shape(arr1),'1')
        arr2 = inputs1[i]
        print(np.shape(arr2),'2')
        arr3 = inputs2[i]
        print(np.shape(arr3),'3')
        result = np.concatenate((arr2, arr3), axis=1)
        result_list.append(result)

    return result_list


def make_arrlist(input_cont, dims):
    arrs = [np.array(input_cont[i]).reshape(*dims[i]) for i in range(len(input_cont))]
    return arrs


def pyemma_reader(top, files):
    feat = pyemma.coordinates.featurizer(top) 
    feat.add_backbone_torsions() 
    reader = pyemma.coordinates.source(files, features=feat) 
    data_output = reader.get_output()
    return data_output





