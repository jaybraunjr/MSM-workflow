
import numpy as np
import MDAnalysis as mda
import mdtraj as md
import pyemma
from pyemma.coordinates.data.featurization.misc import GroupCOMFeature

import featurize.featurize as feat
import featurize.caller as caller

traj_files = ['traj_dat/w1.xtc', 'traj_dat/w2.xtc','traj_dat/w3.xtc']  
top_file = 'traj_dat/10.gro'


def make_nlist():
    nlist = []
    for item in traj_files:
        nlist.append(md.load(item,top=top_file))
    print(nlist)
    return nlist
def calculate_contacts(lipid_selection, prot_selections, universes):
    # Initialize contact analysis class
    contact_analysis = feat.cont()
    # Run analysis for each trajectory and each protein selection
    ls1 = []
    for universe in universes:
        for prot_sel in prot_selections:
            result = contact_analysis.contacts_MDA(universe, universe.select_atoms(prot_sel), universe.select_atoms(lipid_selection))
            ls1.append(result)
    # print(ls1,'mda')
    return ls1

def calulate_dists(top_file, bil, a ,b ,c,d,e,f,g,h,i,j, nlist):
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
#     k = GroupCOMFeature(top_file.topology, k)

    group_features_dict = {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'g': g, 'h': h, 'i': i, 'j': j}
    d_bil = feat.dist()
    ls2 = []
    for traj in nlist:
        for name, feature in group_features_dict.items():
            ls2.append(d_bil.dist_bil(traj,bil, feature))
    # print(ls1)
    return ls2




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

def reshaping(inp1,inp2, traj_chunk, feat_len1, feat_len2):
    splits1 = np.array_split(inp1, traj_chunk)
    splits2 = np.array_split(inp2, traj_chunk)
    processor_a = caller.ChunkProcessor(splits1, feat_len1)
    processor_b = caller.ChunkProcessor(splits2, feat_len2)
    m1 = processor_a.fubar(splits1,feat_len1)
    m2 = processor_b.fubar(splits2,feat_len2)
    compbined_array = np.concatenate((m1,m2), axis=1)
    list_ = compbined_array.tolist()
    processor1 = caller.ReturnInputs()
    processor2 = caller.ReturnInputs()
    processor3 = caller.ReturnInputs()
    in1 = processor1.return_inputs(list_,0,20)
    in2 = processor2.return_inputs(list_,1,20)
    in3 = processor3.return_inputs(list_,2,20)
    conts = in1+in2+in3
    # print(conts,'conts')
    return conts


def make_arrlist(input_cont, dims):
    arrs = [np.array(input_cont[i]).reshape(*dims[i]) for i in range(len(input_cont))]
    return arrs


def main():
    # Load trajectories
    a =[list(range(1,367))]
    bil =  [list(range(628,33617))]
    b =  [list(range(1,2))]
    c = [list(range(626,627))]
    d = [list(range(128,129))]
    e = [list(range(228,229))]
    f = [list(range(328,339))]
    g =  [list(range(428,429))]
    h = [list(range(528,529))]
    i = [list(range(600,601))]
    j =  [list(range(28,29))]


#     # tr = 'traj_dat/10.gro'
#     feat_ = caller.feature(top_file)
#     # traj = load_trajectories(traj_files, top_file)
#     # Extract features
#     # reader = extract_features(traj_files, top_file)
#     traj_files = ['traj_dat/w1.xtc', 'traj_dat/w2.xtc','traj_dat/w3.xtc'] 
#     nlist= make_nlist()
#     # Calculate distances
#     inp_2 = calulate_dists(feat_,bil,a ,b ,c,d,e,f,g,h,i,j,nlist)
#     print(np.shape(inp_2))
#     # Calculate contacts
#     prot_selections = [f"(resid {i}) and (not backbone)" for i in range(1, 11)]
#     lipid_selection = 'resname POPC DOPE SAPI'
#     universes = [mda.Universe('traj_dat/10.gro', traj) for traj in traj_files]
#     # universes = [mda.Universe(feat_, traj_files)]
#     inp_1 = calculate_contacts(lipid_selection, prot_selections, universes)
#     print(np.shape(inp_1))
#     # Reshape and combine
#     traj_chunk = 3
#     feat_len1 = 10
#     feat_len2 = 10
#     feat_len3 = 10
#     conts = reshaping(inp_1, inp_2, traj_chunk, feat_len1, feat_len2)
#     # Make array list
#     dims =  [(11, 20), (5, 20), (11, 20)]
#     arrlist = make_arrlist(conts, dims=dims)
#     # Save
#     caller.save_reader(arrlist, 'arrlist')


# if __name__ == '__main__':
#     main()





