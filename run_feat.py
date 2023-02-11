import numpy as np
import MDAnalysis as mda
import mdtraj as md
import pyemma
from pyemma.coordinates.data.featurization.misc import GroupCOMFeature


import featurize.featurize as feat
import featurize.caller as caller

traj_list = ['traj_dat/w1.xtc', 'traj_dat/w2.xtc','traj_dat/w3.xtc'] 
top_file = 'traj_dat/10.gro'

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

# Load molecular dynamics simulation trajectories
traj = load_trajectories(traj_list, top_file)
# Extract features from trajectories
reader = extract_features(traj_list, top_file)

traj_list = ['traj_dat/w1.xtc', 'traj_dat/w2.xtc','traj_dat/w3.xtc'] 
n_list = []
ls = []
for item in traj_list:
    n_list.append(md.load(item, top='traj_dat/10.gro'))
    u_ = mda.Universe('traj_dat/10.gro', item)
    ls.append(u_)
    print(item)


universes = [mda.Universe('traj_dat/10.gro', traj) for traj in traj_list]

# Define lipid and protein selections
lipid_selection = '(resname POPC DOPE SAPI)'
prot_selections = [f"(resid {i}) and (not backbone)" for i in range(1, 11)]

# Initialize contact analysis class
contact_analysis = feat.cont()

# Run analysis for each trajectory and each protein selection
results = []
for universe in universes:
    for prot_sel in prot_selections:
        result = contact_analysis.contacts_MDA(universe, universe.select_atoms(prot_sel), universe.select_atoms(lipid_selection))
        results.append(result)

tr = 'traj_dat/10.gro'
feat_ = caller.feature(tr)


c1 = GroupCOMFeature(feat_.topology, [list(range(1,367))])
c1_ = GroupCOMFeature(feat_.topology, [list(range(628,33617))])
c2 = GroupCOMFeature(feat_.topology, [list(range(1,2))])
c3 = GroupCOMFeature(feat_.topology, [list(range(626,627))])
c4 = GroupCOMFeature(feat_.topology, [list(range(128,129))])
c5 = GroupCOMFeature(feat_.topology, [list(range(228,229))])
c6 = GroupCOMFeature(feat_.topology, [list(range(328,339))])
c7 = GroupCOMFeature(feat_.topology, [list(range(428,429))])
c8 = GroupCOMFeature(feat_.topology, [list(range(528,529))])
c9 = GroupCOMFeature(feat_.topology, [list(range(600,601))])
c10 = GroupCOMFeature(feat_.topology, [list(range(28,29))])

group_features_dict = {
    'c1': c1,
    'c2': c2,
    'c3': c3,
    'c4': c4,
    'c5': c5,
    'c6': c6,
    'c7': c7,
    'c8': c8,
    'c9': c9,
    'c10': c10,
    }

d_bil = feat.dist()

ls2 = []
def run_func2():
    for traj in n_list:
        for name, feature in group_features_dict.items():
            print('Working on feature:', name)
            ls2.append(d_bil.dist_bil(traj, c1_, feature))
    return ls2

l2 = run_func2()

splits1 = np.array_split(results, 3)
splits2 = np.array_split(l2, 3)

# here we are pushing to have the whole thing reshaped

processor1 = caller.ChunkProcessor(splits1, 10)
processor2 = caller.ChunkProcessor(splits2, 10)
m1 = processor1.fubar(splits1, 10)
m2 = processor2.fubar(splits2, 10)
combined_array = np.hstack((m1, m2))
list_ = combined_array.tolist()

caller.ReturnInputs

processor1 = caller.ReturnInputs()
processor2 = caller.ReturnInputs()
processor3 = caller.ReturnInputs()

in1 = processor1.return_inputs(list_,0,20)
in2 = processor2.return_inputs(list_,1,20)
in3 = processor3.return_inputs(list_,2,20)

conts=in1+in2+in3

print(np.shape(conts))
print(np.shape(conts[0]))



