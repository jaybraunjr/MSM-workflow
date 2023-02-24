# I want to use this main function to call code from feat_v2.py, which is a separate file:

import feat_v3_unstable as f_v2
import featurize.caller as caller
import MDAnalysis as mda
import numpy as np
from pyemma.coordinates import source
import pyemma
import mdtraj as md


def main():
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
    k =  [list(range(439,440))]
    l =  [list(range(462,463))]
    m =  [list(range(498,499))]
    n =  [list(range(534,535))]

    # files = ['traj_dat/whole.1.xtc', 'traj_dat/whole.2.xtc','traj_dat/whole.3.xtc','traj_dat/whole.4.xtc'] 
    top_file = 'traj_dat/10.gro'
    # traj = md.load(files, top=top_file)
    feat_md = pyemma.coordinates.featurizer(top_file)

    files = ['traj_dat/whole.1.xtc', 'traj_dat/whole.2.xtc','traj_dat/whole.3.xtc','traj_dat/w4.xtc']
    md_list = []
    mda_list = []
    for item in files:
        md_list.append(md.load(item, top='traj_dat/10.gro'))
        u_ = mda.Universe('traj_dat/10.gro', item)
        mda_list.append(u_)
        print(item)


    # Determining the number of frames in each trajectory
    n_frames = []
    for item in md_list:
        n_frames.append(len(item))

    results2 = f_v2.calulate_dists(feat_md,bil,a , b ,c,d,e,f,g,h,i,j,k,l,m,n,md_list)
   
    prot_selections = [f"(resid {i}) and (not backbone)" for i in range(1, 36)]
    lipid_selection = 'resname POPC DOPE SAPI'

    results1 = f_v2.calculate_contacts(lipid_selection, prot_selections, mda_list)

    splits1 = np.array_split(results1, len(md_list))
    splits2 = np.array_split(results2, len(md_list))

    feat_len1 = 35  
    feat_len2 = 14

    m1 = f_v2.get_fubar_output(splits1, feat_len1)
    m2 = f_v2.get_fubar_output(splits2, feat_len2)
    inputs1 = f_v2.get_inputs(m1, splits1, feat_len1)
    inputs2 = f_v2.get_inputs(m2, splits2, feat_len2)

    result_list = f_v2.combine_results(inputs1, inputs2,splits1, splits2, feat_len1, feat_len2)
    print(np.shape(result_list),'result_list')
    print(np.shape(result_list[0]),'result_list dim')
    dim_len = np.shape(result_list[0])[1]
    print(dim_len,'dim_len')

    # Now I am going to pass in the n_frames in each trajectory and dim_len to the make_arrlist function
    
    # I want to be able to increase the number of trajectories and have the code still work
    t_list = []
    for i in range(len(md_list)):
        dims =  (n_frames[i], dim_len )
        t_list.append(dims)
    print(t_list,'t_list')

    # dims =  [(n_frames[0], dim_len ), (n_frames[1], dim_len), (n_frames[2], dim_len)]
    arrlist = f_v2.make_arrlist(result_list, dims=t_list)
    caller.save_reader(arrlist, 'arrlist_test')

if __name__ == '__main__':
    main()

