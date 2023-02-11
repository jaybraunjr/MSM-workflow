# I want to use this main function to call code from feat_v2.py, which is a separate file:

import feat_v2 as f_v2
import featurize.caller as caller

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
    top_file = 'traj_dat/10.gro'
    tr = top_file
    feat_ = f_v2.caller.feature(tr)
    traj_files = ['traj_dat/whole.1.xtc', 'traj_dat/whole.2.xtc','traj_dat/whole.3.xtc']
    top_file = 'traj_dat/10.gro'
    nlist= f_v2.make_nlist()
    inp_2 = f_v2.calulate_dists(feat_,bil,a ,b ,c,d,e,f,g,h,i,j,nlist)
    prot_selections = [f"(resid {i}) and (not backbone)" for i in range(1, 11)]
    lipid_selection = 'resname POPC DOPE SAPI'
    universes = [mda.Universe('traj_dat/10.gro', traj) for traj in traj_files]
    inp_1 = f_v2.calculate_contacts(lipid_selection, prot_selections, universes)
    traj_chunk = 3
    feat_len1 = 10
    feat_len2 = 10
    feat_len3 = 10
    conts = f_v2.reshaping(inp_1, inp_2, traj_chunk, feat_len1, feat_len2)
    dims =  [(11, 20), (5, 20), (11, 20)]
    arrlist = f_v2.make_arrlist(conts, dims=dims)
    caller.save_reader(arrlist, 'arrlist')

if __name__ == '__main__':
    main()

