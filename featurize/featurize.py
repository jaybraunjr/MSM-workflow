
import matplotlib.pyplot as plt
# import matplotlib as mpl
import pyemma
from pyemma.util.contexts import settings
import mdtraj
from itertools import combinations
import pandas as pdnp
import numpy as np
from pyemma.coordinates import source
import mdtraj as md
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.core.groups import AtomGroup
import math
from MDAnalysis.analysis import contacts


#### this is inefficient, should find a way to make this simple

# traj = md.load(../)

#######################################################################################
### these are used from MDtraj, implemneted into pyemma. They do the work for you when 
### this is incorporated into the custom featurizer

class dist:


    def dist_bil(self, traj,c1,c2):
        centers1 = c1.transform(traj)
        bulk = c2.transform(traj)
        cen1 = centers1[:,-1]
        cen_b = bulk[:,-1]
        sub1=(cen1-cen_b)
        reshaped1=(sub1.reshape(-1,1))

        j1=np.absolute(reshaped1)
        # k1=j1.astype('float32')
        return 1/j1

### below is different method.
    def dist_resids(self, traj,c1,c2):
        centers1 = c1.transform(traj)  # yields ndarray
        centers2 = c2.transform(traj)  # yields ndarray
        cen1 = centers1[:,-1]
        cen2 = centers2[:,-1]

        xyz = np.hstack((centers1, centers2))
        traj = md.Trajectory(xyz.reshape(-1, 2, 3), topology=None)
        
        cont = md.compute_distances(traj,atom_pairs=[[0, 1]], periodic=False)
        print(np.shape(cont))
        return cont

##########################################################################################

### contact analysis through MDtraj. Sloppy, would not use if have to.
### the below list has to be creating in script
class cont:

    def contacts_mdtraj(self, traj, cutoff, query_indices):

        """
        The function is counting number of contacts at a given cuttoff
        - The inputs for the function are (trajectory, cuttoff, and queru inceces (q))
        - q has to be defined in the script before function:

            q=[]
            for number in range(15,24):
                q.append(number)

        """
        cutoff=0.35

        j=md.compute_neighbors(traj,cutoff=cutoff,query_indices=q)
        ls=[]
        for arr in j:
            lst=arr.tolist()
            sort = [x for x in lst if x>628 and x<36617]
            ls.append(len(sort))
        
        arr=np.array(ls)
        reshaped=(arr.reshape(-1,1))
        j=np.absolute(reshaped)
        k=j.astype('float32')
        return k
#####################################################################
### Now MDAnalyis contact script. Usually a bit more tunable 
### and can be used with MDAnalysis
    def contacts_MDA(self, u, a, b, radius=3.5):

        timeseries = []

        for ts in u.trajectory:
            dist = contacts.distance_array(a.positions, b.positions)
            n_contacts = contacts.contact_matrix(dist,radius).sum()
            timeseries.append(n_contacts)
            arr = np.array(timeseries)
        return np.transpose([arr])

#####################################################################
    def run_coord(self, u_, ag1, ag2, nn=6, mm=12, d0=0, r0=2.5, density=False, b=0, e=None, skip=1):

        """
        coordination in MDAnalysis created by Dr. Siyoung Kim. This is normalized coordination

        """
        times = []; coords = []
        for i, ts in enumerate(u_.trajectory[b:e:skip]):
            times.append(i)
            d = distance_array(ag1.positions, ag2.positions, box=u_.dimensions)
            d[ d < d0 ] = 1
            D = (d - d0)/r0
            sij = (1 - np.power(D, nn))/(1 - np.power(D, mm))
            coords.append(np.sum(sij))
        if density:
            coords = np.array(coords)
            coords /= (ag1.n_atoms * ag2.n_atoms)
        return np.transpose([coords])

################################################################
## IN DEVELOPMENT
## Functions for helix tilt below:
class tilt:
    
    def magnitude(a,b):
        v = a-b
        v=v*v
        sum = np.sum(v)
        return math.sqrt(sum)

    def solve(a,b):

        N = len(a)
        sum = 0

        for j in range(N):
            mag = magnitude(a[j].position,b[j].position)
            c_theta = (a[j].position[2]-b[j].position[2])/mag
            arccos = np.arccos(c_theta)
            degrees = np.degrees(arccos)
            x_norm = degrees - 90
            sum += x_norm
        sum /= N
        return sum


    def helix_tilt(a,b,u):

        """
        This takes in the argument of a=atom_1 b=atom_2, and u=universe

        """
        mylist =[]

        for ts in u.trajectory:
            # calling function to solve order param in each timestep
            t = u.trajectory.time/1000
            y = solve(a,b)
            mylist.append([y])
        return np.array(mylist)
    

class tilt_mdtraj:

    def helix_tilt_mdtraj(traj,a,b):
        """
        this uses mdtraj to caluclate the helix tilt
        """
        atom_indices = np.array([a, b])
        helix_vectors = np.diff(traj.xyz[:, atom_indices, :], axis=1)
        z_axis = np.array([0, 0, 1])
        tilt_angles = np.arccos(np.dot(helix_vectors, z_axis))
        tilt_angles = np.degrees(tilt_angles,dtype=np.float32)
        return tilt_angles
    
