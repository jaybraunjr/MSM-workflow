import matplotlib.pyplot as plt
import numpy as np
import pyemma
from pyemma.util.contexts import settings
import mdtraj as md
from itertools import combinations
from pyemma.coordinates import source
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.core.groups import AtomGroup
import math
from MDAnalysis.analysis import contacts


class dist:
    def dist_bil(self, traj, c1, c2):
        centers1 = c1.transform(traj)
        bulk = c2.transform(traj)
        cen1 = centers1[:, -1]
        cen_b = bulk[:, -1]
        sub1 = (cen1 - cen_b)
        reshaped1 = (sub1.reshape(-1, 1))

        j1 = np.absolute(reshaped1)
        return 1 / j1

    def dist_resids(self, traj, c1, c2):
        centers1 = c1.transform(traj)
        centers2 = c2.transform(traj)
        cen1 = centers1[:, -1]
        cen2 = centers2[:, -1]

        xyz = np.hstack((centers1, centers2))
        traj = md.Trajectory(xyz.reshape(-1, 2, 3), topology=None)

        cont = md.compute_distances(traj, atom_pairs=[[0, 1]], periodic=False)
        print(np.shape(cont))
        return cont

    def compute_penetration_depth(self, traj, protein_ca_indices, lipid_p_indices):
        lipid_p_positions = traj.xyz[:, lipid_p_indices]

        lipid_p_indices_top = lipid_p_indices[lipid_p_positions[..., 2].mean(axis=0) >= traj.unitcell_lengths[:, 2].mean() / 2]
        lipid_p_indices_bottom = lipid_p_indices[lipid_p_positions[..., 2].mean(axis=0) < traj.unitcell_lengths[:, 2].mean() / 2]

        lipid_p_com_top = md.compute_center_of_mass(traj.atom_slice(lipid_p_indices_top))
        lipid_p_com_bottom = md.compute_center_of_mass(traj.atom_slice(lipid_p_indices_bottom))

        protein_ca_positions = traj.xyz[:, protein_ca_indices]

        penetration_depth = np.empty((traj.n_frames, len(protein_ca_indices)))

        for frame in range(traj.n_frames):
            dist_top = np.linalg.norm(protein_ca_positions[frame] - lipid_p_com_top[frame], axis=-1)
            dist_bottom = np.linalg.norm(protein_ca_positions[frame] - lipid_p_com_bottom[frame], axis=-1)
            binding_to_top = dist_top < dist_bottom

            penetration_depth[frame] = np.where(binding_to_top, protein_ca_positions[frame, :, 2] - lipid_p_com_top[frame, 2],
                                                protein_ca_positions[frame, :, 2] - lipid_p_com_bottom[frame, 2])

            box_half_z = traj.unitcell_lengths[frame, 2] / 2

            for i, depth in enumerate(penetration_depth[frame]):
                if depth > 0 and protein_ca_positions[frame, i, 2] < lipid_p_com_bottom[frame, 2] and not binding_to_top[i]:
                    penetration_depth[frame, i] = -depth
                elif depth < 0 and protein_ca_positions[frame, i, 2] < box_half_z:
                    penetration_depth[frame, i] = np.abs(depth)

                if depth > 0 and protein_ca_positions[frame, i, 2] < box_half_z and not binding_to_top[i]:
                    penetration_depth[frame, i] = -depth

        return penetration_depth


class cont:
    def contacts_mdtraj(self, traj, cutoff, query_indices):
        j = md.compute_neighbors(traj, cutoff=cutoff, query_indices=query_indices)
        ls = []
        for arr in j:
            lst = arr.tolist()
            sort = [x for x in lst if x > 628 and x < 36617]
            ls.append(len(sort))
        
        arr = np.array(ls)
        reshaped = (arr.reshape(-1, 1))
        j = np.absolute(reshaped)
        k = j.astype('float32')
        return k

    def contacts_MDA(self, u, a, b, radius=3.5):
        timeseries = []

        for ts in u.trajectory:
            dist = contacts.distance_array(a.positions, b.positions)
            n_contacts = contacts.contact_matrix(dist, radius).sum()
            timeseries.append(n_contacts)
            arr = np.array(timeseries)
        return np.transpose([arr])

    def run_coord(self, u_, ag1, ag2, nn=6, mm=12, d0=0, r0=2.5, density=False, b=0, e=None, skip=1):
        times = []
        coords = []
        for i, ts in enumerate(u_.trajectory[b:e:skip]):
            times.append(i)
            d = distance_array(ag1.positions, ag2.positions, box=u_.dimensions)
            d[d < d0] = 1
            D = (d - d0) / r0
            sij = (1 - np.power(D, nn)) / (1 - np.power(D, mm))
            coords.append(np.sum(sij))
        if density:
            coords = np.array(coords)
            coords /= (ag1.n_atoms * ag2.n_atoms)
        return np.transpose([coords])

class tilt:
    def magnitude(a, b):
        v = a - b
        v = v * v
        sum = np.sum(v)
        return math.sqrt(sum)

    def solve(a, b):
        N = len(a)
        sum = 0

        for j in range(N):
            mag = tilt.magnitude(a[j].position, b[j].position)
            c_theta = (a[j].position[2] - b[j].position[2]) / mag
            arccos = np.arccos(c_theta)
            degrees = np.degrees(arccos)
            x_norm = degrees - 90
            sum += x_norm
        sum /= N
        return sum

    def helix_tilt(a, b, u):
        mylist = []

        for ts in u.trajectory:
            t = u.trajectory.time / 1000
            y = tilt.solve(a, b)
            mylist.append([y])
        return np.array(mylist)

class tilt_mdtraj:
    def helix_tilt_mdtraj(traj, a, b):
        atom_indices = np.array([a, b])
        helix_vectors = np.diff(traj.xyz[:, atom_indices, :], axis=1)
        z_axis = np.array([0, 0, 1])
        tilt_angles = np.arccos(np.dot(helix_vectors, z_axis))
        tilt_angles = np.degrees(tilt_angles, dtype=np.float32)
        return tilt_angles
