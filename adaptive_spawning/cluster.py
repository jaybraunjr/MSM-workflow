import MDAnalysis as mda
from MDAnalysis.analysis import encore
import sklearn
import pyemma.coordinates as coor
import pyemma
import numpy as np
import mdtraj as md
import pandas as pd


def tica(reader,lag, dim):
	tica = pyemma.coordinates.tica(reader, lag, dim)
	tica_output = tica.get_output()
	tica_concatenated = np.concatenate(tica_output)
	return tica_output


def cluster(inp, k):
	cluster = pyemma.coordinates.cluster_kmeans(inp, k, max_iter=10, stride=10, fixed_seed=1)
	return cluster


def binning(dtraj):
	bins=np.bincount(np.concatenate(dtraj))
	for num in bins:
		if num == np.amax(bins):
			print(num)

	new_file = np.amax(bins)
	d = dict(enumerate(bins.flatten(), 1))
	return d, new_file


def get_key(d,val):
	for key, value in d.items():
		if val == value:
			return key-1
	return "key doesn't exist"

# samples = cluster.sample_indexes_by_cluster([get_key(new_file)],20)
# return samples



def make_trajs(trajs,top):
	source = pyemma.coordinates.source(trajs, top=top)
	pyemma.coordinates.save_trajs(source, samples, outfiles=['./samples.gro'])
	return outfiles