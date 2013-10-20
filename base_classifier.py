import sys
lib_path = 'c:/users/kenny/CS221-Song-Classification/lib/'
sys.path.append(lib_path)
import hdf5_getters
import numpy as np
import matplotlib.pyplot as plt


def get_all_timbre(track_paths):
	dic_of_timbres = dict()
	for track_id in track_paths:
		h5 = hdf5_getters.open_h5_file_read(track_paths[track_id])
		timbre_by_segment = hdf5_getters.get_segments_timbre(h5)
		h5.close()
		avg_timbre = np.mean(timbre_by_segment, axis=0)
		list_of_timbers.add(track_id, avg_timbre)
	return dic_of_timbres

def compute_distance(dic_timbres, track_id1, track_id2):
	v1 = dic_timbres[track_id1]
	v2 = dic_timbres[track_id2]
	return np.linalg.norm(v1-v2)


def is_same_clique(tracks_info, track_id1, track_id2):
	track1 = tracks_info[track_id1]
	track2 = tracks_info[track_id2]
	return track1[clique_name]==track2[clique_name]


def compute_all_distance(track_paths, tracks_info):
	dic_timbres = get_all_timbre(track_paths)
	dist_same_clique = []
	dist_diff_clique = []
	for track_id1 in track_paths:
		for track_id2 in track_paths:
			if track_id1 == track_id2:
				continue
			if(is_same_clique(tracks_info, track_id1, track_id2)):
				dist_same_clique.append(compute_distance(dic_timbres, track_id1, track_id2))
			else:
				dist_diff_clique.append(compute_distance(dic_timbres, track_id1, track_id2))
	return dist_same_clique, dist_diff_clique

#key: track_id
#value: 3 dics: clique_name, 

def plot_histogram(data):
	min_val = min(min(l) for l in data)
	max_val = max(max(l) for l in data)
	bins = numpy.linspapce(min_val, max_val, 100)

	for l in data:
		plt.hist(l, bins, alpha=0.5)
	plt.show()


def main():
	track_paths = dict() #place holder for the dictionary of paths
	tracks_info = dict() #place holder for track info
	dist_same_clique, dist_diff_clique = compute_all_distance(track_paths, tracks_info)
	plot_histogram([dist_same_clique, dist_diff_clique])




if __name__ == '__main__':
	main()