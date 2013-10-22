import sys
import os
import json
import time

import numpy as np
import matplotlib.pyplot as plt

lib_path = os.path.abspath('./lib')
sys.path.append(lib_path)
import hdf5_getters

import pdb

DATAPATH = "./MSD-SHS/train/"

def get_all_timbre(track_paths):
    dic_of_timbres = dict()
    count = 0
    for track_id in track_paths:
        path = DATAPATH+track_paths[track_id]
        h5 = hdf5_getters.open_h5_file_read(path)
        try:
            timbre_by_segment = hdf5_getters.get_segments_timbre(h5)
        except Exception as e:
            print repr(e)
            continue
        h5.close()
        avg_timbre = np.mean(timbre_by_segment, axis=0)
        dic_of_timbres[track_id] = avg_timbre
        count +=1
    return dic_of_timbres

def compute_distance(dic_timbres, track_id1, track_id2):
    v1 = dic_timbres[track_id1]
    #print track_id2 in dic_timbres
    v2 = dic_timbres[track_id2]
    return np.linalg.norm(v1-v2)


def is_same_clique(tracks_info, track_id1, track_id2):
    track1 = tracks_info[track_id1]
    track2 = tracks_info[track_id2]
    return track1["clique_name"]==track2["clique_name"]


def compute_all_distance(track_paths, tracks_info):
    dic_timbres = get_all_timbre(track_paths)
    dist_same_clique = []
    dist_diff_clique = []
    for track_id1 in dic_timbres:
        for track_id2 in dic_timbres:
            if track_id1 == track_id2:
                continue
            if(is_same_clique(tracks_info, track_id1, track_id2)):
                dist_same_clique.append(compute_distance(dic_timbres, track_id1, track_id2))
            else:
                dist_diff_clique.append(compute_distance(dic_timbres, track_id1, track_id2))
    return dist_same_clique, dist_diff_clique


def compute_distances(track_paths, tracks_info, ntracks=1000):
    # Pare dictionary down to first n elements, sorted by clique
    kf = lambda (k,v): v['clique_name']
    tracks_info = dict(sorted(tracks_info.items(), key=kf)[:ntracks])
    track_paths = {k:v for k,v in track_paths.items() if k in tracks_info}

    cliqueset = {v['clique_name'] for v in tracks_info.values()}
    print "%d cliques found in %d tracks." % (len(cliqueset), len(tracks_info))

    return compute_all_distance(track_paths, tracks_info)


#key: track_id
#value: 3 dics: clique_name, 

def plot_histogram(data):
    min_val = min(min(l) for l in data)
    max_val = max(max(l) for l in data)
    bins = np.linspace(min_val, max_val, 100)

    print "min_val", min_val
    print "max_val", max_val
    #print "bins", bins
    print "within clique: %d" % len(data[0])
    mean0 = np.mean(data[0])
    stdmean0 = np.std(data[0]) / np.sqrt(len(data[0])-1)
    print "  mean=%.02f +/- %.02f" % (mean0, stdmean0)
    median0 = np.median(data[0])
    print "  median=%.02f" % median0

    print "not in clique: %d" % len(data[1])
    mean1 = np.mean(data[1])
    stdmean1 = np.std(data[1]) / np.sqrt(len(data[1])-1)
    print "  mean=%.02f +/- %.02f" % (mean1, stdmean1)
    median1 = np.median(data[1])
    print "  median=%.02f" % median1

    plt.figure(1).clear()

    plt.subplot(2,1,1)
    plt.hist(data[0], bins, alpha=0.5, label=("Same Clique (%d)" % len(data[0])), color='r')
    plt.xlim(min(bins), max(bins))
    plt.axvline(median0, linewidth=2, linestyle='--', alpha=0.5, color='k', label='median %.02f' % median0)
    plt.legend()

    plt.subplot(2,1,2)
    plt.hist(data[1], bins, alpha=0.5, label=("Different Cliques (%d)" % len(data[1])), color='b')
    plt.xlim(min(bins), max(bins))
    plt.axvline(median1, linewidth=2, linestyle='--', alpha=0.5, color='k', label='median %.02f' % median1)
    plt.legend()


    #for l in data:
    #   plt.hist(l, bins, alpha=0.5)
    plt.show()


def main():
    ntracks = int(sys.argv[1])

    json_data_paths=open('./MSD-SHS/shs_dataset_train/shs_dataset_train.trackpaths.json')
    json_data=open('./MSD-SHS/shs_dataset_train/shs_dataset_train.tracks.json')
    track_paths = json.load(json_data_paths) #place holder for the dictionary of paths
    track_info = json.load(json_data) #place holder for track info
    #print track_paths["TRQPZPQ128F4292D21"]
    #print track_info["TRQPZPQ128F4292D21"]

    t0 = time.time()
    dist_same_clique, dist_diff_clique = compute_distances(track_paths, track_info, ntracks=ntracks)
    plot_histogram([dist_same_clique, dist_diff_clique])
    print "Completed in %.02f s" % (time.time() - t0)

    plt.figure(1)
    plt.suptitle("shs_dataset_train, %d tracks" % ntracks)
    # plt.savefig(sys.argv[2])



if __name__ == '__main__':
    main()