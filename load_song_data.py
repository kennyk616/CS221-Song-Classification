import sys
import os
import json

import numpy as np
import matplotlib.pyplot as plt

lib_path = os.path.abspath('./lib')
sys.path.append(lib_path)
import hdf5_getters

from utils import memoized

DATAPATH = "./MSD-SHS/train/"

# Load the song data as 3 different dictionaries. 
# track path gives a (key, value) as (<work>, <path>)
# track_info gives a (key, value) as (<work>, dictionary) where the keys to the dictionary 
# are TID, AID, work, and clique_name

class Track:
    def __init__(self, track_id, path, clique):
        self.id = track_id
        self.path = path
        self.clique = clique
    
    @memoized
    def get_segments_timbre(self):
        path = DATAPATH + self.path
        h5 = hdf5_getters.open_h5_file_read(path)
        try:
            timbre_by_segment = hdf5_getters.get_segments_timbre(h5)
        except Exception as e:
            print repr(e)
            timbre_by_segment = None

        h5.close()
        return timbre_by_segment



class Track_dataset:
    def __init__(self, data_set):
        def get_dic(path):
            json_data = open(path)
            return json.load(json_data)

        self.track_paths = get_dic('./MSD-SHS/shs_dataset_' + data_set + '/shs_dataset_' + data_set + '.trackpaths.json')
        self.track_info = get_dic('./MSD-SHS/shs_dataset_' + data_set + '/shs_dataset_' + data_set + '.tracks.json')
        # self.track_cliques_shs = get_dic('./MSD-SHS/shs_dataset_' + data_set + '/shs_dataset_' + data_set + '.cliques.json')

    def prune(self, ntracks=1000):
        """Prune the dataset down to a smaller number of tracks."""
        kf = lambda (k,v): v['clique_name']
        def pare_dict(d,n):
            return dict(sorted(d.items(), key=kf)[:n])
        self.track_info = pare_dict(self.track_info, ntracks)
        self.track_paths = {k:v for k,v in self.track_paths.items() if k in self.track_info}


    def get_track(self, track_id):
        return Track(track_id, self.track_paths[track_id], self.track_info[track_id]['clique_name'])

    def get_tracks(self):
        return [self.get_track(track_id) for track_id in self.track_paths.keys()]


