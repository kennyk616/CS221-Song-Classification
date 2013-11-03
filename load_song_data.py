import sys
import os
import json

import numpy as np
import matplotlib.pyplot as plt

lib_path = os.path.abspath('./lib')
sys.path.append(lib_path)
import hdf5_getters

DATAPATH = "./MSD-SHS/train/"

# Load the song data as 3 different dictionaries. 
# track path gives a (key, value) as (<work>, <path>)
# track_info gives a (key, value) as (<work>, dictionary) where the keys to the dictionary 
# are TID, AID, work, and clique_name

class Track:
    def __init__(self, track_id, path, clique):
        self.id = track_id
        self.path = DATAPATH + self.path
        self.clique = clique
        self.h5 = hdf5_getters.open_h5_file_read(path)
    
    def get_duration(self):
        return hdf5_getters.get_duration(h5)

    def get_loudness(self):
        return hdf5_getters.get_loudness(h5)

    def get_mode(self):
        return hdf5_getters.get_mode(h5)

    def get_mode_confidence(self):
        return hdf5_getters.get_mode_confidence(h5)

    def get_segments_loudness_max(self):
        return hdf5_getters.get_segments_loudness_max(h5)

    def get_segments_loudness_max_time(self):
        return hdf5_getters.get_segments_loudness_max_time(h5)

    def get_segments_loudness_start(self):
        return hdf5_getters.get_segments_loudness_start(h5)

    def get_segments_pitches(self):
        return hdf5_getters.get_segments_pitches(h5)

    def get_segments_timbre(self):
        try:
            timbre_by_segment = hdf5_getters.get_segments_timbre(h5)
        except Exception as e:
            print repr(e)
            timbre_by_segment = None
        return timbre_by_segment

    def get_tatums_start(self):
        return hdf5_getters.get_tatums_start(h5)

    def get_tatums_confidence(self):
        return hdf5_getters.get_tatums_confidence(h5)

    def get_tempo(self):
        return hdf5_getters.get_tempo(h5)

    def get_time_signature(self):
        return hdf5_getters.get_time_signature(h5)

    def get_time_signature_confidence(self):
        return hdf5_getters.get_time_signature_confidence(h5)



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


