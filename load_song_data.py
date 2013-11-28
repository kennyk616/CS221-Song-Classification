import sys
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

lib_path = os.path.abspath('./lib')
sys.path.append(lib_path)
import hdf5_getters

from utils import memoized

DATAPATH_ROOT = "./MSD-SHS/"

# Load the song data as 3 different dictionaries. 
# track path gives a (key, value) as (<work>, <path>)
# track_info gives a (key, value) as (<work>, dictionary) where the keys to the dictionary 
# are TID, AID, work, and clique_name

class Track:
    def __init__(self, track_id, path, clique, DATAPATH='.'):
        self.id = track_id
        self.path = os.path.join(DATAPATH, path)
        self.clique = clique
        try:
            self.h5 = hdf5_getters.open_h5_file_read(self.path)
            self.h5.close()
            self.h5 = None
        except Exception as e:
            self.h5 = -1
    
    def close(self):
        if self.h5 == None: return # do nothing
        self.h5.close()
        self.h5 = None

    def open(self):
        self.h5 = hdf5_getters.open_h5_file_read(self.path)

    @memoized
    def get_duration(self):
        if self.h5 == None: self.open()
        return hdf5_getters.get_duration(self.h5)

    @memoized
    def get_loudness(self):
        if self.h5 == None: self.open()
        return hdf5_getters.get_loudness(self.h5)

    @memoized
    def get_mode(self):
        if self.h5 == None: self.open()
        return hdf5_getters.get_mode(self.h5)

    @memoized
    def get_mode_confidence(self):
        if self.h5 == None: self.open()
        return hdf5_getters.get_mode_confidence(self.h5)

    @memoized
    def get_segments_loudness_max(self):
        if self.h5 == None: self.open()
        return hdf5_getters.get_segments_loudness_max(self.h5)

    @memoized
    def get_segments_loudness_max_time(self):
        if self.h5 == None: self.open()
        return hdf5_getters.get_segments_loudness_max_time(self.h5)

    @memoized
    def get_segments_loudness_start(self):
        if self.h5 == None: self.open()
        return hdf5_getters.get_segments_loudness_start(self.h5)

    @memoized
    def get_segments_pitches(self):
        if self.h5 == None: self.open()
        return hdf5_getters.get_segments_pitches(self.h5)

    @memoized
    def get_segments_timbre(self):
        if self.h5 == None: self.open()
        try:
            timbre_by_segment = hdf5_getters.get_segments_timbre(self.h5)
        except Exception as e:
            print repr(e)
            timbre_by_segment = None

        return timbre_by_segment

    @memoized
    def get_tatums_start(self):
        if self.h5 == None: self.open()
        return hdf5_getters.get_tatums_start(self.h5)

    @memoized
    def get_tatums_confidence(self):
        if self.h5 == None: self.open()
        return hdf5_getters.get_tatums_confidence(self.h5)

    @memoized
    def get_tempo(self):
        if self.h5 == None: self.open()
        return hdf5_getters.get_tempo(self.h5)

    @memoized
    def get_time_signature(self):
        if self.h5 == None: self.open()
        return hdf5_getters.get_time_signature(self.h5)

    @memoized
    def get_time_signature_confidence(self):
        if self.h5 == None: self.open()
        return hdf5_getters.get_time_signature_confidence(self.h5)



class Track_dataset:
    def __init__(self):
        def get_dic(path):
            json_data = open(path)
            return json.load(json_data)

        self.track_paths_full = get_dic(DATAPATH_ROOT + 'shs_dataset_full/shs_dataset_full.trackpaths.json')
        self.track_info_full = get_dic(DATAPATH_ROOT + 'shs_dataset_full/shs_dataset_full.tracks.json')
        self.track_paths_train = dict()
        self.track_info_train = dict()
        self.track_paths_test = dict()
        self.track_info_test = dict()

        self.track_paths_loc_train = get_dic(DATAPATH_ROOT + 'shs_dataset_train/shs_dataset_train.trackpaths.json')
        self.track_paths_loc_test = get_dic(DATAPATH_ROOT + 'shs_dataset_test/shs_dataset_test.trackpaths.json')


    def prune(self, ncliques=300):
        """Prune the dataset down to a smaller number of tracks."""

        # get the count for cliques, and sort by size of the clique
        clique_counter = Counter()
        for k, v in self.track_info_full.iteritems():
            clique_counter[v['clique_name']] += 1
        sorted_clique_counter = dict(sorted(clique_counter.items(), key= lambda item: item[1], reverse=True)[:ncliques])
        #print sorted_clique_counter

        #trim down self.track_info_full
        # get a dictionary of {clique_name : list of track_ids}
        tmp_dic = dict()
        clique_dic = dict()
        for track_id, values in self.track_info_full.iteritems():
            if values['clique_name'] in sorted_clique_counter:
                tmp_dic[track_id] = values
                if values['clique_name'] in clique_dic:
                    clique_dic[values['clique_name']].append(track_id)
                else:
                    clique_dic[values['clique_name']] = [track_id]
                    
        self.track_info_full = tmp_dic

        # split up training and testing sets. 1/3 for testing (round down), and 2/3 for training (round up)
        for clique, tracks in clique_dic.iteritems():
            n_test = len(tracks)/3
            for i in xrange(len(tracks)):
                if i < n_test:
                    self.track_info_test[tracks[i]] = self.track_info_full[tracks[i]]
                else:
                    self.track_info_train[tracks[i]] = self.track_info_full[tracks[i]]

        # populate self.track_paths
        self.track_paths_train = {k:v for k,v in self.track_paths_full.items() if k in self.track_info_train}
        self.track_paths_test = {k:v for k,v in self.track_paths_full.items() if k in self.track_info_test}


    def get_track(self, track_id):
        def path_loc(track_id):
            if track_id in self.track_paths_loc_train:
                return 'train'
            elif track_id in self.track_paths_loc_test:
                return 'test'
            else:
                # should never go to this line
                print "Error: not in train or test!"
                return None
        if track_id in self.track_info_train:
            clique_name = self.track_info_train[track_id]['clique_name']
            track_path = self.track_paths_train[track_id]
        else:
            clique_name = self.track_info_test[track_id]['clique_name']
            track_path = self.track_paths_test[track_id]
        return Track(track_id, track_path, clique_name, DATAPATH=os.path.join(DATAPATH_ROOT, path_loc(track_id)))

    def get_tracks_train(self):
        return self.get_tracks(self.track_paths_train)

    def get_tracks_test(self):
        return self.get_tracks(self.track_paths_test)

    def get_tracks(self, paths):
        tracks = []
        for track_id in paths:
            new_track = self.get_track(track_id)
            if new_track.h5 == -1: 
                print "Error: unable to open track %s" % track_id
                continue
            tracks.append(new_track)

        return tracks
