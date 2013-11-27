import sys
import os
import json

import numpy as np
import matplotlib.pyplot as plt

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

        #self.data_set = data_set # test, train, full
        self.track_paths_train = get_dic(DATAPATH_ROOT + 'shs_dataset_train/shs_dataset_train.trackpaths.json')
        self.track_paths_test = get_dic(DATAPATH_ROOT + 'shs_dataset_test/shs_dataset_test.trackpaths.json')
        self.track_info_train = get_dic(DATAPATH_ROOT + 'shs_dataset_train/shs_dataset_train.tracks.json')
        self.track_info_test = get_dic(DATAPATH_ROOT + 'shs_dataset_test/shs_dataset_test.tracks.json')
        # self.track_cliques_shs = get_dic('./MSD-SHS/shs_dataset_' + data_set + '/shs_dataset_' + data_set + '.cliques.json')

    def prune(self, ntracks=1000):
        """Prune the dataset down to a smaller number of tracks."""
        kf = lambda (k,v): v['clique_name']
        def pare_dict(d,n):
            return dict(sorted(d.items(), key=kf)[:n])
        self.track_info_train = pare_dict(self.track_info_train, ntracks)
        self.track_paths_train = {k:v for k,v in self.track_paths_train.items() if k in self.track_info_train}

        all_cliques = set()
        for k, v in self.track_info_train.iteritems():
            all_cliques.add(v['clique_name'])
        print "# clicks = ", len(all_cliques)

        tmp_dic = dict()
        for track_id, values in self.track_info_test.iteritems():
            if values['clique_name'] in all_cliques:
                tmp_dic[track_id] = values
        self.track_info_test = tmp_dic
        self.track_paths_test = {k:v for k,v in self.track_paths_test.items() if k in self.track_info_test}


    def get_track(self, track_id):
        if track_id in self.track_info_train:
            clique_name = self.track_info_train[track_id]['clique_name']
            track_path = self.track_paths_train[track_id]
            path = 'train'
        else:
            clique_name = self.track_info_test[track_id]['clique_name']
            track_path = self.track_paths_test[track_id]
            path = 'test'
        return Track(track_id, track_path, clique_name, DATAPATH=os.path.join(DATAPATH_ROOT, path))

    def get_tracks_train(self):
        return self.get_tracks('train')

    def get_tracks_test(self):
        return self.get_tracks('test')

    def get_tracks(self, path):
        tracks = []
        if path =='train':
            paths = self.track_paths_train
        else:
            paths = self.track_paths_test

        for track_id in paths:
            newtrack = self.get_track(track_id)
            if newtrack.h5 == -1: 
                print "Error: unable to open track %s" % track_id
                continue
            tracks.append(newtrack)

        return tracks
