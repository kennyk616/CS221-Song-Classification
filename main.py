import os
import sys

from pylab import *

import logistic
import load_song_data
import feature_util

import time
import pdb

def main(ntracks=20000):

    # Load data
    dataset_name = 'train'
    dataset = load_song_data.Track_dataset(dataset_name)
    dataset.prune(ntracks)
    songList = dataset.get_tracks()

    # Timbre features
    def averageTimbreFeatureExtractor(song):
        t_segments = song.get_segments_timbre()
        return np.mean(t_segments, axis=0)

    featureExtractor = averageTimbreFeatureExtractor
    pairFeatureExtractor = feature_util.make_subtractivePairFeatureExtractor(featureExtractor)

    # Initialize classifier
    classifier = logistic.LogisticClassifier()

    t0 = time.time()
    print "Generating pairwise dataset..."
    data = classifier.pair_dataset(songList, 
                                   pairFeatureExtractor, 
                                   rseed=10,
                                   verbose=True)
    print "Completed in %.02f s" % (time.time() - t0)

    # Run classifier and show output
    t0 = time.time()
    print "Training logistic classifier on %d song pairs..." % len(data)
    score = classifier.fit(data)    
    print "Completed in %.02f s" % (time.time() - t0)
    print "Training error: %.02f%%" % (score*100.0)
    print "Weights: %s" % str(classifier.getWeights())



if __name__ == '__main__':
    try:
        main(1000)
    except Exception as e:
        print repr(e)
        pdb.post_mortem()