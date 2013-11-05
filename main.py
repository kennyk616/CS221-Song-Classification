import os
import sys

from pylab import *

import logistic
import load_song_data
import feature_util
import knn

import time
import pdb

def main(ntracks=20000):

    # Load data
    dataset_name = 'train'
    dataset = load_song_data.Track_dataset(dataset_name)
    dataset.prune(ntracks)
    track_list = dataset.get_tracks()

    # Timbre features
    def averageTimbreFeatureExtractor(track):
        t_segments = track.get_segments_timbre()
        return np.mean(t_segments, axis=0)

    #featureExtractor = averageTimbreFeatureExtractor
    #pairFeatureExtractor = feature_util.make_subtractivePairFeatureExtractor(featureExtractor)

    comboFeatureExtractor = feature_util.combo_feature_extractor

    # # Initialize classifier
    # classifier = logistic.LogisticClassifier()

    # t0 = time.time()
    # print "Generating pairwise dataset..."
    # data = classifier.pair_dataset(track_list, 
    #                                pairFeatureExtractor, 
    #                                rseed=10,
    #                                verbose=True)
    # print "Completed in %.02f s" % (time.time() - t0)

    # # Run classifier and show output
    # t0 = time.time()
    # print "Training logistic classifier on %d song pairs..." % len(data)
    # score = classifier.fit(data)    
    # print "Completed in %.02f s" % (time.time() - t0)
    # print "Training error: %.02f%%" % (score*100.0)
    # print "Weights: %s" % str(classifier.getWeights())

    # weights = classifier.getWeights()
    data, label = feature_util.get_feature_and_labels(util.combo_feature_extractor, track_list)
    weights = [1]*len(data[0])
    knn_classifier = knn.KNearestNeighbor(weights, data, label, k=5)
    accuracy = knn_classifier(data, label)
    print "KNN accuracy on training data: ", accuracy

    test_dataset = load_song_data.Truck_dataset('test')
    test.dataset.prune(1000)
    test_track_list = test_dataset.get_tracks()
    test_data, test_label = feature_util.get_feature_and_labels(util.combo_feature_extractor, test_track_list)
    accuracy_test = knn_classifier(test_data, test_label)
    print "KNN accuracy on testing data: ", accuracy_test



if __name__ == '__main__':
    try:
        main(1000)
    except Exception as e:
        print repr(e)
        pdb.post_mortem()