import os
import sys

from pylab import *

import logistic
import load_song_data
import feature_util
import knn

import time
import pdb


def run_logistic(track_list, pairFeatureExtractor, verbose=False):
    # Initialize classifier
    classifier = logistic.LogisticClassifier()

    t0 = time.time()
    if verbose: print "Generating pairwise dataset..."
    data = classifier.pair_dataset(track_list, 
                                   pairFeatureExtractor, 
                                   rseed=10,
                                   verbose=True)
    if verbose: print "Completed in %.02f s" % (time.time() - t0)

    # Run classifier and show output
    t0 = time.time()
    if verbose: print "Training logistic classifier on %d song pairs..." % len(data)
    score = classifier.fit(data)    
    if verbose: print "Completed in %.02f s" % (time.time() - t0)
    if verbose: print "Training accuracy: %.02f%%" % (score*100.0)

    weights = classifier.getWeights()
    if verbose: print "Weights: %s" % str(weights)

    return weights


def test_knn(train_list, test_dataset, featureExtractor, weights=None):
    if weights == None:
        weights = ones(len(data[0])) # DUMMY

    data, label = feature_util.get_feature_and_labels(featureExtractor, train_list)
    knn_classifier = knn.KNearestNeighbor(weights, data, label, k=5)
    accuracy = knn_classifier.calculate_accuracy(data, label)
    print "KNN training accuracy: %.02f%%" % (accuracy*100.0)


    test_track_list = test_dataset.get_tracks()
    print "Loaded test set of %d tracks -> %d loaded successfully" % (args.ntest, len(test_track_list))

    test_data, test_label = feature_util.get_feature_and_labels(feature_util.combo_feature_extractor, test_track_list)
    accuracy_test = knn_classifier.calculate_accuracy(test_data, test_label)
    print "KNN test accuracy: %.02f%%" % (accuracy_test*100.0)


def main(args):

    # Load Training Data
    dataset = load_song_data.Track_dataset('train')
    dataset.prune(args.ntrain)
    train_list = dataset.get_tracks()
    print "Loaded train set of %d tracks -> %d loaded successfully" % (args.ntrain, len(train_list))

    #featureExtractor = averageTimbreFeatureExtractor
    featureExtractor = feature_util.combo_feature_extractor
    pairFeatureExtractor = feature_util.make_subtractivePairFeatureExtractor(featureExtractor)

    weights = run_logistic(train_list, pairFeatureExtractor, verbose=True)


    ##
    # Run KNN
    #
    test_dataset = load_song_data.Track_dataset('test')
    test_dataset.prune(args.ntest)
    test_knn(train_list, test_dataset, featureExtractor, weights=weights)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Cover Song Identifier")

    parser.add_argument('-t', '--ntrain', dest='ntrain', type=int, default=5000)
    parser.add_argument('-e', '--ntest', dest='ntest', type=int, default=1000)

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print repr(e)
        pdb.post_mortem()