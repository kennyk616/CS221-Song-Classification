import os
import sys

# For matplotlib png output
# import matplotlib
# matplotlib.use('Agg')

from pylab import *

import logistic
import load_song_data
import feature_util
import knn

import time
import pdb


def run_logistic(track_list, pairFeatureExtractor, 
                 verbose=False, 
                 do_scale=True,
                 reg='l2'):
    # Initialize classifier
    classifier = logistic.LogisticClassifier(reg=reg)

    t0 = time.time()
    if verbose: print "Generating pairwise dataset..."
    data_dict = classifier.pair_dataset(track_list, 
                                   pairFeatureExtractor, 
                                   rseed=10,
                                   verbose=True)
    data = classifier.data_toarray(data_dict)
    if verbose: print "Completed in %.02f s" % (time.time() - t0)

    if do_scale:
        print "Preprocessing data with scaler..."
        classifier.fit_scaler(data[0]) # calculate preprocessor parameters
        data = (classifier.scale(data[0]), data[1]) # preprocess

    # Run classifier and show output
    t0 = time.time()
    if verbose: print "Training logistic classifier on %d song pairs..." % len(data[0])
    score = classifier.fit(*data)    
    if verbose: print "Completed in %.02f s" % (time.time() - t0)
    if verbose: print "Training accuracy: %.02f%%" % (score*100.0)

    weights = classifier.getWeights()
    if verbose: print "Weights: %s" % str(weights)

    return weights, classifier.scaler


def test_knn(train_list, test_dataset, featureExtractor, weights=None, transform=None):
    if weights == None:
        weights = ones(len(data[0])) # DUMMY

    data, label = feature_util.get_feature_and_labels(featureExtractor, train_list)

    # Transform data (preprocessor)
    if transform != None: data = transform.transform(data)

    knn_classifier = knn.KNearestNeighbor(weights, data, label, k=5)
    accuracy = knn_classifier.calculate_accuracy(data, label)
    print "KNN training accuracy: %.02f%%" % (accuracy*100.0)


    test_track_list = test_dataset.get_tracks()
    print "Loaded test set of %d tracks -> %d loaded successfully" % (args.ntest, len(test_track_list))

    test_data, test_label = feature_util.get_feature_and_labels(feature_util.combo_feature_extractor, test_track_list)

    # Transform data (preprocessor)
    if transform != None: test_data = transform.transform(test_data)

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

    weights, scaler = run_logistic(train_list, 
                                   pairFeatureExtractor, 
                                   verbose=True,
                                   do_scale=args.preprocess)
    ##
    # Plot weight vector
    #
    figure(1).clear()
    bar(range(weights.size), abs(weights.flatten()), width=0.9, align='center')
    xlim(0,weights.size)
    xlabel("Index ($i$)")
    ylabel("$|Weight_i|$")
    show()

    
    ##
    # Run KNN
    #
    test_dataset = load_song_data.Track_dataset('test')
    test_dataset.prune(args.ntest)
    test_knn(train_list, test_dataset, featureExtractor, 
             weights=weights, 
             transform=scaler)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Cover Song Identifier")

    parser.add_argument('-t', '--ntrain', dest='ntrain', type=int, default=5000)
    parser.add_argument('-e', '--ntest', dest='ntest', type=int, default=1000)

    # Run preprocessing (scaler)?
    parser.add_argument('-p', '--pre', dest='preprocess', action='store_true')
    parser.add_argument('-r', '--reg', dest='reg', metavar='regularization', default='l2',
                        choices=['l1','l2'])

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print repr(e)
        pdb.post_mortem()