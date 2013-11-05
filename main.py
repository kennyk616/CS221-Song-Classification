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


def run_logistic(train_list, test_list, pairFeatureExtractor, 
                 verbose=False, 
                 do_scale=True,
                 reg='l2'):
    # Initialize classifier
    classifier = logistic.LogisticClassifier(reg=reg)

    def load_data(track_list, fit=True):
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
            if fit: classifier.fit_scaler(data[0]) # calculate preprocessor parameters
            data = (classifier.scale(data[0]), data[1]) # preprocess

        return data

    train_data = load_data(train_list, fit=True)

    # Run classifier and show output
    t0 = time.time()
    if verbose: print "Training logistic classifier on %d song pairs..." % len(train_data[0])
    score = classifier.fit(*train_data)    
    if verbose: print "Completed in %.02f s" % (time.time() - t0)
    if verbose: print "Training accuracy: %.02f%%" % (score*100.0)

    weights = classifier.getWeights()
    if verbose >= 2: print "Weights: %s" % str(weights)

    ##
    # Evaluate on test set
    # TO-DO: do a proper cross-validation here
    test_data = load_data(test_list, fit=False)
    if verbose: print "Testing logistic classifier on %d song pairs..." % len(test_data[0])
    score = classifier.test(*test_data)
    if verbose: print "Completed in %.02f s" % (time.time() - t0)
    if verbose: print "Test accuracy: %.02f%%" % (score*100.0)

    return weights, classifier.scaler


def test_knn(train_list, test_list, featureExtractor, weights=None, transform=None):
    if weights == None:
        weights = ones(len(data[0])) # DUMMY

    data, label = feature_util.get_feature_and_labels(featureExtractor, train_list)

    # Transform data (preprocessor)
    if transform != None: data = transform.transform(data)

    knn_classifier = knn.KNearestNeighbor(weights, data, label, k=5)
    accuracy = knn_classifier.calculate_accuracy(data, label)
    print "KNN training accuracy: %.02f%%" % (accuracy*100.0)

    test_data, test_label = feature_util.get_feature_and_labels(feature_util.combo_feature_extractor, test_list)

    # Transform data (preprocessor)
    if transform != None: test_data = transform.transform(test_data)

    accuracy_test = knn_classifier.calculate_accuracy(test_data, test_label)
    print "KNN test accuracy: %.02f%%" % (accuracy_test*100.0)


def main(args):

    # Load Training list
    dataset = load_song_data.Track_dataset('train')
    dataset.prune(args.ntrain)
    train_list = dataset.get_tracks()
    print "Reading train set of %d tracks -> %d loaded successfully" % (args.ntrain, len(train_list))

    # Load Test list
    test_dataset = load_song_data.Track_dataset('test')
    test_dataset.prune(args.ntest)
    test_list = test_dataset.get_tracks()
    print "Reading test set of %d tracks -> %d loaded successfully" % (args.ntest, len(test_list))

    #featureExtractor = averageTimbreFeatureExtractor
    featureExtractor = feature_util.combo_feature_extractor
    pairFeatureExtractor = feature_util.make_subtractivePairFeatureExtractor(featureExtractor)

    weights, scaler = None, None
    if args.do_logistic:
        weights, scaler = run_logistic(train_list, test_list,
                                       pairFeatureExtractor, 
                                       verbose=1,
                                       do_scale=args.preprocess)
        ##
        # Plot weight vector
        #
        if args.do_plot:
            figure(1).clear()
            bar(range(weights.size), abs(weights.flatten()), width=0.9, align='center')
            xlim(0,weights.size)
            xlabel("Index ($i$)")
            ylabel("$|Weight_i|$")
            show()

      
    if args.do_knn:  
        ##
        # Run KNN
        # If logistic was run, will use the data scaler and weights
        # as an improved metric
        #
        test_knn(train_list, test_list, featureExtractor, 
                 weights=weights, 
                 transform=scaler)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Cover Song Identifier")

    parser.add_argument('-t', '--ntrain', dest='ntrain', type=int, default=5000)
    parser.add_argument('-e', '--ntest', dest='ntest', type=int, default=1000)

    parser.add_argument('--logistic', dest='do_logistic', action='store_true')
    parser.add_argument('--knn', dest='do_knn', action='store_true')
    parser.add_argument('-p', '--pre', dest='preprocess', action='store_true')
    parser.add_argument('-r', '--reg', dest='reg', metavar='regularization', default='l1',
                        choices=['l1','l2'])

    # Enable plotting
    parser.add_argument('--plot', dest='do_plot', action='store_true')


    args = parser.parse_args()
    if not args.do_logistic and not args.do_knn:
        args.do_knn = True
        args.do_logistic = True

    try:
        main(args)
    except Exception as e:
        print repr(e)
        pdb.post_mortem()