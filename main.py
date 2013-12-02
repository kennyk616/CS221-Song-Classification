import os
import sys

# For matplotlib png output
import matplotlib
matplotlib.use('Agg')

from pylab import *

import logistic
import transform
import load_song_data
import feature_util
import knn

import time
import pdb


def run_logistic(train_list, test_list, pairFeatureExtractor, 
                 rseed=10,
                 verbose=False, 
                 pre_mode='none',
                 npca=None,
                 reg='l2',
                 rstrength=1.0):
    # Initialize classifier
    if pre_mode == 'scale': xform = transform.StandardScaler()
    elif pre_mode == 'whiten': xform = transform.PCAWhitener()
    elif pre_mode == 'both': xform = transform.ScaleThenWhiten()
    elif pre_mode == 'pca': 
        print "PCA: using %g components" % npca
        xform = transform.PCAWhitener(n_components=npca)
    else: xform = transform.IdentityTransformer()
    print "Data preprocessor: %s" % str(type(xform))

    classifier = logistic.LogisticClassifier(reg=reg, 
                                             rstrength=rstrength,
                                             dataTransformer=xform)

    def load_data(track_list, fit=True):
        t0 = time.time()
        if verbose: 
            print "\n===================="
            print "Generating pairwise dataset..."
        data_dict = classifier.pair_dataset(track_list, 
                                       pairFeatureExtractor, 
                                       rseed=rseed,
                                       verbose=True)
        data = classifier.transformer.data_toarray(data_dict)
        if verbose: print "Completed in %.02f s" % (time.time() - t0)

        print "Preprocessing data..."
        if fit: classifier.transformer.fit(data[0]) # calculate preprocessor parameters
        data = (classifier.transformer.transform(data[0]), data[1]) # preprocess

        return data

    train_data = load_data(train_list, fit=True)

    # Run classifier and show output
    t0 = time.time()
    if verbose: print "Training logistic classifier on %d song pairs..." % len(train_data[0])
    score = classifier.fit(*train_data)    
    if verbose: print "Completed in %.02f s" % (time.time() - t0)
    if verbose: print "==> Training accuracy: %.02f%%" % (score*100.0)

    weights = classifier.getWeights()
    if verbose >= 2: print "Weights: %s" % str(weights)

    ##
    # Evaluate on test set
    # TO-DO: do a proper cross-validation here
    test_data = load_data(test_list, fit=False)
    if verbose: print "Testing logistic classifier on %d song pairs..." % len(test_data[0])
    score = classifier.test(*test_data)
    if verbose: print "Completed in %.02f s" % (time.time() - t0)
    if verbose: print "==> Test accuracy: %.02f%%" % (score*100.0)

    return weights, classifier.transformer


def test_knn(train_list, test_list, featureExtractor, 
             k = 5,
             metric='euclidean',
             weights=None, transform=None):

    data, label = feature_util.get_feature_and_labels(featureExtractor, train_list)

    if weights == None:
        weights = ones(len(data[0])) # DUMMY

    # Transform data (preprocessor)
    if transform != None: data = transform.transform(data)

    knn_classifier = knn.KNearestNeighbor(weights, data, label, k=k, metric=metric)
    print "Running KNN with k=%d and %s metric" % (k, metric)
    accuracy = knn_classifier.calculate_accuracy(data, label)
    print "==> KNN training accuracy: %.02f%%" % (accuracy*100.0)

    test_data, test_label = feature_util.get_feature_and_labels(featureExtractor, test_list)

    # Transform data (preprocessor)
    if transform != None: test_data = transform.transform(test_data)

    accuracy_test = knn_classifier.calculate_accuracy(test_data, test_label)
    print "==> KNN test accuracy: %.02f%%" % (accuracy_test*100.0)


def main(args):

    # Load Training List
    dataset = load_song_data.Track_dataset()
    dataset.prune(args.nclique, rseed=args.rseed_xval)
    train_list = dataset.get_tracks_train()
    test_list = dataset.get_tracks_test()

    ##
    # Count tracks and cliques
    def count_cliques(track_list):
        cliqueset = {s.clique for s in track_list}
        print "%d cliques found in %d tracks." % (len(cliqueset), len(track_list))
    print "Training set: ", 
    count_cliques(train_list)
    print "Test set: ", 
    count_cliques(test_list)

    ##
    # Set up feature extractors
    if args.features == 'timbre': 
        print "-- using averaged timbre features --"
        featureExtractor = feature_util.averageTimbreFeatureExtractor
    if args.features == 'combo': 
        print "-- using combo features --"
        featureExtractor = feature_util.combo_feature_extractor
    if args.features == 'combo2': 
        print "-- using combo2 features --"
        featureExtractor = feature_util.combo2_feature_extractor
    if args.features == 'comboPlus': 
        print "-- using comboPlus features --"
        featureExtractor = feature_util.comboPlus_feature_extractor

    pairFeatureExtractor = feature_util.make_subtractivePairFeatureExtractor(featureExtractor, 
                                                                             # take_abs=False)
                                                                             take_abs=True)

    weights, xform = None, None
    if args.do_logistic:
        weights, xform = run_logistic(train_list, test_list,
                                       pairFeatureExtractor,
                                       reg=args.reg,
                                       rseed=args.rseed_pairs,
                                       rstrength=args.rstrength,
                                       verbose=1,
                                       pre_mode=args.preprocess,
                                       npca=args.npca)
        ##
        # Plot weight vector
        #
        if args.do_plot:
            figure(1).clear()
            bar(range(weights.size), abs(weights.flatten()), width=0.9, align='center')
            # bar(range(weights.size), weights.flatten(), width=0.9, align='center')
            xlim(0,weights.size)
            xlabel("Index ($i$)")
            ylabel("$|Weight_i|$")
            show()

            sfname = "_".join(sys.argv)
            savefig(os.path.join(args.outdir,sfname)+".png")

        print ""
        print "==> Weight vector (feature) dimension: %d" % weights.size
      
    if args.do_knn:  
        ##
        # Run KNN
        # If logistic was run, will use the data scaler and weights
        # as an improved metric
        #
        test_knn(train_list, test_list, featureExtractor, 
                 weights=weights, 
                 transform=xform, 
                 k=args.k,
                 metric=args.knnMetric)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Cover Song Identifier")

    #parser.add_argument('-t', '--ntrain', dest='ntrain', type=int, default=5000)
    #parser.add_argument('-e', '--ntest', dest='ntest', type=int, default=1000)
    parser.add_argument('-c', '--nclique', dest='nclique', type=int, default=300)

    parser.add_argument('--logistic', dest='do_logistic', action='store_true')
    parser.add_argument('--knn', dest='do_knn', action='store_true')

    ##
    # Preprocessing mode and parameters
    parser.add_argument('-p', '--pre', dest='preprocess', 
                        default='none',
                        choices=['none', 'scale', 'whiten', 'both', 'pca'])
    parser.add_argument('--pca', dest='npca', type=float,
                        default=100)

    # Options for logistic classifier
    parser.add_argument('-r', '--reg', dest='reg', metavar='regularization', 
                        default='l2',
                        choices=['l1','l2'])
    # Regularization strength: higher is stronger
    parser.add_argument('--rstrength', dest='rstrength', default=1.0, type=float)

    # Options for KNN classifier
    parser.add_argument('-k', dest='k', default=5, type=int)
    parser.add_argument('--knnMetric', dest='knnMetric', default='euclidean')

    # Random seeds
    # for cross-validation (train|test partition)
    # and pair selection (logistic classifier)
    parser.add_argument('--seed_xval', dest='rseed_xval', type=int, default=10)
    parser.add_argument('--seed_pairs', dest='rseed_pairs', type=int, default=10)

    # Test fracton
    parser.add_argument("--test_fraction", dest='test_fraction', type=float, default=0.33)

    # select features
    parser.add_argument('-f', '--features', dest='features', 
                        default='combo',
                        choices=['timbre', 'combo', 'combo2', 'comboPlus'])

    # Enable plotting
    parser.add_argument('--plot', dest='do_plot', action='store_true')
    parser.add_argument('--outdir', dest='outdir', default="output")

    args = parser.parse_args()
    if not args.do_logistic and not args.do_knn:
        args.do_knn = True
        args.do_logistic = True

    # Make output directory
    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    try:
        main(args)
    except Exception as e:
        print repr(e)
        pdb.post_mortem()