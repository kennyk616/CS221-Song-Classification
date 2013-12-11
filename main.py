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

from stages import run_logistic, run_LMNN, test_knn

def main(args):

    # Load Training List
    dataset = load_song_data.Track_dataset()
    total_clique_counter, restricted_clique_counter = dataset.prune(args.nclique, rseed=args.rseed_xval)
    train_list = dataset.get_tracks_train()
    test_list = dataset.get_tracks_test()

    ##
    # Histogram clique distribution
    if args.do_plot:
        def histo_counter(c):
            sizes = c.values()
            bins = arange(min(sizes) - 0.5, max(sizes) + 0.5, 1.0)
            h, bins = histogram(sizes, bins)
            centers = 0.5*(bins[1:] + bins[:-1])
            return h, centers

        def plot_histo((h,c), plotTitle, savename):
            figure(1, figsize=(18,6)).clear()
            subplot(1,3,1)
            bar(c, h, width=1.0, align='center', color='c')
            axvline(1.5, color='k', alpha=0.5, linestyle='--', linewidth=1.5)
            xlabel("Clique size (tracks)")
            ylabel("Frequency")

            cdf = cumsum(h[::-1])

            subplot(1,3,2)
            bar(c[::-1], cdf/float(cdf[-1]), width=1.0, align='center', color='r')
            axvline(1.5, color='k', alpha=0.5, linestyle='--', linewidth=1.5)
            xlabel("Clique size (tracks)")
            ylabel("Cumulative Fraction (size < s)")

            subplot(1,3,3)
            # plot(cdf/float(cdf[-1]), c[::-1], linewidth=1.5, color='r', marker='o')
            # hlines(c[::-1], zeros((cdf.shape)), cdf/float(cdf[-1]), color='k')
            # vlines(cdf/float(cdf[-1]), zeros((c.shape)), c[::-1], color='m')
            # axhline(1.5, color='k', alpha=0.5, linestyle='--', linewidth=1.5)
            # xlabel("Cumulative Fraction")
            plot(cdf, c[::-1], linewidth=1.5, color='r', marker='o')
            hlines(c[::-1], zeros((cdf.shape)), cdf, color='k')
            vlines(cdf, zeros((c.shape)), c[::-1], color='m')
            axhline(1.5, color='k', alpha=0.5, linestyle='--', linewidth=1.5)
            ylim(ymin=0)
            xlabel("Cumulative Count")
            ylabel("Clique size (tracks)")

            suptitle(plotTitle)

            show()
            savefig(os.path.join(args.outdir,sfname)+".png")

        ## Turn this off: needs to only be run once for the whole set...
        ## otherwise just makes a lot of junk .pngs
        sfname = "_".join(sys.argv) + "-histo-all"
        plot_histo(histo_counter(total_clique_counter), "All Cliques", sfname)
        
        sfname = "_".join(sys.argv) + "-histo-res"
        plot_histo(histo_counter(restricted_clique_counter), ("Top %d Cliques" % args.nclique), sfname)


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

    ##
    # Set preprocessing transform
    # NOTE: pre_xform.fit() should be called by the learning algorithm
    # based on the particular dataset it requires.
    if args.preprocess == 'scale': pre_xform = transform.StandardScaler()
    elif args.preprocess == 'whiten': pre_xform = transform.PCAWhitener()
    elif args.preprocess == 'both': pre_xform = transform.ScaleThenWhiten()
    elif args.preprocess == 'pca': 
        print "PCA: using %g components" % args.npca
        pre_xform = transform.PCAWhitener(n_components=args.npca)
    else: pre_xform = transform.IdentityTransformer()
    print "Data preprocessor: %s" % str(type(pre_xform))

    weights = None
    metric = args.knnMetric # default
    metricKW = {}
    knnAlgorithm = "auto"
    L = None

    ##
    # Run Large Margin Nearest Neighbors to generate a Mahalanobis matrix
    #
    if args.do_LMNN:
        L, L2, pre_xform = run_LMNN(train_list, featureExtractor, 
                                pre_xform,
                                diagonal=args.lmnnDiagonal,
                                mu=args.lmnnMu)
        metric = 'mahalanobis'
        metricKW = {'VI':L2} # need L'L for metric -> KNN

    ##
    # Test a pairwise binary classifier using
    # a Mahalanobis metric, by hacking the logistic apparatus
    # and using a custom featureExtractor to pre-transform the data
    if args.do_binclass_maha and L != None:

        # metricPairFeatureExtractor = feature_util.make_metricPairFeatureExtractor(featureExtractor, L, pre_xform)
        pre_xform_maha = transform.MetricDistanceTransformer(L, pre_xform)

        weights, px = run_logistic(train_list, test_list,
                                   pairFeatureExtractor,
                                   reg=args.reg,
                                   rseed=args.rseed_pairs,
                                   rstrength=args.rstrength,
                                   verbose=1,
                                   pre_xform=pre_xform_maha)

    ##
    # Run logistic regression to set feature weights
    #
    if args.do_logistic:
        weights, logistic_classifier = run_logistic(train_list, test_list,
                                       pairFeatureExtractor,
                                       reg=args.reg,
                                       rseed=args.rseed_pairs,
                                       rstrength=args.rstrength,
                                       verbose=1,
                                       pre_xform=pre_xform,
                                       trim_train=args.pairTrimRatio,
                                       intercept_scaling=args.logisticIntercept,
                                       do_test=args.testLogistic)
        
        pre_xform = logistic_classifier.transformer

        ##
        # Plot weight vector
        if args.do_plot:
            figure(2, figsize=(10,10)).clear()
            subplot(2,1,1)
            bar(range(weights.size), abs(weights.flatten()), width=0.9, align='center')
            xlim(0,weights.size)
            xlabel("Index ($i$)")
            ylabel("$|Weight_i|$")

            subplot(2,1,2)
            bar(range(weights.size), weights.flatten(), width=0.9, align='center', color='g')
            xlim(0,weights.size)
            xlabel("Index ($i$)")
            ylabel("$|Weight_i|$")
            show()

            sfname = "_".join(sys.argv)
            savefig(os.path.join(args.outdir,sfname)+".png")

        print ""
        print "==> Weight vector (feature) dimension: %d" % weights.size


    ##
    # Run KNN
    # If logistic or LMNN was run, will use the preprocessor transform
    #
    if args.do_knn:  
        test_knn(train_list, test_list, featureExtractor, 
                 weights=weights, 
                 pre_xform=pre_xform, 
                 k=args.k,
                 metric=metric,
                 algorithm=knnAlgorithm,
                 dWeight=args.knnDWeight, metricKW=metricKW, knn_relax_n=args.knn_relax_n)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Cover Song Identifier")

    ##
    # Dataset options
    #
    #parser.add_argument('-t', '--ntrain', dest='ntrain', type=int, default=5000)
    #parser.add_argument('-e', '--ntest', dest='ntest', type=int, default=1000)
    parser.add_argument('-c', '--nclique', dest='nclique', type=int, default=300)
    parser.add_argument("--test_fraction", dest='test_fraction', type=float, default=0.33)
    parser.add_argument('-f', '--features', dest='features', 
                        default='combo',
                        choices=['timbre', 'combo', 'combo2', 'comboPlus'])

    ##
    # Random seeds
    # for cross-validation (train|test partition) and pair selection (logistic classifier)
    parser.add_argument('--seed_xval', dest='rseed_xval', type=int, default=10)
    parser.add_argument('--seed_pairs', dest='rseed_pairs', type=int, default=10)

    ##
    # Preprocessing mode and parameters
    parser.add_argument('-p', '--pre', dest='preprocess', 
                        default='none',
                        choices=['none', 'scale', 'whiten', 'both', 'pca'])
    parser.add_argument('--pca', dest='npca', type=float,
                        default=100)

    ##
    # Options for LMNN algorithm
    parser.add_argument('--LMNN', dest='do_LMNN', action='store_true')
    parser.add_argument('--lmnnDiagonal', dest='lmnnDiagonal', 
                        action='store_true') # default = false
    parser.add_argument('--lmnnMu', dest='lmnnMu', type=float, default=0.5)
    # Run logistic classifier afterwards
    parser.add_argument('--mahaClass', dest='do_binclass_maha', action='store_true')

    ##
    # Options for logistic classifier
    parser.add_argument('--logistic', dest='do_logistic', action='store_true')
    parser.add_argument('-r', '--reg', dest='reg', metavar='regularization', 
                        default='l2',
                        choices=['l1','l2'])
    # Regularization strength: higher is stronger
    parser.add_argument('--rstrength', dest='rstrength', default=1.0, type=float)
    # Intercept scaling: use higher values to compensate for regularization
    parser.add_argument('--intercept', dest='logisticIntercept', default=1.0, type=float)
    parser.add_argument('--trim', dest='pairTrimRatio', default=1.0, type=float)
    parser.add_argument('--noTestLogistic', dest='testLogistic', action='store_false') # by default, test logistic classifer on dev set

    ##
    # Options for KNN classifier
    parser.add_argument('--knn', dest='do_knn', action='store_true')
    parser.add_argument('-k', dest='k', default=5, type=int)
    parser.add_argument('--knnMetric', dest='knnMetric', 
                        default='euclidean') # overridden by LMNN
    parser.add_argument('--knnDWeight', dest='knnDWeight',
                        default='uniform',
                        choices=['uniform', 'distance'])
    parser.add_argument('--knn_relax_n', dest='knn_relax_n', default = 5, type=int)



    # Enable plotting
    parser.add_argument('--plot', dest='do_plot', action='store_true')
    parser.add_argument('--outdir', dest='outdir', default="output")

    parser.add_argument('--NORUN', dest='NORUN', action='store_true')

    args = parser.parse_args()
    
    if not args.do_logistic and not args.do_knn and not args.do_LMNN:
        args.do_knn = True
        args.do_logistic = True
    # if args.do_LMNN:
    #     args.do_knn = True

    if args.NORUN:
        print "NORUN mode: will not run any classifiers or learning"
        print "            will only load dataset and generate histograms (if enabled)"
        args.do_knn = False
        args.do_logistic = False
        args.do_LMNN = False

    # Make output directory
    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    try:
        main(args)
    except Exception as e:
        print repr(e)
        pdb.post_mortem()