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
                 pre_xform=transform.IdentityTransformer(),
                 npca=None,
                 reg='l2',
                 rstrength=1.0):
    # Initialize classifier
    classifier = logistic.LogisticClassifier(reg=reg, 
                                             rstrength=rstrength,
                                             dataTransformer=pre_xform)

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


def run_LMNN(train_list, featureExtractor, pre_xform, 
             diagonal=False, mu=0.5,
             tempdir='temp/',
             outdir='temp/',
             libpath='lib/mLMNN2.4/'):
    """Call MATLAB to run the Large Margin Nearest Neighbor (LMNN) algorithm to learn a
    Mahalanobis matrix."""

    t0 = time.time()
    print "Loading training set...",
    data, label = feature_util.get_feature_and_labels(featureExtractor, train_list)
    print " completed in %d seconds." % int(time.time() - t0)

    # Unsupervised preprocessing (PCA, etc.)
    t0 = time.time()
    print "Preprocessing data...",
    if pre_xform != None: 
        pre_xform.fit(data)
        data = pre_xform.transform(data)
    print " completed in %.03g seconds." % (time.time() - t0)

    ##
    # Save data for MATLAB to use
    from scipy import io
    params = {'diagonal':diagonal, 'mu':mu}
    mdict = {'X':data, 'y':label, 'params':params}
    outfile = os.path.join(tempdir,'LMNN-data.temp.mat')
    print ("Creating temp file \'%s\'" % outfile),
    io.savemat(outfile, mdict)
    print " : %.02g MB" % (os.path.getsize(outfile)/(2.0**20))

    ##
    # Invoke MATLAB from the command line
    # matlab -nodisplay -nojvm -r "cd('lib/mLMNN2.4/'); run('setpaths.m'); cd('main'); load('temp/LMNN.temp.mat'); [L,Det] = lmnn2(X',y'); save('temp/dummy.mat', 'L', 'Det', '-v6'); quit;"
    Lfile = os.path.join(outdir,'LMNN-res.temp.mat')
    logfile = os.path.join(outdir, 'LMNN.log')
    # call_base = ("""matlab -nodisplay -nojvm -logfile '%s'""" % logfile)
    call_base = """matlab -nodisplay -nojvm -r"""
    
    idict = {'libpath':libpath, 'outfile':outfile, 'Lfile':Lfile}
    code = """cd '%(libpath)s'; run('setpaths.m'); cd '../../'; load('%(outfile)s'); [L,Det] = lmnn2(X',y', 'diagonal', params.diagonal, 'mu', params.mu); save('%(Lfile)s', 'L', 'Det', '-v6'); quit;""" % idict
    # callstring = """%s \"%s\"""" % (call_base, code)
    import shlex
    # callstring = shlex.split(call_base) + ["\"%s\"" % code]
    callstring = shlex.split(call_base) + [code]


    # callstring = """/bin/echo "hello world" """ # DEBUG

    import subprocess as sp
    t0 = time.time()
    print "Invoking MATLAB with command:\n>> %s" % (call_base + (" \"%s\"" % code))
    print " logging results to %s" % logfile
    with open(logfile, 'w') as lf:
        sp.call(callstring, stdout=lf, stderr=sys.stderr)
    print "LMNN optimization completed in %.02g minutes." % ((time.time() - t0)/60.0)
    print " results logged to %s" % logfile

    ##
    # PLACEHOLDER
    # L = eye(data.shape[1]) # Identity Matrix
    Ldict = io.loadmat(Lfile)
    L = Ldict['L']

    from sklearn import neighbors
    print "Mahalanobis matrix: \n  %d dimensions\n  %d nonzero elements" % (L.shape[0], L.flatten().nonzero()[0].size)
    metric = neighbors.DistanceMetric.get_metric('mahalanobis', VI=L)
    return metric, pre_xform


def test_knn(train_list, test_list, featureExtractor, 
             k = 5,
             metric='euclidean',
             weights=None, pre_xform=None):

    data, label = feature_util.get_feature_and_labels(featureExtractor, train_list)

    if weights == None:
        weights = ones(len(data[0])) # DUMMY

    # Transform data (preprocessor)
    if pre_xform != None: data = pre_xform.transform(data)

    knn_classifier = knn.KNearestNeighbor(weights, data, label, k=k, metric=metric)
    print "Running KNN with k=%d and %s metric" % (k, metric)
    accuracy = knn_classifier.calculate_accuracy(data, label)
    print "==> KNN training accuracy: %.02f%%" % (accuracy*100.0)

    test_data, test_label = feature_util.get_feature_and_labels(featureExtractor, test_list)

    # Transform data (preprocessor)
    if pre_xform != None: test_data = pre_xform.transform(test_data)

    accuracy_test = knn_classifier.calculate_accuracy(test_data, test_label)
    print "==> KNN test accuracy: %.02f%%" % (accuracy_test*100.0)


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
            bar(c, h, width=1.0, align='center')
            axvline(1.5, color='k', alpha=0.5, linestyle='--', linewidth=1.5)
            xlabel("Clique size (tracks)")
            ylabel("Frequency")
            title("PDF")

            cdf = cumsum(h[::-1])

            subplot(1,3,2)
            bar(c[::-1], cdf/float(cdf[-1]), width=1.0, align='center', color='r')
            axvline(1.5, color='k', alpha=0.5, linestyle='--', linewidth=1.5)
            xlabel("Clique size (tracks)")
            ylabel("Cumulative Fraction (size < s)")
            title("CDF")

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
            title("CDF, inverted")

            suptitle(plotTitle)

            show()
            savefig(os.path.join(args.outdir,sfname)+".png")

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
    metric = args.knnMetric


    ##
    # Run Large Margin Nearest Neighbors to generate a Mahalanobis matrix
    #
    if args.do_LMNN:
        weights = None
        metric, pre_xform = run_LMNN(train_list, featureExtractor, 
                                     pre_xform,
                                     diagonal=args.lmnnDiagonal,
                                     mu=args.lmnnMu)

    ##
    # Run logistic regression to set feature weights
    #
    if args.do_logistic:
        weights, pre_xform = run_logistic(train_list, test_list,
                                       pairFeatureExtractor,
                                       reg=args.reg,
                                       rseed=args.rseed_pairs,
                                       rstrength=args.rstrength,
                                       verbose=1,
                                       pre_xform=pre_xform)
        ##
        # Plot weight vector
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
                 dWeight=args.knnDWeight)


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

    ##
    # Options for logistic classifier
    parser.add_argument('--logistic', dest='do_logistic', action='store_true')
    parser.add_argument('-r', '--reg', dest='reg', metavar='regularization', 
                        default='l2',
                        choices=['l1','l2'])
    # Regularization strength: higher is stronger
    parser.add_argument('--rstrength', dest='rstrength', default=1.0, type=float)

    ##
    # Options for KNN classifier
    parser.add_argument('--knn', dest='do_knn', action='store_true')
    parser.add_argument('-k', dest='k', default=5, type=int)
    parser.add_argument('--knnMetric', dest='knnMetric', default='euclidean')
    parser.add_argument('--knnDWeight', dest='knnDWeight',
                        default='uniform',
                        choices=['uniform', 'distance'])



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