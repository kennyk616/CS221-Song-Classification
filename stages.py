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
                 rstrength=1.0,
                 trim_train=1.0,
                 intercept_scaling=1.0,
                 do_test=True):
    # Initialize classifier
    classifier = logistic.LogisticClassifier(reg=reg, 
                                             rstrength=rstrength,
                                             intercept_scaling=intercept_scaling,
                                             dataTransformer=pre_xform)

    def load_data(track_list, fit=False, trim=1.0):
        t0 = time.time()
        if verbose: 
            print "\n===================="
            print "Generating pairwise dataset..."
        data_dict = classifier.pair_dataset(track_list, 
                                       pairFeatureExtractor, 
                                       rseed=rseed,
                                       trimRatio=trim,
                                       verbose=True)
        data = classifier.transformer.data_toarray(data_dict)
        if verbose: print "Completed in %.02f s" % (time.time() - t0)

        print "Preprocessing data..."
        if fit: classifier.transformer.fit(data[0]) # calculate preprocessor parameters
        data = (classifier.transformer.transform(data[0]), data[1]) # preprocess

        return data

    train_data = load_data(train_list, fit=True, trim=trim_train)

    # Run classifier and show output
    t0 = time.time()
    if verbose: 
        print "Training logistic classifier on %d song pairs..." % len(train_data[0])
        print "Train data dimensions: %s" % str(train_data[0].shape)
    score = classifier.fit(*train_data)
    if verbose: print "Completed in %.02f s" % (time.time() - t0)
    if verbose: print "==> Logistic training accuracy: %.02f%%" % (score*100.0)

    weights = classifier.getWeights()
    if verbose >= 2: print "Weights: %s" % str(weights)

    ##
    # Run tests of logistic classifier
    # can skip this for debugging KNN
    if do_test:
        print "Logistic: TRAIN set"
        train_X, train_y = train_data[0], train_data[1]
        classifier.confusion_matrix(train_X, train_y, verbose=True, normalize=True)

        ##
        # Evaluate on test set
        # TO-DO: do a proper cross-validation here
        test_data = load_data(test_list, fit=False, trim=-1)
        if verbose: print "Testing logistic classifier on %d song pairs..." % len(test_data[0])
        score = classifier.test(*test_data)
        if verbose: print "Completed in %.02f s" % (time.time() - t0)
        if verbose: print "==> Logistic test accuracy: %.02f%%" % (score*100.0)

        print "Logistic: TEST set"
        test_X, test_y = test_data[0], test_data[1]
        classifier.confusion_matrix(test_X, test_y, verbose=True, normalize=True)

    return weights, classifier

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
    call_base = """matlab -nodisplay -nojvm -r"""
    
    idict = {'libpath':libpath, 'outfile':outfile, 'Lfile':Lfile}
    # code = """cd '%(libpath)s'; run('setpaths.m'); cd '../../'; load('%(outfile)s'); [L,Det] = lmnn2(X',y', 'diagonal', params.diagonal, 'mu', params.mu); save('%(Lfile)s', 'L', 'Det', '-v6'); quit;""" % idict
    code = """cd '%(libpath)s'; run('setpaths.m'); cd '../../'; load('%(outfile)s'); [L,Det] = lmnn2(X',y', 'diagonal', params.diagonal, 'mu', params.mu, 'obj', 0); save('%(Lfile)s', 'L', 'Det', '-v6'); quit;""" % idict
    import shlex
    callstring = shlex.split(call_base) + [code]

    import subprocess as sp
    t0 = time.time()
    print "Invoking MATLAB with command:\n>> %s" % (call_base + (" \"%s\"" % code))
    print " logging results to %s" % logfile
    with open(logfile, 'w') as lf:
        sp.call(callstring, stdout=lf, stderr=sys.stderr)
    print "LMNN optimization completed in %.02g minutes." % ((time.time() - t0)/60.0)
    print " results logged to %s" % logfile

    Ldict = io.loadmat(Lfile)
    L = Ldict['L']

    L2 = dot(L.T, L) ## TEST??? -> this seems to give more "correct" results.

    from sklearn import neighbors
    print "Mahalanobis matrix: \n  %d dimensions\n  %d nonzero elements" % (L.shape[0], L.flatten().nonzero()[0].size)
    # metric = neighbors.DistanceMetric.get_metric('mahalanobis', VI=L)
    # return metric, pre_xform
    return L, L2, pre_xform


def test_knn(train_list, test_list, featureExtractor, 
             k = 5, knn_relax_n=5,
             metric='euclidean',
             algorithm='auto',
             weights=None, pre_xform=None, dWeight='uniform', metricKW={}):

    train_data, train_label = feature_util.get_feature_and_labels(featureExtractor, train_list)

    # Transform data (preprocessor)
    if pre_xform != None: train_data = pre_xform.transform(train_data)

    if weights == None:
        # do this after PCA, to be sure dimension is correct...
        weights = ones(len(train_data[0])) # DUMMY

    knn_classifier = knn.KNearestNeighbor(weights, train_data, train_label, k=k, algorithm=algorithm,
                                          metric=metric, dWeight=dWeight, metricKW=metricKW)
    print "Running KNN with k=%d and %s metric" % (k, metric)
    accuracy = knn_classifier.calculate_accuracy(train_data, train_label)
    print "==> KNN training accuracy: %.02f%%" % (accuracy*100.0)

    test_data, test_label = feature_util.get_feature_and_labels(featureExtractor, test_list)

    # Transform data (preprocessor)
    if pre_xform != None: test_data = pre_xform.transform(test_data)

    accuracy_test = knn_classifier.calculate_accuracy(test_data, test_label)
    print "==> KNN test accuracy: %.02f%%" % (accuracy_test*100.0)

    ##
    # Test relaxed test accuracy (top n matches)
    #
    print "Checking KNN, relaxed to top %d membership" % knn_relax_n
    accuracy_relax = knn_classifier.calculate_accuracy_relax(train_data, train_label, knn_relax_n)
    print "==> KNN relax training accuracy: %.02f%%" % (accuracy_relax*100.0)

    accuracy_test_relax = knn_classifier.calculate_accuracy_relax(test_data, test_label, knn_relax_n)
    print "==> KNN relax test accuracy: %.02f%%" % (accuracy_test_relax*100.0)

    ##
    # DEBUG
    ##
    # print "Running KNN predict with k=%d and %s metric" % (k, metric)
    # accuracy_predict = knn_classifier.calculate_accuracy_predict(train_data, train_label)
    # print "==> KNN predict training accuracy: %.02f%%" % (accuracy_predict*100.0)

    # accuracy_test_predict = knn_classifier.calculate_accuracy_predict(test_data, test_label)
    # print "==> KNN predict test accuracy: %.02f%%" % (accuracy_test_predict*100.0)

