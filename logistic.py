from pylab import *
# from scipy.stats import futil
# from scipy.sparse.csgraph import _validation
from sklearn import linear_model, preprocessing
from sklearn import metrics


import random

# Data transformer classes
import transform

##
# Global parameters
#
Y_SAME = 1
Y_DIFF = -1

##
# Template class
#
class PairwiseSongClassifier(object):
    def __init__(self, dataTransformer):
        self.transformer = dataTransformer

    def getMetric():
        """
        Return a metric function that takes two songs
        and returns the distance between them.
        """
        raise Exception("Not implemented yet.")

    def pairClassify(s1,s2):
        """
        Returns true if s1 and s2 belong to the same clique,
        false otherwise.
        """
        raise Exception("Not implemented yet.")

    ##
    # Confusion Matrix
    def confusion_matrix(self, X, y, verbose=True, normalize=True):
        y_pred = self.predict(X)
        cmat = metrics.confusion_matrix(y, y_pred)

        if normalize: 
            cmat = cmat /(1.0*sum(cmat))

        if verbose:
            print "Confusion Matrix:"
            print "        neg    | pos"
            print "  neg | %.04g | %.04g " % (cmat[0,0], cmat[0,1])
            print "  pos | %.04g | %.04g " % (cmat[1,0], cmat[1,1])

        return cmat

    def predict(self, X):
        raise Exception("Not implemented yet.")

    def pair_dataset(self, songList, pairFeatureExtractor, 
                     trimRatio=1.0,
                     rseed=None, verbose=False):
        """
        Generate a dataset of pairwise comparisons from a
        list of songs, using the pairFeatureExtractor(s1,s2)
        function.

        Returns a dictionary of {(id1,id2):(x,y)}
        where x is a feature vector (ndarray) and
        y is Y_SAME (+1) for same clique, Y_DIFF (-1) for different.
        """
        def getKey(i,j):
            s1 = songList[i]
            s2 = songList[j]
            return (s1.id, s2.id)

        ##
        # Generate lists of same and different-song pairs
        same_clique = []
        diff_clique = []
        for i,s1 in enumerate(songList):
            # avoid duplicates
            for j,s2 in enumerate(songList[:i]):
                if s1.id == s2.id: continue

                if s1.clique == s2.clique:
                    same_clique.append((i,j))
                else:
                    diff_clique.append((i,j))

        ##
        # Trim the different-clique list to a subset
        # with len(same_clique)*trimRatio pairs
        if trimRatio > 0:
            samples_diff = int(trimRatio * len(same_clique))
            rng = random.Random()
            rng.seed(rseed)
            diff_clique = rng.sample(diff_clique, samples_diff)

        ##
        # Extract features
        same_dict = {getKey(i,j):(pairFeatureExtractor(songList[i],songList[j]),Y_SAME) for (i,j) in same_clique}
        diff_dict = {getKey(i,j):(pairFeatureExtractor(songList[i],songList[j]),Y_DIFF) for (i,j) in diff_clique}

        ##
        # Combine and return
        outdict = same_dict
        outdict.update(diff_dict)
        return outdict


class LogisticClassifier(PairwiseSongClassifier):
    def __init__(self, reg='l2', rstrength=1.0, dataTransformer=transform.IdentityTransformer):
        super(LogisticClassifier, self).__init__(dataTransformer)
        self.engine = linear_model.LogisticRegression(penalty=reg,
                                                      C = 1.0/rstrength,
                                                      dual=False)

    def fit(self, X_mat, y_list):
        self.trainset = (X_mat,y_list)
        self.engine.fit(X_mat, y_list)

        return self.engine.score(X_mat, y_list)

    def test(self, X_mat, y_list):
        return self.engine.score(X_mat, y_list)

    def predict(self, X):
        return self.engine.predict(X)

    def getMetric(self, pairFeatureExtractor):
        def metric(s1,s2):
            """Returns zero for same song, 1 for different."""
            X = pairFeatureExtractor(s1,s2)
            X.reshape( (1,len(X)) )
            predict = self.engine.predict(X)
            if predict == Y_SAME: return 0
            else: return 1

        return metric

    def getWeights(self):
        """Return the internal weight vector from them logistic classifier."""
        return array(self.engine.coef_).copy()



if __name__ == '__main__':
    pass