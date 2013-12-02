from pylab import *

from sklearn import preprocessing, decomposition

##
# Template class
#
class IdentityTransformer(object):
    def __init__(self):
        pass

    def data_toarray(self, data_dict):
        X_mat = array([x for x,y in data_dict.values()])
        y_list = array([y for x,y in data_dict.values()])
        return X_mat, y_list


    def fit(self, X, **kwargs):
        pass

    def transform(self, X, copy=False):
        if copy: return X.copy()
        else: return X



class StandardScaler(IdentityTransformer):
    def __init__(self, **kwargs):
        self.scaler = preprocessing.StandardScaler(**kwargs)

    def fit(self, X):
        """Create a StandardScaler object to transform training and test data."""
        self.scaler.fit(X) # compute mean and std for each feature

    def transform(self, X, copy=False):
        """In-place (or copy) transform of dataset."""
        X = self.scaler.transform(X, copy=copy)
        return X



class PCAWhitener(IdentityTransformer):
    def __init__(self, copy=True, n_components=None):
        if n_components > 1: n_components = int(n_components)
        self.whitener = decomposition.PCA(n_components=n_components,
                                          whiten=True, copy=copy)

    def fit(self,X):
        self.whitener.fit(X)

    def transform(self, X):
        return self.whitener.transform(X)


##
# Implements a sequential transform:
# whiten data, then scale means to zero (with StandardScaler)
#
class ScaleThenWhiten(IdentityTransformer):
    def __init__(self, copy=True, n_components=None):
        self.scaler = StandardScaler()
        self.whitener = PCAWhitener(n_components=n_components,
                                    copy=True)

    def fit(self, X):
        self.whitener.fit(X)
        X_scaled = self.whitener.transform(X)
        self.scaler.fit(X_scaled)

    def transform(self, X):
        return self.scaler.transform(self.whitener.transform(X))