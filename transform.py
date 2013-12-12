from pylab import *

from sklearn import preprocessing, decomposition

##
# Template class
#
class IdentityTransformer(object):
    def __init__(self):
        self.is_fit = False

    def data_toarray(self, data_dict):
        X_mat = array([x for x,y in data_dict.values()])
        y_list = array([y for x,y in data_dict.values()])
        return X_mat, y_list

    def fit(self, X, **kwargs):
        self.is_fit = True

    def transform(self, X, copy=False):
        if copy: return X.copy()
        else: return X



class StandardScaler(IdentityTransformer):
    def __init__(self, **kwargs):
        self.scaler = preprocessing.StandardScaler(**kwargs)
        self.is_fit = False

    def fit(self, X):
        """Create a StandardScaler object to transform training and test data."""
        self.scaler.fit(X) # compute mean and std for each feature
        self.is_fit = True

    def transform(self, X, copy=False):
        """In-place (or copy) transform of dataset."""
        X = self.scaler.transform(X, copy=copy)
        return X



class PCAWhitener(IdentityTransformer):
    def __init__(self, copy=True, n_components=None):
        if n_components > 1: n_components = int(n_components)
        self.whitener = decomposition.PCA(n_components=n_components,
                                          whiten=True, copy=copy)
        self.is_fit = False

    def fit(self,X):
        self.whitener.fit(X)
        self.is_fit = True

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
        self.is_fit = False

    def fit(self, X):
        self.whitener.fit(X)
        X_scaled = self.whitener.transform(X)
        self.scaler.fit(X_scaled)
        self.is_fit = True

    def transform(self, X):
        return self.scaler.transform(self.whitener.transform(X))


class MahalanobisTransformer(IdentityTransformer):
    def __init__(self, L, pre_xform=None):
        self.pre_xform = pre_xform
        self.L = L
        self.is_fit = True

    def fit(self, X):
        self.is_fit = True

    def transform(self, X):
        if self.pre_xform != None:
            X = self.pre_xform.transform(X)

        # Apply transform to each row of X
        return dot(X, self.L.T)


class MetricDistanceTransformer(MahalanobisTransformer):
    def transform(self, X):
        Xt = super(MetricDistanceTransformer, self).transform(X)
        # return linalg.norm(X, axis=1) # norm each row
        rvec = sqrt(np.sum(Xt*Xt, axis=1)) # norm each row
        return reshape(rvec, (len(rvec),1))