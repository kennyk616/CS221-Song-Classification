##
# Helper functions for processing data and extracting features
#

def make_subtractivePairFeatureExtractor(featureExtractor):

    def pairFeatureExtractor(s1,s2):
        f1 = featureExtractor(s1)
        f2 = featureExtractor(s2)
        return (f1 - f2)

    return pairFeatureExtractor