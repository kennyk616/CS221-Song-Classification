import load_song_data
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNearestNeighbor(object):

    def __init__(self, weights, training_data, training_labels, 
                 k=5,
                 metric='euclidean',
                 dWeight='uniform',
                 metricKW={}):
        self.knc = KNeighborsClassifier(n_neighbors=k, 
                                        metric=metric, 
                                        weights=dWeight,
                                        **metricKW)
        print "KNN: using \'%s\' metric with params %s" % (metric, str(metricKW.keys()))
        self.weights = np.array(weights)
        self.training_data = np.array(training_data)
        self.training_labels = np.array(training_labels)
        self.knc.fit(self.training_data * weights, self.training_labels)
        self.labels = sorted(set(self.training_labels))

    def calculate_accuracy(self, test_tracks, labels):
        test_data = np.array(test_tracks) * self.weights
        return self.knc.score(test_data, labels)

    # def calculate_accuracy_relax_old(self, test_tracks, test_labels, n):
    #     test_data = np.array(test_tracks) * self.weights
    #     predict_probability = self.knc.predict_proba(test_tracks)

    #     predict = self.knc.predict(test_tracks)
        
    #     nCorrect = 0
    #     nWrong = 0
        
    #     for idx, prob in enumerate(predict_probability):
    #         indecies = sorted(range(len(prob)), key=lambda i:prob[i])[-1*n:]
    #         #print indecies
    #         correct = False
    #         for i in indecies:
    #             if self.labels[i] == test_labels[idx]:
    #                 correct = True
    #             if self.labels[i] != predict[idx]:
    #                 print "idx = ", idx
    #                 print "label_real = ", test_labels[idx]
    #                 print "label_predict = ", predict[idx]
    #                 print "label_predict_proba = ", self.labels[i]
    #         if correct == True:
    #             nCorrect += 1
    #         else:
    #             nWrong += 1

    #     print "correct = ", nCorrect
    #     print "wrong = ", nWrong
    #     return nCorrect*1.0/(nCorrect+nWrong)

    # when knn returns a list of neighboring labels, if the actual label is wihin the 
    # top n labels, this classifier will consider it correct. 
    def calculate_accuracy_relax(self, data, labels, n):
        data = np.array(data) * self.weights
        y_pred = self.knc.predict_proba(data)
        y_true = np.array(labels)

        classlabels = self.knc.classes_ # forbidden! but I'm lazy
        # print "DEBUG: %d classes found" % len(classlabels)

        def check_found(probas, label, n):
            pl = zip(probas, classlabels) # (p, label) pairs
            pl_n = [l for p,l in sorted(pl)[-n:]] # top n labels
            return (1 if label in pl_n else 0) # 1 for match

        scorelist = [check_found(ps,yt,n) for ps,yt in zip(y_pred, y_true)] # compare each row
        return sum(scorelist)/float(len(scorelist)) # compute percent correct


    # def calculate_accuracy_predict_old(self, test_tracks, test_labels):
    #     test_data = np.array(test_tracks) * self.weights
    #     predict = self.knc.predict(test_tracks)
    #     nCorrect = 0
    #     nWrong = 0
    #     for idx, val in enumerate(predict):
    #         if val == test_labels[idx]:
    #             nCorrect +=1
    #         else:
    #             nWrong += 1
    #     print "correct = ", nCorrect
    #     print "wrong = ", nWrong
    #     return nCorrect*1.0/(nCorrect+nWrong)

    def calculate_accuracy_predict(self, data, labels):
        data = np.array(data) * self.weights
        y_pred = self.knc.predict(data)
        y_true = np.array(labels)

        accuracy = sum(y_pred == y_true) / float(len(y_pred))
        return accuracy
