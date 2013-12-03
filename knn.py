import load_song_data
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNearestNeighbor(object):

	def __init__(self, weights, training_data, training_labels, 
                 k=5,
                 metric='euclidean',
                 dWeight='uniform'):
		self.knc = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=dWeight)
		self.weights = np.array(weights)
		self.training_data = np.array(training_data)
		self.training_labels = np.array(training_labels)
		self.knc.fit(self.training_data * weights, self.training_labels)

	def calculate_accuracy(self, test_tracks, labels):
		test_data = np.array(test_tracks) * self.weights
		return self.knc.score(test_data, labels)

