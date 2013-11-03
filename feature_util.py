##
# Helper functions for processing data and extracting features
#

def make_subtractivePairFeatureExtractor(featureExtractor):

    def pairFeatureExtractor(s1,s2):
        f1 = featureExtractor(s1)
        f2 = featureExtractor(s2)
        return (f1 - f2)

    return pairFeatureExtractor

# n refers to how many samples we want to take from the feature
def sample_segments_feature(feature, n=10):
    sample_features = feature[0::len(feature)/n]
    return sample_features[0:n]

# features used: loudness, duration, loudness_max, 
# segments_pitches, segments_timbre, tempo
def combo_feature_extractor(track):
    loudness = track.get_loudness()
    duration = track.get_duration()
    sample_segment_pitches = sample_segments_feature(track.get_segments_pitches())
    sample_segment_timbre = sample_segments_feature(track.get_segments_timbre())
    sample_segment_timbre = [t for seg in sample_segment_timbre for t in seg]
    tempo = track.get_tempo()
    return [loudness] + [duration] + sample_segment_pitches + sample_segment_timbre + [tempo]




def get_feature_and_labels(feature_extractor, tracks):
	data = []
	label = []
	for track in tracks:
		data.append(feature_extractor(track))
		label.append(track['clique_name'])
	return data, label
