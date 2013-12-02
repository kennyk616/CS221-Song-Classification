from pylab import *

##
# Helper functions for processing data and extracting features
#

def make_subtractivePairFeatureExtractor(featureExtractor, take_abs=False):

    def pairFeatureExtractor(s1,s2):
        f1 = featureExtractor(s1)
        f2 = featureExtractor(s2)
        if take_abs: return np.abs(f1-f2)
        else: return (f1 - f2)

    return pairFeatureExtractor

# n refers to how many samples we want to take from the feature
def sample_segments_feature(feature, n=10):
    slice_size = max(len(feature)/n,1)
    sample_features = feature[0::slice_size]
    out = zeros((n,feature.shape[1]))
    out[:len(sample_features),:] = sample_features[:n]
    # return sample_features[0:n]
    return out

def sample_vector_feature(feature, n=10):
    feature = feature.reshape((len(feature),1))
    return sample_segments_feature(feature, n=n)

# features used: loudness, duration, loudness_max, 
# segments_pitches, segments_timbre, tempo
def combo_feature_extractor(track, ns=10):

    # One-dimensional features
    loudness = track.get_loudness()
    duration = track.get_duration()
    tempo = track.get_tempo()

    # Sampled vector features
    sample_segment_pitches = sample_segments_feature(track.get_segments_pitches(), n=ns).flatten()
    sample_segment_timbre = sample_segments_feature(track.get_segments_timbre(), n=ns).flatten()

    track.close() # filesystem garbage collection

    # one big frickin vector
    return concatenate(([loudness],[duration],[tempo],
                     sample_segment_pitches,
                     sample_segment_timbre))

# like combo features, but super-sized!
def combo2_feature_extractor(track, ns=10):
    # One-dimensional objective features
    loudness = track.get_loudness()
    duration = track.get_duration()
    tempo = track.get_tempo()
    key = track.get_key()
    mode = track.get_mode()
    danceability = track.get_danceability()
    energy = track.get_energy()

    # Sampled vector features
    sample_segment_pitches = sample_segments_feature(track.get_segments_pitches(), n=ns).flatten()
    sample_segment_timbre = sample_segments_feature(track.get_segments_timbre(), n=ns).flatten()
    sample_segment_loudness_max = sample_vector_feature(track.get_segments_loudness_max(), n=ns).flatten()

    track.close() # filesystem garbage collection

    # one big frickin vector
    return concatenate(([loudness],[duration],[tempo],
                       [key],[mode],[danceability],[energy],
                       sample_segment_pitches,
                       sample_segment_timbre,
                       sample_segment_loudness_max))


def averageTimbreFeatureExtractor(track):
    t_segments = track.get_segments_timbre()
    track.close()
    return np.mean(t_segments, axis=0)


def get_feature_and_labels(feature_extractor, tracks):
	data = []
	label = []
	for track in tracks:
		data.append(feature_extractor(track))
		label.append(track.clique)
	return data, label
