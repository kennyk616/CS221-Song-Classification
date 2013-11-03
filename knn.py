import load_song_data
from sklearn import KNeighborsClassifier
import numpy as np

lib_path = os.path.abspath('./lib')
sys.path.append(lib_path)
import hdf5_getters


def main():
	ntracks = int(sys.argv[1])

    song_data = load_song_data.Song_data()
    track_paths = song_data.get_track_paths_train
    track_info = song_data.get_info_train()
    


    k = 5
	knc = KNeighborsClassifier(k)

if __name__ == '__main__':
	main()

