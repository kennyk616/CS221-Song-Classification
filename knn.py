import sys
import os
from sklearn import KNeighborsClassifier
import numpy as np

lib_path = os.path.abspath('./lib')
sys.path.append(lib_path)
import hdf5_getters


def main():
	ntracks = int(sys.argv[1])

    json_data_paths=open('./MSD-SHS/shs_dataset_train/shs_dataset_train.trackpaths.json')
    json_data=open('./MSD-SHS/shs_dataset_train/shs_dataset_train.tracks.json')
    track_paths = json.load(json_data_paths) #place holder for the dictionary of paths
    track_info = json.load(json_data) #place holder for track info


    k = 5
	knc = KNeighborsClassifier(k)

if __name__ == '__main__':
	main()

