CS221 Project: Cover Song Classification
========================================
Ian Tenney
Ken Kao
December 12, 2013

Running the Project
===================

Because our project focused on testing our classifier on a development set, and because we were not able to achieve production-level acccuracy, we did not build a direct query system. Instead, the classifier loads data from the `/MSD-SHS/` directory (symlinked to a separate filesystem) and partitions into a training and a test set using a seeded random sample (below).

To run our classifier, type:
`python main.py <params>`

Parameters are passed as UNIX-style flags, and specify all the parameters that we used during testing to fine-tune the data selection, feature extraction, pre-processing, and the logistic regression, metric learning, and KNN classification stages of our system. The parameters described below are those used in testing; some others exist in the code but were not used.

Data:
-----

+ `-c` : number of cliques to use; larger cliques are chosen first
+ `--test_fraction` : approximate portion of the full dataset to use as the testing/development set
+ `-f` : features to use: `combo` or `comboPlus` *(other features may be available, but not used in our final results)*
+ `--seed_xval` : random seed for train | test partitioning
+ `--seed_pairs` : random seed for sampling negative pairs (used by pairwise binary classifier)

Preprocessing:
--------------

+ `-p` : preprocessing mode: `none`, `scale`, `whiten`, or `pca`
+ `--pca` : if using `-p pca`, specifies the number of components to use

Metric Learning (LMNN):
-----------------------

+ `--LMNN` : use the LMNN algorithm to learn a Mahalanobis matrix
+ `--lmnnMu` : inverse regularization strength for the LMNN algorithm

Logistic Regression:
--------------------

+ `--logistic` : use pairwise binary logistic regression
+ `-r` : regularization type (`l1` or `l2`)
+ `--rstrength` : regularization strength (higher is stronger)
+ `--noTestLogistic` : disable running logistic classifier on test set, for faster runs for KNN testing

K Nearest Neighbors:
--------------------

+ `--knn` : use K Nearest Neighbors (KNN) classifier, with optional scaling from learned weights, or Mahalanobis metric from LMNN
+ `-k` : k-value for KNN
+ `--knn_relax_n` : number of candidate cliques to consider for "relaxed" testing

Misc:
-----

+ `--plot` : plot feature vectors and clique size distributions, saved as .png files in `./output/`


As an example, to run with `comboPlus` features and whitening on 50 cliques with k=7, use:
`python main.py -c 50 -f comboPlus -p whiten --logistic -r l1 --rstrength 100.0 --knn -k 7 --knn_relax_n 5`


Testing Methodology
===================

To obtain our analysis, we ran our system many times with different parameters. Additionally, we changed the data partitioning seed (`--seed_xval`) to perform cross-validation for more robust results. This is a tedious process to run manually, and so we include a series of bash scripts to automate the process without making major changes to our python codebase:

`xval.sh <n> "params"` : will run our program repeatedly with a different seed, and output concatenated results to the `./results/` directory. n refers to the number of cross-validation runs to perform.

`run_list.sh <file> <n>` : will parse a text file containing lists of parameters as they would be passed to `main.py`, one list per line, and run the cross-validation script (`xval.sh`) on each.

`parse_set.sh <field_glob> <fname_glob_VAR> <VAR1> [<VAR2> <VAR3> ...]` : this will parse output files in `./results/` by matching the output lines to `field_glob` in files matching `fname_glob`. The string `VAR` in the file match string will be substituted by each following argument, providing array-like functionality. Output is in comma-separated format.

For example, to extract data from a variety of clique counts, and those runs that used combo features and the standard scaler:

	>> ./parse_set.sh "KNN relax test" c_VAR*scale*combo_*100.0 25 35 50 75 100 200 300
	# Field: <KNN relax test> :: match: *c_25*scale*combo_*100.0*,35.80,35.19,32.10,29.01,30.86
	# Field: <KNN relax test> :: match: *c_35*scale*combo_*100.0*,23.76,19.80,23.27,27.72,28.22
	# Field: <KNN relax test> :: match: *c_50*scale*combo_*100.0*,18.70,17.18,18.70,17.56,17.56
	# Field: <KNN relax test> :: match: *c_75*scale*combo_*100.0*,15.50,13.45,10.53,14.62,14.62
	# Field: <KNN relax test> :: match: *c_100*scale*combo_*100.0*,11.51,12.47,9.35,9.83,12.23
	# Field: <KNN relax test> :: match: *c_200*scale*combo_*100.0*,8.03,9.55,7.88,7.88,6.21
	# Field: <KNN relax test> :: match: *c_300*scale*combo_*100.0*,6.51,4.88,4.42,5.35,4.65

