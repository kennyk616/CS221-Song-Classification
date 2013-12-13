import os
import sys

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
tld = sys.argv[1]

# Make three levels of alphabet soup
os.chdir(tld)

for l in alphabet:
	os.mkdir(l)
	os.chdir(l)
	for l in alphabet:
		os.mkdir(l)
		os.chdir(l)
		for l in alphabet:
			os.mkdir(l)
		os.chdir('..')
	os.chdir('..')