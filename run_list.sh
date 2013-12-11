#!/bin/bash

# Run the cross-validation script on a set of parameters
# reads parameters from each line of the given file
# usage: run_list.sh <file> <n>

# for line in $(cat $1)
while read line;
do
	[[ "$line" =~ ^#.*$ ]] && continue
	echo "PARAMS: $line"
	./xval.sh $2 "$line"
done < $1