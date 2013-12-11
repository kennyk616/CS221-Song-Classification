#!/bin/bash

# Run the cross-validation script on a set of parameters
# reads parameters from each line of the given file
# usage: run_list.sh <file> <n>

COUNTER=0

echo ""
echo "==== Reading configuration from $1 ===="
while read line;
do
	[[ "$line" =~ ^#.*$ ]] && continue	
	[[ "$line" == "" ]] && continue	
	echo "PARAMS: $line"
	COUNTER=$[$COUNTER + 1]
done < $1
echo "==== read $COUNTER parameter sets, $2 cross-validation trials each ===="
echo ""

while read line;
do
	# Skip comments and blank lines
	[[ "$line" =~ ^#.*$ ]] && continue	
	[[ "$line" == "" ]] && continue	

	# Run cross-validation script
	echo "PARAMS: $line"
	./xval.sh $2 "$line"
done < $1