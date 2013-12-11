#!/bin/bash

# Cross-Validation Script
# runs tests sequentially on the local machine
# usage: xval.sh <n> "params"

BASE="python main.py"
RESDIR="results"
OUTFILE="$RESDIR/main.py$(echo "$2" | sed "y/ /_/").xval$1.out.txt"
rm -f $OUTFILE

echo "Saving output to \"$OUTFILE\""

for i in $(seq $1)
do
CMD="$BASE --seed_xval $i --seed_pairs $i $2"
echo ">> $CMD" | tee -a $OUTFILE
$CMD | tee -a $OUTFILE
done
