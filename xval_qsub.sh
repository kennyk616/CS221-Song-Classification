#!/bin/bash

# Submit a cross-validation job to the barley queue
# usage: xval_qsub.sh <nval> "params"

BASE="python main.py"
RESDIR="results"

for i in $(seq $1)
do
CMD="$BASE --seed_xval $i --seed_pairs $i $2"
FNAME="$RESDIR/$(echo "$2" | sed "y/ /_/").xval$1.$i" # tag with number of runs, and index
OUTFILE="$FNAME.out"
ERRFILE="$OUTFILE.err.log"

SCRIPT="$OUTFILE.temp.sh" # batch submit script
rm -f $SCRIPT # delete old script
echo "Submitting job with command:"
echo "$CMD" | tee $SCRIPT
echo "Saving output to \"$OUTFILE\""
qsub -cwd -testq=1 -pe fah 1 $CMD -o $OUTFILE -e $ERRFILE $SCRIPT
done
