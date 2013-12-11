#!/bin/bash

# Submit a cross-validation job to the barley queue
# usage: xval_qsub.sh <nval> "params"

BASE="python main.py"
RESDIR="results"

for i in $(seq $1)
do
CMD="$BASE --seed_xval $i --seed_pairs $i $2"
FNAME="$RESDIR/main.py$(echo "$2" | sed "y/ /_/").xval$1.$i" # tag with number of runs, and index
OUTFILE="$FNAME.out.txt"
ERRFILE="$OUTFILE.err.log"

touch $OUTFILE
touch $ERRFILE

SCRIPT="$OUTFILE.temp.sh" # batch submit script
rm -f $SCRIPT # delete old script
echo "#!/bin/bash\n" > $SCRIPT

echo "Submitting job with command:"
echo "$CMD" | tee -a $SCRIPT
echo "Saving output to \"$OUTFILE\""

# OPTS=""
OPTS="-l testq=1"
qsub -cwd $OPTS -o $OUTFILE -e $ERRFILE $SCRIPT
done
