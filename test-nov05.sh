#!/bin/bash

OUTFILE=$1
BASECMD="python main.py -e 200 -p --reg l2"

CMD="$BASECMD -t 500"
echo "\n" | tee -a $1
echo $CMD | tee -a $1
$CMD | tee -a $1

CMD="$BASECMD -t 1000"
echo "\n" | tee -a $1
echo $CMD | tee -a $1
$CMD | tee -a $1

CMD="$BASECMD -t 2000"
echo "\n" | tee -a $1
echo $CMD | tee -a $1
$CMD | tee -a $1

CMD="$BASECMD -t 4000"
echo "\n" | tee -a $1
echo $CMD | tee -a $1
$CMD | tee -a $1
