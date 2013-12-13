#!/bin/bash

TLD=$1
TARGET=$2
PATHFILE=$3

for LINE in $(cat $PATHFILE)
do
	echo cp $TLD/$LINE $TARGET/$LINE
	cp $TLD/$LINE $TARGET/$LINE
done