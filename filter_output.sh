#!/bin/bash

# filter output files
# usage: filter_output.sh <line_glob> <fname_glob>

# grep -h "KNN relax test" *k_7*scale* | grep -Po '[0-9.]+' | sed ':a;N;$!ba;s/\n/\t/g'
echo "# Field: <$1> :: match: *$2*"
grep -h "$1" *$2* | grep -Po '[0-9.]+' | sed ':a;N;$!ba;s/\n/\t/g'