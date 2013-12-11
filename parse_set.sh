#!/bin/bash

# filter output files with a list of globs
# usage: parse_set.sh <line_glob> <fixed_glob_VAR> <VAR1> <VAR2> ...

line_glob="$1"
fname_glob="$2"
args=("$@")

for ((i=2; i < $#; i++)) {
	# echo "VAR$i = ${args[$i]}"
	glob=$(echo "$fname_glob" | sed s/VAR/${args[$i]}/)
	# echo $glob
	./filter_output.sh "$line_glob" "$glob"
}