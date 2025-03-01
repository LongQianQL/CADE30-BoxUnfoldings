#!/bin/bash

if [ "$#" -ne 6 ]; then
    echo "You must enter exactly 6 integers as arguments"
    exit 1
fi

for arg in "$@"
do
    if ! [[ "$arg" =~ ^-?[0-9]+$ ]]; then
        echo "Argument '$arg' is not an integer"
        exit 1
    fi
done

if [ "$1" -eq 1 ] && [ "$2" -eq 1 ]; then
    orientation=0
else
    orientation=5
fi

echo "First box: $1 x $2 x $3"
echo "Second box: $4 x $5 x $6"
python3 experiments/experiment_set_up.py -enc encoding/encoder.py  -pair encoding/find_iso_pairs.py -d $1 $2 $3 $4 $5 $6 --orient2=$orientation -o .
cd $1x$2x$3_$4x$5x$6_orient2_$orientation
