#!/bin/bash

declare -a arr=("lsa" "spectral" "normal")

for file in ./configurations/*
do

    echo "----- Clustering with configs $file -----"
    for red in "${arr[@]}"
    do
        # if you are not on OS X, remove '' -e
        echo "----- dimensionality_reduction:  $red -----"
        sed -i '' -e "s/dimensionality_reduction:.*/dimensionality_reduction: $red/g" $file

        for n in {2..25}
        do
            # if you are not on OS X, remove '' -e
            echo "----- n_components: $n -----"
            sed -i '' -e "s/n_components:.*/n_components: $n/g" $file
            pipenv run python3 main.py --config $file
        done
    done
done