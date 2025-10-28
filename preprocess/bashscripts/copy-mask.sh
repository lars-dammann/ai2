#!/bin/sh
cd "$(dirname "$0")"
cd ..

for data in train val test
do
    for time in before after
    do
        for file in ../data/$data/$time/image/*
        do
            basefile=$(basename $file)
            cp "rawdata/processed/$time/mask/${basefile::-4}.npy" ../data/$data/$time/mask/.
        done
    done
done
