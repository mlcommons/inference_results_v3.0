#!/bin/bash
# Prepare models by joining files
cd models

filename="large_model_files.txt"
lines=$(cat $filename)
for base_file in $lines
do
    cat ${base_file}_p* > ${base_file}
    rm ${base_file}_p*
done

cd ..
