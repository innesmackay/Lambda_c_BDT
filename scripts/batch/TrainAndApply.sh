#!/bin/bash
cd $BASE

snakemake -j 1 -s train_and_apply.snake --configfile configs/${1}.yaml

cd -



