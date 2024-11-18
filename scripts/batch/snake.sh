#!/bin/bash
yaml_file=${1}
rule=${2}
snakefile=train_and_apply.snake

# Which rules to run
if [ "$rule" = "all" ];
  then
    echo "Running all the rules!"
    snakemake --delete-all-output -j 1 -s ${snakefile} --configfile configs/${yaml_file}.yaml
    snakemake -j 1 -s ${snakefile} --configfile configs/${yaml_file}.yaml
elif [ "$rule" = "train" ];
  then
    echo "Running the cache"
    snakemake -R --until train -j 1 -s ${snakefile} --configfile configs/${1}.yaml
elif [ "$rule" = "apply" ];
  then
    echo "Applying the BDT to data"
    snakemake -R --until apply -j 1 -s ${snakefile} --configfile configs/${1}.yaml
  fi

cd -


