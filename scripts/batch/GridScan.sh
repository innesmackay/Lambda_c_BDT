#!/bin/bash
cd $BASE

theConfig=configs/reduced_after_comp.txt
python Main.py --config ${theConfig} --nfiles 5 --verbose True --grid True

cd -



