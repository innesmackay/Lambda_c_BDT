#!/bin/bash
cd $BASE

theConfig=configs/testing/${1}.txt
python Main.py --config ${theConfig}

cd -



