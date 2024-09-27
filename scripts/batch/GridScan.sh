#!/bin/bash
cd $BASE

theConfig=configs/reduced.txt
python Main.py --config ${theConfig}

cd -



