#!/bin/bash
cd $BASE

python -m Train --config configs/no_dira.txt --nfiles 5 --verbose True --apply True

cd -



