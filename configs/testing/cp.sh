cd ../training/reduced_dimensionality/
for file in `ls *`
  do
    cp ../../testing/template.txt ../../testing/$file
  done
cd -
