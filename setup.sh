# Set local conda environment
conda activate pid_env

# Set the paths
. setup/paths.sh

# Make the documentation
cd docs
make html
cd $BASE
