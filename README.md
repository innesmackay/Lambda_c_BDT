# $\Lambda_c^+$ BDT
Coding framework designed to train and apply a BDT using the `scikit-learn` and `pandas` packages.

## Code organisation
To get started run ```conda env create -f env.yml``` which creates a conda environment with all the required packages.
Then, run ```source setup.sh``` which sets necessary paths (you will need to configure these to 
your individual needs) and makes the documentation.

There are two main applications:
  * Training the BDT [located in ```src/bdt/apps/Train.py```]
  * Applying a persisted BDT, saved in a `pkl` file by the training application, to a dataset [located in ```src/bdt/apps/Apply.py```]
The applications are configured using `yaml` files stored in the `configs` directory.

In addition there are a number of helper modules and classes in ```src/bdt/utils```.
  * ```Preprocess.py```: Class used to preprocess a dataset before training/application, including 
                         automatically performing variable transformations and dealing with `NaNs`.
  * ```TestAndTrain.py```: Class used to perform all of the BDT training and testing including studies of
                           the input variables and output metrics.
  * ```Data.py```: Contains a number of hepful function for handling the data.
  * ```Utilities.py```: Contains general utility functions

More detailed documentation and instructions for the python code can be found on the [readthedocs](XXX).

## Snakemake
The `snakemake` package is used to automatically run the training, testing and application
of a BDT given a particular config file. You can run the full script using:
```
snakemake -j ${N_CORES} -s train_and_apply.snake --configfile ${CONFIG}
```
or a particular rule using:
```
snakemake -R until ${RULE_NAME} -j ${N_CORES} -s train_and_apply.snake --configfile ${CONFIG}
```

It is often useful to run a dry run, by adding ```--dry-run``` to the above, to ensure the code will run as expected.
Another useful command is to add ```--delete-output-files``` which will delete the persisted model and
the applied data.

## HTCondor
Batch scripts are run using ```HTCondor``` and can be found in `scripts/batch`. They can be run using ```condor_submit submit.sh```
after writing the configuration filename and snakemake rule in `configs.txt`.

## Notebooks
A few jupyter notebooks can be found in the `notebooks` directory which display comparisons between
various training configuration alongiside some commentary. The effect of the BDT can be most
clearly seen in the ```notebooks/Summary.ipynb``` notebook.

