Train BDT
============

Application for training the BDT. It can be used as follows from the base directory:

``
python -m Train --config ${YAML_WITH_SETTINGS} --verbose [optional boolean]
``

A yaml file with all of the training settings is passed to the `config` arguments. An example can be found in `configs/test.yaml`.
The training arguments include:

  * list of training variables
  * number of data files to use in the training
  * any preselection requirements
  * BDT hyperparameters
  * output directories and files

.. automodule:: Train
   :members:
   :undoc-members:
   :show-inheritance:
