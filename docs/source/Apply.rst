Apply BDT
=============

Application to get the BDT scores in a dataset using a persisted model. It can be used in the base directory as follows:

``
python -m Apply --model ${MODEL_PKL_FILE} --config ${YAML_WITH_SETTINGS} --outdir ${OUTPUT_DIRECTORY} --combine [optional boolean] --verbose [optional boolean] --file_to_apply [optional str] ${SINGLE_FILE}
``

A number of details are set in a configuration file which must be passed to the `config` argument. An example can be found in `configs/test.yaml`.

By default the script will load in N files from the CERN server and apply the BDT scores one at a time. If you want to combine all the files set the `combine` argument to `True`.

If you want to apply the BDT scores to a single file which is not on the CERN server then pass the path to the `file_to_apply` argument.

.. automodule:: Apply
   :members:
   :undoc-members:
   :show-inheritance:
