Preprocess class
=================

Class for transforming the columns in data. The transormations are included in
a scikit-learn pipeline and are used to combine raw variables prior to training.
The general strcture of the code is as follows:

    1. Supply list of new variables names (which are defined
       in `self.definitions`.

    2. Parse the definition using `parse_transformation`
       which returns some variables, and a string
       identifying how they should be combined.

    3. Make a new dataframe with transformed variables
       in `transform`


.. automodule:: Preprocess
   :members:
   :undoc-members:
   :show-inheritance:
