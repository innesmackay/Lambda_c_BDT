from Log import *
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class TransformTheColumns(BaseEstimator, TransformerMixin):
    """
    Class for transforming the columns in data. Included in
    a scikit-learn pipeline. Used for combining raw variables.
    General idea:
       1. Supply list of new variables names (which are defined
          in ```self.definitions```
       2. Parse the definition using ```parse_transformation```
          which returns some variables, and a string
          identifying how they should be combined.
       3. Make a new dataframe with transformed variables
          in ```transform```
    """
    def __init__(self, new_variables, verbose=True, dropna=False):
        """
        :param new_variables: list of transformed variable names
        :param verbose: verbosity boolean
        :param dropna: should rows with NaN be dropped from df?
        """
        self.dropna = dropna
        self.verbose = verbose
        self.new_variables = new_variables
        self.definitions = {
            "sum_GHOSTPROB" : "p_GHOSTPROB+pi_GHOSTPROB+K_GHOSTPROB",
            "sum_MINIPCHI2" : "p_MINIPCHI2+pi_MINIPCHI2+K_MINIPCHI2",
            "sum_PT" : "p_PT+K_PT+pi_PT",
            "sum_TCHI2DOF" : "p_TCHI2DOF+K_TCHI2DOF+pi_TCHI2DOF",
            "max_PT" : "max(p_PT,K_PT,pi_PT)",
            "min_PT" : "min(p_PT,K_PT,pi_PT)",
            "max_TCHI2DOF" : "max(p_TCHI2DOF,K_TCHI2DOF,pi_TCHI2DOF)",
            "min_TCHI2DOF" : "min(p_TCHI2DOF,K_TCHI2DOF,pi_TCHI2DOF)",
            "max_GHOSTPROB" : "max(p_GHOSTPROB,K_GHOSTPROB,pi_GHOSTPROB)",
            "min_GHOSTPROB" : "min(p_GHOSTPROB,K_GHOSTPROB,pi_GHOSTPROB)",
            "max_MINIPCHI2" : "max(p_MINIPCHI2,K_MINIPCHI2,pi_MINIPCHI2)",
            "min_MINIPCHI2" : "min(p_MINIPCHI2,K_MINIPCHI2,pi_MINIPCHI2)",
            "log_Lc_BPVDIRA" : "log(Lc_BPVDIRA)",
            "asym_p_pi_PT" : "asym(p_PT,pi_PT)",
            "asym_p_K_PT" : "asym(p_PT,K_PT)",
            "log_Lc_BPVFDCHI2" : "log(Lc_BPVFDCHI2)",
            "log_p_MINIPCHI2" : "log(p_MINIPCHI2)",
            "log_K_MINIPCHI2" : "log(K_MINIPCHI2)",
            "log_pi_MINIPCHI2" : "log(pi_MINIPCHI2)",
            "lgsm_DOCACHI2" : "lgsm(Lc_DOCACHI2_12*Lc_DOCACHI2_13*Lc_DOCACHI2_23)",
            "asym_K_pi_MINIPCHI2": "asym(K_MINIPCHI2,pi_MINIPCHI2)",
            "lgnm_Lc_BPVDIRA": "lgnm(Lc_BPVDIRA,1e-6)"
        }


    def fit(self, X, y = None):
        """
        Necessary function for scikit-learn pipeline.
        """
        return self


    def parse_transformation(self, definition):
        """
        Classifier training variables are created
        from combinations of "raw" variables to
        reduce the number of dimensions. The
        combinations are defined in
        ```self.definitions```. This function
        parses the string definition such that the
        dataframe can be manipulated.

        :param definition: string definition of combined
        variable
        :returns toCombine: variables to combine
        :returns transType: key to how they will be
        combined
        """
        transType = None
        if "+" in definition: # e.g. X+Y+Z
            toCombine = definition.split("+")
            transType = "sum"
        elif "max" in definition: # e.g. max(X,Y,Z)
            strippedDef = definition[4:len(definition)-1] # X,Y,Z
            transType = "max"
            toCombine = strippedDef.split(",")
        elif "min" in definition: # e.g. min(X,Y,Z)
            strippedDef = definition[4:len(definition)-1] # X,Y,Z
            transType = "max"
            toCombine = strippedDef.split(",")
        elif "log" in definition: # e.g. log(X)
            strippedDef = definition[4:len(definition)-1] # X
            toCombine = strippedDef
            transType = "log"
        elif "asym" in definition: # e.g. asym(X,Y)
            strippedDef = definition[5:len(definition)-1] # X,Y
            toCombine = strippedDef.split(",")
            transType = "asym"
        elif "lgsm" in definition: # e.g. lgsm(X+Y+Z)
            strippedDef = definition[5:len(definition)-1] # X+Y+Z
            toCombine = strippedDef.split("*")
            transType = "logsum"
        elif "lgnm" in definition: # e.g. lgnm(X,2)
            strippedDef = definition[5:len(definition)-1] # X,2
            toCombine = strippedDef.split(",")
            transType = "lognorm"
        else:
            error("Couldn't parse definition")

        if self.verbose:
            toPrint = " ".join(str(var) for var in toCombine)
            if isinstance(toCombine, str):
                toPrint = str(toCombine)
            info("Parsing {} using type {} and variables {}".format(definition, transType, toPrint))

        return toCombine, transType


    def transform(self, X):
        """
        Transform the dataframe to reduce the number of
        dimensions in the training variables by combining them
        to form new variables.
        :param X: pandas dataframe with raw variables
        :returns new_df: pandas dataframe with transformed variables
        """
        new_df = pd.DataFrame()
        for col in self.new_variables:

            # Does the column have a definition
            if col in list(self.definitions.keys()):
                varsToCombine, howToCombine = self.parse_transformation( self.definitions[col] )
                if howToCombine == "sum":
                    new_df[col] = X[varsToCombine].sum(axis=1)
                elif howToCombine == "max":
                    new_df[col] = X[varsToCombine].max(axis=1)
                elif howToCombine == "min":
                    new_df[col] = X[varsToCombine].min(axis=1)
                elif howToCombine == "log":
                    new_df[col] = np.log10(X[varsToCombine])
                elif howToCombine == "asym":
                    new_df[col] = (X[varsToCombine[0]] - X[varsToCombine[1]]) / (X[varsToCombine[0]] + X[varsToCombine[1]])
                elif howToCombine == "logsum":
                    new_df[col] = np.log10( X[varsToCombine].sum(axis=1) )
                elif howToCombine == "lognorm":
                    new_df[col] = np.log10( X[varsToCombine[0]] ) / float(varsToCombine[1])
            else:
                if self.verbose:
                    info("No definition for {}, using raw variable".format(col))
                new_df[col] = X[col]

        return new_df.dropna() if self.dropna else new_df
