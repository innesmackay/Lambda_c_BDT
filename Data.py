import numpy as np
import pandas as pd
import uproot
import os
from TextFileHandling import *
from Log import *

def LoadCachedData(columns, cut=None, verbose=False):
    """
    Loads cached data stored in a ".root" file into
    a pandas dataframe.
    :param columns: columns to read.
    :param cut: query string to apply.
    :param verbose: verbosity boolean
    """
    if verbose:
        info("Loading in cahced data sample and applying {} cut".format(cut))
    data_root = uproot.open(
        "{}/MagDown_2024_WithUT/data.root:DecayTree".format(os.environ["DATA_PATH"])
    )
    data = data_root.arrays(columns, cut=cut, library="pd")
    return data


def LoadMC(columns, cut=None, verbose=False):
    """
    Loads signal MC stored in a ".root" file into
    a pandas dataframe.
    :param columns: columns to read.
    :param cut: query string to apply.
    :param verbose: verbosity boolean
    """
    if verbose:
        info("Loading in MC sample and applying {} cut".format(cut))
    mc_root = uproot.open("{}/mc.root:LcToPKPi/DecayTree".format(os.environ["MC_PATH"]))
    mc = mc_root.arrays(columns, cut=cut, library="pd")
    return mc


def LoadNFiles(columns, n=2, cut=None, verbose=False):
    """
    Load first N data files from the CERN server into a
    pandas dataframe.
    :param columns: columns to read.
    :param cut: query string to apply.
    :param verbose: verbosity boolean
    """
    if verbose:
        info("Loading in {} files from eos and applying {} cut".format(n, cut))
    files = [
        f"root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision24/PID_TURBOONLY_TUPLE.ROOT/00232503/0000/00232503_{i:08}_1.pid_turboonly_tuple.root"
        for i in range(1,n)
    ]
    data = uproot.concatenate(
        ["{}:LcToPKPi/DecayTree".format(f) for f in files],
        expressions=columns,
        library="pd",
        cut=cut
    )
    return data


def ParseTransformation(definition, verbose=False):
    """
    Classifier training variables are created
    from combinations of "raw" variables to
    reduce the number of dimensions. The
    combinations are defined in
    configs/training/definitions.txt. This function
    parses the string definition such that the
    dataframe can be manipulated.

    :param definition: string definition of combined
    variable
    :param verbose: verbosity boolean

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
    elif "lgsm" in definition: # e.g. log(X+Y+Z)
        strippedDef = definition[5:len(definition)-1] # X+Y+Z
        toCombine = strippedDef.split("*")
        transType = "logsum"
    else:
        error("Couldn't parse definition")

    if verbose:
        info("Parsing {} using type {} and variables {}".format(definition, transType, " ".join(str(var) for var in toCombine)))

    return toCombine, transType


def TransformData(init_df, training_cols, dropna=True):
    """
    Reduce the number of dimensions in the training
    variables by applying transformations. The transformations
    are defined in configs/training/definitions.txt and
    parsed by the ```ParseTransformation``` function.

    :param init_df: pandas dataframe to be transformed
    :param training_cols: list of transformed variable names
    :param dropna: drop rows with NaN values

    :returns new_df: transformed frame
    :returns df: old dataframe (with dropped NaN)
    """
    # Read the definitions
    definitions = Settings("configs/training/definitions.txt")

    df = init_df.dropna() if dropna else init_df
    new_df = pd.DataFrame()
    for col in training_cols:

        # Does the column have a definition
        if col in definitions.GetKeys():
            info("Adding column {} = {}".format(col, definitions.GetS(col)))
            varsToCombine, howToCombine = ParseTransformation( definitions.GetS(col) )
            if howToCombine == "sum":
                new_df[col] = df[varsToCombine].sum(axis=1)
                df[col] = df[varsToCombine].sum(axis=1)
            elif howToCombine == "max":
                new_df[col] = df[varsToCombine].max(axis=1)
                df[col] = df[varsToCombine].max(axis=1)
            elif howToCombine == "min":
                new_df[col] = df[varsToCombine].min(axis=1)
                df[col] = df[varsToCombine].min(axis=1)
            elif howToCombine == "log":
                new_df[col] = np.log10(df[varsToCombine])
                df[col] = np.log10(df[varsToCombine])
            elif howToCombine == "asym":
                new_df[col] = (df[varsToCombine[0]] - df[varsToCombine[1]]) / (df[varsToCombine[0]] + df[varsToCombine[1]])
                df[col] = (df[varsToCombine[0]] - df[varsToCombine[1]]) / (df[varsToCombine[0]] + df[varsToCombine[1]])
            elif howToCombine == "logsum":
                new_df[col] = np.log10( df[varsToCombine].sum(axis=1) )
                df[col] = np.log10( df[varsToCombine].sum(axis=1) )
        else:
            info("No definition for {}, using raw variable".format(col))
            new_df[col] = df[col]

    return new_df.dropna(), df.dropna()





