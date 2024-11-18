import warnings

warnings.simplefilter(
    action="ignore", category=FutureWarning
)  # Suppress pandas FutureWarning

import numpy as np
import pandas as pd
import uproot
import os
from TextFileHandling import *
from Log import *


def LoadCachedData(columns, cut=None, verbose=False):
    """Load data cached in a local directory.
    :param columns: columns to read.
    :param cut: query string to apply.
    :param verbose: verbosity boolean.
    :returns: data in a pandas dataframe
    """
    if verbose:
        info("Loading in cahced data sample and applying {} cut".format(cut))
    data_root = uproot.open(
        "{}/MagDown_2024_WithUT/data.root:DecayTree".format(os.environ["DATA_PATH"])
    )
    data = data_root.arrays(columns, cut=cut, library="pd")
    return data


def LoadMC(columns, cut=None, verbose=False):
    """Load signal MC from a local directory.
    :param columns: columns to read.
    :param cut: query string to apply.
    :param verbose: verbosity boolean.
    :returns: MC in a pandas dataframe.
    """
    if verbose:
        info("Loading in MC sample and applying {} cut".format(cut))
    mc_root = uproot.open("{}/mc.root:LcToPKPi/DecayTree".format(os.environ["MC_PATH"]))
    mc = mc_root.arrays(columns, cut=cut, library="pd")
    return mc


def LoadNFiles(columns, n=2, cut=None, verbose=False):
    """Load the first N data files from the CERN server.
    :param columns: columns to read.
    :param cut: query string to apply.
    :param verbose: verbosity boolean.
    :returns: data in a pandas dataframe.
    """
    if verbose:
        info("Loading in {} files from eos and applying {} cut".format(n, cut))
    files = [
        f"root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision24/PID_TURBOONLY_TUPLE.ROOT/00232503/0000/00232503_{i:08}_1.pid_turboonly_tuple.root"
        for i in range(1, n)
    ]
    data = uproot.concatenate(
        ["{}:LcToPKPi/DecayTree".format(f) for f in files],
        expressions=columns,
        library="pd",
        cut=cut,
    )
    return data


def LoadFileN(columns, n, cut=None):
    """Load the Nth data file from the CERN server
    :param columns: columns to read.
    :param n: index of file to load.
    :param cut: query string to apply.
    :returns: data in a pandas dataframe.
    """
    files = [
        f"root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision24/PID_TURBOONLY_TUPLE.ROOT/00232503/0000/00232503_{i:08}_1.pid_turboonly_tuple.root"
        for i in range(1, 149)
    ]

    data_root = uproot.open("{}:LcToPKPi/DecayTree".format(files[n]))
    data = data_root.arrays(columns, cut=cut, library="pd")

    return data


def LoadFile(path, columns, cut=None):
    """Load data from a particular file
    :param path: path to file.
    :param columns: columns to read.
    :param cut: query string to apply.
    :returns: data in pandas dataframe.
    """
    data_root = uproot.open(path)
    data = data_root.arrays(columns, cut=cut, library="pd")
    return data
