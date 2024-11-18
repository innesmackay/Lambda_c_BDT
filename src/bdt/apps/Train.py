from TestAndTrain import TestAndTrain
from Utilities import ParseCut
from Data import *
import numpy as np
import pandas as pd
import yaml

pd.set_option("mode.chained_assignment", None)  # Suppress warning
import argparse
from root_pandas import to_root
from logzero import logger as log

# ==============================
# Read in the input arguments
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Name of config file",
)
parser.add_argument(
    "--apply",
    type=bool,
    required=False,
    default=False,
    help="Apply the BDT to some data",
)
parser.add_argument(
    "--nfiles",
    type=int,
    default=5,
    help="Number of data files to train on",
)
parser.add_argument(
    "--verbose",
    type=bool,
    required=False,
    default=False,
    help="Verbosity boolean",
)
parser.add_argument(
    "--grid",
    type=bool,
    required=False,
    default=False,
    help="Run a grid scan",
)
args = parser.parse_args()
verbose = args.verbose
with open(args.config) as f:
    config = yaml.safe_load(f)

# Read in different necessary variables
transformed_training_cols = config["training"]
raw_training_cols = config["reqd_vars"]
data_cols = config["data_vars"]
mc_cols = config["mc_vars"]
all_data_cols = data_cols + raw_training_cols
all_mc_cols = mc_cols + data_cols + raw_training_cols
if verbose:
    log.info(
        "Raw variables needed for training:\n{}\n".format(
            "\n".join(str(var) for var in raw_training_cols)
        )
    )
    log.info(
        "Additional data variables to be kept:\n{}\n".format(
            "\n".join(str(var) for var in data_cols)
        )
    )
    log.info(
        "Additional mc variables to be kept:\n{}\n".format(
            "\n".join(str(var) for var in mc_cols)
        )
    )


# ==============================
# Load in the data and MC
# ==============================
mc = LoadMC(all_mc_cols, cut=ParseCut(config["mc_cut"]), verbose=verbose)
all_data = LoadNFiles(
    all_data_cols, n=args.nfiles, cut=ParseCut(config["data_cut"]), verbose=verbose
)

# Apply cuts
sideband = all_data.query(ParseCut(config["sideband_cut"]))
data = all_data.query("not ({})".format(ParseCut(config["sideband_cut"])))


# ==============================
# Setup the training
# ==============================
# Set the target branch for training
sideband["signal"] = list(np.zeros(len(sideband)))
mc["signal"] = list(np.ones(len(mc)))
training_sample = pd.concat(
    [sideband[raw_training_cols + ["signal"]], mc[raw_training_cols + ["signal"]]],
    ignore_index=True,
)


# ==============================
# Run the ML algorithm
# ==============================
training = TestAndTrain(config, training_sample, transformed_training_cols)
if args.grid:
    training.GridSearch()
else:
    train_params = {
        "max_depth": int(config["max_depth"]),
        "n_estimators": int(config["n_estimators"]),
        "learning_rate": float(config["learning_rate"]),
    }
    training.Train(train_params)
log.info("Training done!")

# ==============================
# Look at metrics
# ==============================
# training.BinaryKFoldValidation() # Takes too long and doesn't provide much info
training.MakeROC()
training.Importance()
training.CompareVariables()
training.MakeCorrelationMatrix()
training.PersistModel()

# ==============================
# Apply to data
# ==============================
if args.apply:
    bdt_scores = training.Apply(data)
    data["signal_score"] = bdt_scores
    if len(data) != len(bdt_scores):
        log.warn(
            "There is not a 1:1 correspondence between the data and the transformed data"
        )
    if verbose:
        log.info("Writing data to {}".format(config["test_outfile"]))
    data.to_root(config["test_outfile"], key="tree")
training.Close()
