from TestAndTrain import TestAndTrain
from TextFileHandling import Settings, ReadList, ParseCut
from Data import *
import numpy as np
import pandas as pd
import argparse
from root_pandas import to_root
from Log import *

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
config = Settings(args.config)

# Read in different necessary variables
transformed_training_cols = ReadList("configs/training/{}.txt".format(config.GetS("training") ))
raw_training_cols = ReadList("configs/training/{}.txt".format(config.GetS("reqd_vars")))
data_cols = ReadList("configs/other_variables/{}.txt".format(config.GetS("data_vars")))
mc_cols = ReadList("configs/other_variables/{}.txt".format(config.GetS("mc_vars")))
all_data_cols = data_cols + raw_training_cols
all_mc_cols = mc_cols + data_cols + raw_training_cols
if verbose:
    info("Raw variables needed for training:\n{}\n".format("\n".join(str(var) for var in raw_training_cols )))
    info("Additional data variables to be kept:\n{}\n".format("\n".join(str(var) for var in data_cols )))
    info("Additional mc variables to be kept:\n{}\n".format("\n".join(str(var) for var in mc_cols )))


# ==============================
# Load in the data and MC
# ==============================
mc = LoadMC(all_mc_cols, cut=ParseCut(config.GetS("mc_cut")), verbose=verbose)
all_data = LoadNFiles(all_data_cols, n=args.nfiles, cut=ParseCut(config.GetS("data_cut")), verbose=verbose)

# Apply cuts
sideband = all_data.query(ParseCut(config.GetS("sideband_cut")))
data = all_data.query("not ({})".format(ParseCut(config.GetS("sideband_cut"))))


# ==============================
# Setup the training
# ==============================
# Set the target branch for training
sideband["signal"] = np.zeros(len(sideband))
mc["signal"] = np.ones(len(mc))
training_sample = pd.concat([sideband[raw_training_cols + ["signal"]], mc[raw_training_cols + ["signal"]]], ignore_index=True)


# ==============================
# Run the ML algorithm
# ==============================
training = TestAndTrain(config, training_sample, transformed_training_cols)
if args.grid:
    training.GridSearch()
else:
    train_params = {"max_depth": config.GetI("max_depth"), "n_estimators": config.GetI("n_estimators"), "learning_rate": config.GetF("learning_rate")}
    training.Train(train_params)
success("Training done!")

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
    if (len(data) != len(bdt_scores)):
        warning("There is not a 1:1 correspondence between the data and the transformed data")
    if verbose:
        info("Writing data to {}".format(config.GetS("test_outfile")))
    data.to_root(config.GetS("test_outfile"), key="tree")
training.Close()
