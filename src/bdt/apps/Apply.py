import pandas as pd
from TextFileHandling import Settings, ReadList, ParseCut
from Data import LoadFileN
import argparse
from root_pandas import to_root
from Log import *
from pickle import load
from Utilities import CheckDir

# ==============================
# Read in the input arguments
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to model file",
)
parser.add_argument(
    "--nfiles",
    type=int,
    default=5,
    help="Number of data files to train on",
)
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Name of config file",
)
parser.add_argument(
    "--outdir",
    type=str,
    required=True,
    help="Output directory",
)
parser.add_argument(
    "--combine",
    type=bool,
    required=False,
    default=True,
    help="Combine the files",
)
parser.add_argument(
    "--verbose",
    type=bool,
    required=False,
    default=True,
    help="Verbosity boolean",
)

args = parser.parse_args()
verbose = args.verbose
outdir = args.outdir
CheckDir(outdir)
config = Settings(args.config)

# Read in different necessary variables
raw_training_cols = ReadList("configs/training/{}.txt".format(config.GetS("reqd_vars")))
data_cols = ReadList("configs/other_variables/{}.txt".format(config.GetS("data_vars")))
mc_cols = ReadList("configs/other_variables/{}.txt".format(config.GetS("mc_vars")))
all_data_cols = data_cols + raw_training_cols
if verbose:
    info("Variables taken from tuple:\n{}\n".format("\n".join(str(var) for var in all_data_cols )))

# ==============================
# Load the ML algorithm
# ==============================
with open(args.model, "rb") as f:
    alg = load(f)


# ==============================
# Apply to data
# ==============================
combined_df = pd.DataFrame()
for i in range(args.nfiles):

    if verbose:
        info("Getting BDT scores for file {}".format(i))

    all_data = LoadFileN(all_data_cols, n=i, cut=ParseCut(config.GetS("data_cut")))
    data = all_data.query("not ({})".format(ParseCut(config.GetS("sideband_cut"))))

    data_probs = alg.predict_proba(data)
    bdt_scores = data_probs[:,1]

    if (len(data) != len(bdt_scores)):
        error("There is not a 1:1 correspondence between the data and the transformed data")
        break
    else:
        data["signal_score"] = bdt_scores

    if args.combine:
        combined_df = pd.concat([data, combined_df])
        info("Combined length: {}".format(len(combined_df)))
    else:
        outfile = f"{outdir}/file_{str(i)}.root"
        data.to_root(outfile, key="tree")

if args.combine:
    outfile = f"{outdir}/data_with_bdt.root"
    combined_df.to_root(outfile, key="tree")
