import pandas as pd
import yaml
from Data import LoadFileN, LoadFile
import argparse
from root_pandas import to_root
from logzero import logger as log
from pickle import load
from Utilities import CheckDir, ParseCut

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
    "--file_to_apply",
    type=str,
    required=False,
    default=None,
    help="Apply the bdt to a single file",
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
with open(args.config) as f:
    config = yaml.safe_load(f)

# Read in different necessary variables
raw_training_cols = config["reqd_vars"]
data_cols = config["data_vars"]
mc_cols = config["mc_vars"]
all_data_cols = data_cols + raw_training_cols
if verbose:
    log.info(
        "Variables taken from tuple:\n{}\n".format(
            "\n".join(str(var) for var in all_data_cols)
        )
    )

# ==============================
# Load the ML algorithm
# ==============================
with open(args.model, "rb") as f:
    alg = load(f)


# ==============================
# Apply to data
# ==============================
if args.file_to_apply is not None:
    all_data = LoadFile(
        args.file_to_apply, all_data_cols, cut=ParseCut(config["data_cut"])
    )
    data = all_data.query("not ({})".format(ParseCut(config["sideband_cut"])))

    data_probs = alg.predict_proba(data)
    bdt_scores = data_probs[:, 1]

    if len(data) != len(bdt_scores):
        log.error(
            "There is not a 1:1 correspondence between the data and the transformed data"
        )
    else:
        data[config["bdt_branch_name"]] = bdt_scores

    full_outfile = args.file_to_apply.replace(
        ".root", "_{}.root".format(config["bdt_branch_name"])
    )
    outfile = full_outfile.split(":")[0]  # Remove branch name
    tree_name = full_outfile.split(":")[1]
    data.to_root(outfile, key=tree_name)

else:
    combined_df = pd.DataFrame()
    for i in range(config["n_files_apply"]):

        if verbose:
            log.info("Getting BDT scores for file {}".format(i))

        all_data = LoadFileN(all_data_cols, n=i, cut=ParseCut(config["data_cut"]))
        data = all_data.query("not ({})".format(ParseCut(config["sideband_cut"])))

        data_probs = alg.predict_proba(data)
        bdt_scores = data_probs[:, 1]

        if len(data) != len(bdt_scores):
            log.error(
                "There is not a 1:1 correspondence between the data and the transformed data"
            )
            break
        else:
            data["signal_score"] = bdt_scores

        if args.combine:
            combined_df = pd.concat([data, combined_df])
            log.info("Combined length: {}".format(len(combined_df)))
        else:
            outfile = f"{outdir}/file_{str(i)}.root"
            data.to_root(outfile, key="tree")

    if args.combine:
        outfile = f"{outdir}/data_with_bdt.root"
        combined_df.to_root(outfile, key="tree")
