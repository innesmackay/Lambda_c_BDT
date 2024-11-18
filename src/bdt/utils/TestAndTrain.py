from Preprocess import TransformTheColumns
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.utils.class_weight import compute_sample_weight
from Utilities import CheckDir
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, FunctionTransformer, FeatureUnion
from logzero import logger as log
from pickle import dump
from sklearn.impute import SimpleImputer


class TestAndTrain:

    def __init__(self, config, dataset, training_variables, verbose=False):
        """Initialise class.

        :param config: loaded yaml file
        :param dataset: training dataset
        :param training_variables: list of transformed variables used in training
        """
        self.config = config
        self.train_vars = training_variables
        self.dataset = dataset
        self.train = dataset.drop(["signal"], axis=1)
        self.target = dataset["signal"]
        self.verbose = verbose

        tfr = TransformTheColumns(
            new_variables=self.train_vars + ["signal"], verbose=True
        )
        self.transformed_df = tfr.transform(self.dataset)

        self.output_folder = config["output_folder"]
        CheckDir("output/{}".format(self.output_folder))

        summary_path = "output/{}/training_summary.txt".format(self.output_folder)
        self.summary = open(summary_path, "w")

    def Train(self, params, weight=True):
        """Train the ML algorithm

        :param params: dictionary containing BDT training parmaeters including:

          1. `n_est` - number of estimators for boosting

          2. `lr` - BDT learning rate

          3. `max_depth` - depth of each tree.

        :param weight: weight the signal/bkg contributions
        """

        # Divide the sample
        data_train, data_test, target_train, target_test = train_test_split(
            self.train, self.target, test_size=0.25, random_state=16
        )

        # Weight the sample
        weights = None
        if weight:
            weights = compute_sample_weight(class_weight="balanced", y=target_train)
            log.info("Background and signal weights are {}".format(np.unique(weights)))

        # Scale the variables and run training
        ml_alg = Pipeline(
            steps=[
                (
                    "transformer",
                    TransformTheColumns(new_variables=self.train_vars, verbose=False),
                ),
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),  # Replace NaN with median
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        learning_rate=params["learning_rate"],
                        max_depth=params["max_depth"],
                        n_estimators=params["n_estimators"],
                    ),
                ),
            ]
        )
        ml_alg.fit(data_train, target_train, **{"classifier__sample_weight": weights})
        self.alg = ml_alg

        self.ResponsePlot(
            data_train, data_test, target_train, target_test, weights=weights
        )

        return

    def BinaryKFoldValidation(self, nfolds=5):
        """Run K-fold validation.
        Writes recall, precision and f1 scores to the `self.summary` file.
        Classifies each event as binary so not perfect procedure.
        """
        scores = cross_val_predict(self.alg, self.train, self.target, cv=nfolds)

        precision = precision_score(self.target, scores)
        recall = recall_score(self.target, scores)
        fscore = f1_score(self.target, scores)
        confusion = confusion_matrix(self.target, scores)

        self.summary.write("precision {}\n".format(precision))
        self.summary.write("recall {}\n".format(recall))
        self.summary.write("f1_score {}\n".format(fscore))
        self.summary.write("n00 {}\n".format(confusion[0][0]))
        self.summary.write("n01 {}\n".format(confusion[0][1]))
        self.summary.write("n10 {}\n".format(confusion[1][0]))
        self.summary.write("n11 {}\n".format(confusion[1][1]))

        return

    def ResponsePlot(
        self, data_train, data_test, target_train, target_test, weights=None
    ):
        """Draw histograms comparing the BDT response
        in the training and testing samples of signal
        and background. Useful for checking overtraining
        and BDT generalisation.

        :param data_train: data training sample.
        :param data_test: data testing sample.
        :param target_train: target training sample.
        :param target_test: target testing sample.
        :param weights: weights for the signal/bkg.
        """
        train_response = self.alg.predict_proba(data_train)
        data_train["score"] = train_response[:, 1]
        data_train["signal"] = target_train
        sig_train = data_train.query("signal==1")
        bkg_train = data_train.query("signal==0")

        test_response = self.alg.predict_proba(data_test)
        data_test["score"] = test_response[:, 1]
        data_test["signal"] = target_test
        sig_test = data_test.query("signal==1")
        bkg_test = data_test.query("signal==0")

        bkg_weight = None
        sig_weight = None
        if weights is not None:
            bkg_weight = np.amin(np.unique(weights))
            sig_weight = np.amax(np.unique(weights))

        n_sig_test, bin_edges = np.histogram(
            sig_test["score"],
            bins=20,
            range=(0, 1),
            weights=np.ones(len(sig_test)) * sig_weight,
        )
        n_bkg_test, _ = np.histogram(
            bkg_test["score"],
            bins=20,
            range=(0, 1),
            weights=np.ones(len(bkg_test)) * bkg_weight,
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            sig_train["score"],
            bins=20,
            range=(0, 1),
            color="b",
            histtype="stepfilled",
            label="Signal train",
            alpha=0.3,
            weights=np.ones(len(sig_train)) * (sig_weight / 3),
        )
        ax.hist(
            bkg_train["score"],
            bins=20,
            range=(0, 1),
            color="r",
            histtype="stepfilled",
            label="Bkg train",
            alpha=0.3,
            weights=np.ones(len(bkg_train)) * (bkg_weight) / 3,
        )
        ax.plot(
            bin_centers,
            n_bkg_test,
            color="r",
            label="Bkg test",
            linestyle="None",
            marker="o",
            markersize=2,
        )
        ax.plot(
            bin_centers,
            n_sig_test,
            color="b",
            label="Signal test",
            linestyle="None",
            marker="o",
            markersize=2,
        )
        ax.legend(loc="best")
        ax.set_xlabel("BDT response", fontsize=14)
        ax.set_ylabel("Candidates", fontsize=14)
        plt.savefig(
            "output/{}/response.pdf".format(self.output_folder),
            bbox_inches="tight",
            format="pdf",
        )

        return

    def MakeROC(self, write=True, nfolds=5):
        """Draw the ROC curve.
        :param write: boolean - save the roc curve.
        :param nfolds: number of times to fold the test sample.
        """
        probs = cross_val_predict(
            self.alg, self.train, self.target, cv=nfolds, method="predict_proba"
        )
        signal_probs = probs[:, 1]
        fpr, tpr, thresholds = roc_curve(self.target, signal_probs)
        roc_score = roc_auc_score(self.target, signal_probs)
        self.summary.write("roc_area {}".format(roc_score))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color="k")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        if write:
            plt.savefig(
                "output/{}/roc.pdf".format(self.output_folder),
                bbox_inches="tight",
                format="pdf",
            )
        else:
            plt.show()

        return

    def Importance(self):
        """Write the importance of the training variables to file."""
        importances = self.alg.named_steps["classifier"].feature_importances_
        outfile = open("output/{}/importance.txt".format(self.output_folder), "w")
        for i, imp in enumerate(importances):
            outfile.write("{} {}\n".format(self.train_vars[i], imp))
        outfile.close()

        return

    def CompareVariables(self):
        """Compare signal/background distributions for the training variables."""

        def GetRange(s, b):
            min, max = np.amin(s), np.amax(s)
            b_min, b_max = np.amin(b), np.amax(b)
            if b_min < min:
                min = b_min
            if b_max > max:
                max = b_max
            return (min, max)

        bkg_sample = self.transformed_df.query("signal==0")
        sig_sample = self.transformed_df.query("signal==1")
        for var in self.train_vars:
            fig, ax = plt.subplots(figsize=(6, 4))
            plt_range = GetRange(sig_sample[var], bkg_sample[var])
            ax.hist(
                bkg_sample[var],
                bins=50,
                histtype="step",
                color="b",
                label="bkg",
                range=plt_range,
            )
            ax.hist(
                sig_sample[var],
                bins=50,
                weights=np.ones(len(sig_sample)) * (len(bkg_sample) / len(sig_sample)),
                histtype="step",
                color="r",
                label="sig",
                range=plt_range,
            )
            ax.set_xlabel(var)
            ax.set_ylabel("Normalised candidates")
            ax.legend(loc="best")
            plt.savefig(
                "output/{}/{}.pdf".format(self.output_folder, var),
                bbox_inches="tight",
                format="pdf",
            )

        return

    def MakeCorrelationMatrix(self):
        """Plot a correlation matrix of the training variables in
        signal and background.
        """
        samples = {}
        samples["bkg"] = self.transformed_df.query("signal==0").drop("signal", axis=1)
        samples["sig"] = self.transformed_df.query("signal==1").drop("signal", axis=1)
        for sample in samples.keys():
            correlation_matrix = samples[sample].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                correlation_matrix,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1.0,
                vmax=1.0,
                square=True,
                ax=ax,
                annot=True,
                fmt=".2f",
                linewidth=0.5,
            )
            ax.xaxis.tick_top()
            plt.xticks(rotation=45)
            plt.savefig(
                "output/{}/{}_correlation.pdf".format(self.output_folder, sample),
                bbox_inches="tight",
                format="pdf",
            )

        return

    def Apply(self, input):
        """Apply the algorithm to data sample and return the BDT scores.

        :param input: input data (pandas dataframe) with transformed variables
        """
        data_probs = self.alg.predict_proba(input)
        return data_probs[:, 1]

    def GridSearch(self):
        """Run a grid scan on the BDT hyperparameters"""
        data_train, data_test, target_train, target_test = train_test_split(
            self.train, self.target, test_size=0.75, random_state=16
        )
        weights = compute_sample_weight(class_weight="balanced", y=target_train)

        params = {
            "n_estimators": [2, 5, 10, 20, 50],
            "learning_rate": [0.05, 0.1, 0.2, 0.5],
            "max_depth": [2, 3, 4],
        }
        ml_alg = Pipeline(
            [
                (
                    "transformer",
                    TransformTheColumns(new_variables=self.train_vars, verbose=False),
                ),
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),  # Replace NaN with median
                ("scaler", StandardScaler()),
                (
                    "grid_search",
                    GridSearchCV(
                        GradientBoostingClassifier(),
                        param_grid=params,
                        refit=False,
                        scoring="roc_auc",
                        n_jobs=50,
                        cv=5,
                    ),
                ),
            ]
        )
        ml_alg.fit(data_train, target_train, **{"grid_search__sample_weight": weights})
        log.info(
            "Grid search complete\nBest score: {}".format(
                ml_alg.named_steps["grid_search"].best_score_
            )
        )
        log.info(
            "Best parameters: {}".format(ml_alg.named_steps["grid_search"].best_params_)
        )
        log.info("All results:")
        print(ml_alg.named_steps["grid_search"].cv_results_)

        # Train with best parameters
        self.Train(ml_alg.named_steps["grid_search"].best_params_)

        return

    def PersistModel(self):
        """Persist (save) the model to a pickle file."""
        log.warn(
            "Reminder: be careful when applying this with different versions of scikit-learn!"
        )
        with open(self.config["model_file"], "wb") as f:
            dump(self.alg, f, protocol=5)
        return

    def Close(self, persist=True):
        """Close the summary file"""
        self.summary.close()
        return
