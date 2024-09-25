import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.utils.class_weight import compute_sample_weight
from Utilities import CheckDir
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from Log import info, success
from pickle import dump

class TestAndTrain:

    def __init__(self, config, dataset, training_variables, verbose=False):
        """
        :param config: Settings class
        :param dataset: training dataset with transformed columns
        :param training_variables: list of variables used in training
        """
        self.config = config
        self.train_vars = training_variables
        self.dataset = dataset
        self.train = dataset[training_variables]
        self.target = dataset["signal"]
        self.verbose = verbose

        self.output_folder = config.GetS("output_folder")
        CheckDir("output/{}".format(self.output_folder))

        summary_path = "output/{}/training_summary.txt".format(self.output_folder)
        self.summary = open(summary_path, "w")


    def Train(self, depth, n_est, lr, weight=True):
        """
        Function to train the ML algorithm
        :param depth: depth of the trees
        :param n_est: number of estimators for boosting
        :param lr: BDT learning rate
        :param weight: weight the signal/bkg contributions
        """

        # Divide the sample
        data_train, data_test, target_train, target_test = train_test_split(self.train, self.target, test_size=0.75, random_state=16)

        # Weight the sample
        weights = None
        if weight:
            weights = compute_sample_weight(class_weight="balanced", y=target_train)
            info("Background and signal weights are {}".format(np.unique(weights)))

        # Scale the variables and run training
        ml_alg = Pipeline(
            [
                ("transform", StandardScaler()),
                ("classifier", GradientBoostingClassifier(max_depth=depth, n_estimators=n_est, learning_rate=lr)),
            ]
        )
        ml_alg.fit(data_train, target_train, **{"classifier__sample_weight":weights})
        self.alg = ml_alg

        return


    def BinaryKFoldValidation(self, nfolds=5):
        """
        Run K-fold validation.
        Writes recall, precision and f1 scores to the summary file.
        Classifies each event as binary so not perfect procedure.
        """
        scores = cross_val_predict(self.alg, self.train, self.target, cv=nfolds)

        precision = precision_score(self.target, scores)
        recall = recall_score(self.target, scores)
        fscore = f1_score(self.target, scores)
        confusion = confusion_matrix(self.target, scores)

        self.summary.write("precision {}".format(precision))
        self.summary.write("recall {}".format(recall))
        self.summary.write("f1_score {}".format(fscore))
        self.summary.write("n00 {}".format(confusion[0][0]))
        self.summary.write("n01 {}".format(confusion[0][1]))
        self.summary.write("n10 {}".format(confusion[1][0]))
        self.summary.write("n11 {}".format(confusion[1][1]))

        return


    def MakeROC(self, write=True, nfolds=5):
        """
        Make the ROC curve.
        :param write: save the roc curve as a pdf.
        """
        probs = cross_val_predict(self.alg, self.train, self.target, cv=nfolds, method="predict_proba")
        signal_probs = probs[:,1]
        fpr, tpr, thresholds = roc_curve(self.target, signal_probs)
        roc_score = roc_auc_score(self.target, signal_probs)
        self.summary.write("roc_area {}".format(roc_score))

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(fpr, tpr, color='k')
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        if write:
            plt.savefig("output/{}/roc.pdf".format(self.output_folder), bbox_inches='tight', format='pdf')
        else:
            plt.show()

        return


    def Importance(self):
        """
        Write the importance of the training variables to file.
        """
        importances = self.alg.steps[1][1].feature_importances_
        outfile = open("output/{}/importance.txt".format(self.output_folder), "w")
        for i, imp in enumerate(importances):
            outfile.write("{} {}\n".format(self.train_vars[i], imp))
        outfile.close()

        return


    def CompareVariables(self):
        """
        Compare signal/background distributions for the training variables.
        """
        def GetRange(s, b):
            min, max = np.amin(s), np.amax(s)
            b_min, b_max = np.amin(b), np.amax(b)
            if b_min < min:
                min = b_min
            if b_max > max:
                max = b_max
            return (min, max)

        total = self.train
        total["signal"] = self.target
        bkg_sample = total.query("signal==0")
        sig_sample = total.query("signal==1")
        for var in self.train_vars:
            fig, ax = plt.subplots(figsize=(6,4))
            plt_range = GetRange(sig_sample[var], bkg_sample[var])
            ax.hist(bkg_sample[var], bins=50, histtype="step", color="b", label="bkg", range=plt_range)
            ax.hist(sig_sample[var], bins=50, weights=np.ones(len(sig_sample))*(len(bkg_sample)/len(sig_sample)), histtype="step", color="r", label="sig", range=plt_range)
            ax.set_xlabel(var)
            ax.set_ylabel("Normalised candidates")
            ax.legend(loc='best')
            plt.savefig("output/{}/{}.pdf".format(self.output_folder, var))

        return


    def MakeCorrelationMatrix(self):
        """
        Plot a correlation matrix of the training variables in
        signal and background.
        """
        samples = {}
        total = self.train
        total["signal"] = self.target
        samples["bkg"] = total.query("signal==0").drop("signal", axis=1)
        samples["sig"] = total.query("signal==1").drop("signal", axis=1)

        for sample in samples.keys():
            correlation_matrix = samples[sample].corr()
            fig, ax = plt.subplots(figsize=(8,6))
            #ax.matshow(correlation_matrix)
            sns.heatmap(correlation_matrix,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1.0, vmax=1.0,
                square=True, ax=ax, annot=True, fmt=".2f", linewidth=0.5)
            ax.xaxis.tick_top()
            plt.xticks(rotation=45)
            plt.savefig("output/{}/{}_correlation.pdf".format(self.output_folder, sample), bbox_inches="tight", format="pdf")

        return


    def Apply(self, input):
        """
        Apply the algorithm to data sample and return the signal scores.
        :param input: input data (pandas dataframe) with transformed variables
        """
        data_probs = self.alg.predict_proba(input[self.train_vars])
        return data_probs[:,1]


    def GridSearch(self):
        """
        Run a grid scan on the BDT hyperparameters
        """
        data_train, data_test, target_train, target_test = train_test_split(self.train, self.target, test_size=0.75, random_state=16)

        weights = compute_sample_weight(class_weight="balanced", y=target_train)

        params = {'n_estimators': [2, 5, 10, 15, 20, 30, 50, 75, 100, 150], 'learning_rate': [0.05, 0.1, 0.2, 0.5, 1.0, 2.], 'max_depth': [2, 3, 4, 5, 6, 7, 8]}
        ml_alg = Pipeline(
            [
                ("transform", StandardScaler()),
                ("grid_search", GridSearchCV(GradientBoostingClassifier(),
                                             param_grid=params,
                                             refit=False,
                                             scoring='roc_auc',
                                             n_jobs=50,
                                             cv=5))
            ]
        )
        ml_alg.fit(data_train, target_train, **{"grid_search__sample_weight":weights})
        self.alg = ml_alg.best_estimator_
        if self.verbose:
            success("Best score: {}".format(ml_alg.named_steps["grid_search"].best_score_))
            success("Best parameters: {}".format(ml_alg.named_steps["grid_search"].best_params_))
            info("All results:")
            print(ml_alg.named_steps["grid_search"].cv_results_) # Write to json?

        return

    def PersistModel(self):
        warning("Reminder: be careful when applying this with different versions of scikit-learn!")
        with open(self.config.GetS("model_file"), "wb") as f:
            dump(self.alg, f, protocol=5)
        return


    def Close(self, persist=True):
        """
        Close the output file
        """
        self.summary.close()
        return
