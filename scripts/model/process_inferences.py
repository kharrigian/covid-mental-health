
####################
### Configuration
####################

## Result Directory
RESULTS_DIR = "./data/results/reddit/inference/weekly/"

## Condition
CONDITION = "depression"

## Parameters
POS_THRESHOLD = 0.5

####################
### Imports
####################

## Standard Library
import os
import sys
import gzip
import json
from glob import glob
from datetime import datetime

## External Libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

####################
### Globals
####################

_ = register_matplotlib_converters()

####################
### Helpers
####################

def bootstrap_sample(X,
                     Y=None,
                     func=np.mean,
                     axis=0,
                     sample_percent=70,
                     samples=100):
    """

    """
    sample_size = int(sample_percent / 100 * X.shape[0])
    estimates = []
    for sample in range(samples):
        sample_ind = np.random.choice(X.shape[0], size=sample_size, replace=True)
        X_sample = X[sample_ind]
        if Y is not None:
            Y_sample = Y[sample_ind]
            sample_est = func(X_sample, Y_sample)
        else:
            sample_est = func(X_sample, axis=axis)
        estimates.append(sample_est)
    estimates = np.vstack(estimates)
    ci = np.percentile(estimates, [2.5, 50, 97.5], axis=axis)
    return ci

####################
### Load Predictions
####################

## Identify Prediction Files
pred_files = glob(f"{RESULTS_DIR}*/{CONDITION}.predictions.json.gz")

## Load Predictions
predictions = {}
for pred_file in sorted(pred_files):
    start, stop = pred_file.split("/")[-2].split("_")
    with gzip.open(pred_file,"r") as the_file:
        predictions[start] = json.load(the_file)
predictions = pd.DataFrame(predictions)

## Binarize Predictions
predictions_binary = predictions.applymap(lambda x: x > POS_THRESHOLD if not pd.isnull(x) else np.nan)

####################
### Visualize Population Level
####################

## Dates
dates = pd.to_datetime(predictions.columns)

## Population-level Bootstrap
pred_CI = bootstrap_sample(predictions.dropna().values,
                           func=np.nanmean,
                           axis=0,
                           sample_percent=30,
                           samples=250)
pred_CI = pd.DataFrame(pred_CI.T,
                       index=dates,
                       columns=["lower","median","upper"])
pred_CI["n"] = (~predictions.isnull()).sum(axis=0)

## Population-Level Binary
pred_CI_binary = bootstrap_sample(predictions_binary.dropna().values,
                                  func=np.nanmean,
                                  axis=0,
                                  sample_percent=30,
                                  samples=250)
pred_CI_binary = pd.DataFrame(pred_CI_binary.T,
                              index=dates,
                              columns=["lower","median","upper"])
pred_CI_binary["n"] = (~predictions_binary.isnull()).sum(axis=0)

## Visualize Population-Level Predictions
for CI, CI_name, ylbl in zip([pred_CI, pred_CI_binary],
                       ["population_level","population_level_binary"],
                       [f"Mean Pr({CONDITION.title()})", f"Percent Pr({CONDITION.title()}) > {POS_THRESHOLD}"]):
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.errorbar(CI.index,
                CI["median"],
                yerr=np.vstack([(CI["median"]-CI["lower"]).values,
                                (CI["upper"]-CI["median"]).values]),
                color="C0",
                linewidth=2,
                label="95% Confidence Interval",
                marker="o",
                linestyle="--")
    ax.set_xlabel("Date Start", fontweight="bold")
    ax.set_ylabel(ylbl, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}inferences_{CI_name}_{CONDITION}.png", dpi=300)
    plt.close()

####################
### Visualize Individual Level
####################

# ## Bin Predictions
# bin_function = lambda x: int(np.floor(x * 10)) if not pd.isnull(x) else np.nan
# predictions_binned = predictions.applymap(bin_function)

# ## Compute Transition Matrices
# tm = {}
# for start, end in zip(predictions_binned.columns[:-1], predictions_binned.columns[1:]):
#     x = predictions_binned[[start, end]].rename(columns={start:"start",end:"end"})
#     x = x.dropna().astype(int)
#     tm_ = metrics.confusion_matrix(x["start"], x["end"], labels=list(range(10)))
#     tm[(start, end)] = tm_

# ## Plot Transition Matrix
# fig, ax = plt.subplots(1, len(tm))
# for i, ((start, end), matrix) in enumerate(tm.items()):
#     ax[i].imshow(np.log(matrix),
#                  cmap=plt.cm.Purples,
#                  aspect="auto")
#     ax[i].set_ylabel(start, fontsize=6)
#     ax[i].set_xlabel(end, fontsize=6)
# fig.tight_layout()
# plt.savefig("test.png", dpi=300)
# plt.close()
