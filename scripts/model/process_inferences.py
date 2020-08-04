
####################
### Configuration
####################

## Result Directory
RESULTS_DIR = "./data/results/reddit/2017-2020/inference/weekly/"
# RESULTS_DIR = "./data/results/twitter/2018-2020/inference/weekly/"

## Condition
CONDITION = "depression"

## Parameters
POS_THRESHOLD = 0.5
MIN_POSTS_PER_WINDOW = 5
MIN_TOKENS_PER_WINDOW = 25

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
from statsmodels.api import tsa
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
pred_files = glob(f"{RESULTS_DIR}*/{CONDITION}.predictions.csv")

## Load Predictions
predictions = {}
support = {}
tokens = {}
unique_tokens = {}
date_ranges = {}
for pred_file in sorted(pred_files):
    start, stop = pred_file.split("/")[-2].split("_")
    date_ranges[start] = (start, stop)
    pred_file_df = pd.read_csv(pred_file, index_col=0)
    predictions[start] = pred_file_df["y_pred"].to_dict()
    support[start] = pred_file_df["support"].to_dict()
    tokens[start] = pred_file_df["matched_tokens"].to_dict()
    unique_tokens[start] = pred_file_df["unique_matched_tokens"].to_dict()

## Format
predictions = pd.DataFrame(predictions)
support = pd.DataFrame(support)
tokens = pd.DataFrame(tokens)
unique_tokens = pd.DataFrame(unique_tokens)

## Date Filtering
dates = pd.to_datetime(predictions.columns)
date_diffs = [(y-x).days for x, y in zip(dates[:-1],dates[1:])]
dates_drop = [d.date().isoformat() for d, dd in zip(dates[:-1], date_diffs) if dd != np.median(date_diffs)]
for df in [predictions, support, tokens, unique_tokens]:
    df.drop(dates_drop, axis=1, inplace=True)
dates = pd.to_datetime(predictions.columns)

## Apply Activity Thresholds
predictions_filtered = predictions.copy()
support_filtered = support.copy()
tokens_filtered = tokens.copy()
unique_tokens_filtered = unique_tokens.copy()
for thresh, df in zip([MIN_POSTS_PER_WINDOW, MIN_TOKENS_PER_WINDOW],
                      [support, tokens]):
    mask = df.copy()
    for col in mask.columns:
        mask.loc[mask[col] < MIN_POSTS_PER_WINDOW, col] = np.nan
        mask.loc[mask[col] >= MIN_POSTS_PER_WINDOW, col] = 1
    predictions_filtered = predictions_filtered * mask
    support_filtered = support_filtered * mask
    tokens_filtered = tokens_filtered * mask
    unique_tokens_filtered = unique_tokens_filtered * mask

## Binarize Cleaned Predictions
predictions_binary = predictions_filtered.applymap(lambda x: x > POS_THRESHOLD if not pd.isnull(x) else np.nan)

####################
### Visualize Population Level
####################

## Population-level Bootstrap
pred_CI = bootstrap_sample(predictions_filtered.values,
                           func=np.nanmean,
                           axis=0,
                           sample_percent=30,
                           samples=1000)
pred_CI = pd.DataFrame(pred_CI.T,
                       index=dates,
                       columns=["lower","median","upper"])
pred_CI["n"] = (~predictions_filtered.isnull()).sum(axis=0)

## Population-Level Binary
pred_CI_binary = bootstrap_sample(predictions_binary.values,
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
    ax.axvline(pd.to_datetime("2020-03-01"),
               linestyle="--",
               color="black",
               linewidth=3,
               alpha=.9,
               label="COVID-19 Spike in US (March 1, 2020)")
    ax.fill_between(CI.index,
                    CI["lower"].astype(float).values,
                    CI["upper"].astype(float).values,
                    color="C0",
                    alpha=0.3)
    ax.plot(CI.index,
            CI["median"].astype(float),
            marker="o",
            markersize=5,
            color="C0",
            linewidth=2,
            linestyle="-",
            alpha=.8,
            label="Population Average")
    ax2 = ax.twinx()
    ax2.plot(CI.index,
             CI["n"].values.astype(float),
             marker="o",
             color="C1",
             linewidth=2,
             linestyle="--",
             alpha=.5,
             label="# Users Modeled")
    ax2.set_ylabel("# Users Modeled",
                   fontweight="bold")
    ax.set_xlabel("Date Start", fontweight="bold")
    ax.set_ylabel(ylbl, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left", frameon=True)
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
