
## Timestamp Resolution
DATE_RES = "day"

## Platform
PLATFORM = "twitter"
# PLATFORM = "reddit"

## Data Director
DATA_DIR = "./data/processed/twitter/2018-2020/timelines/"
PLOT_DIR = "./plots/twitter/2018-2020/timelines/"
# DATA_DIR = "./data/processed/reddit/2017-2020/histories/"
# PLOT_DIR = "./plots/reddit/2017-2020/timelines/"

## Date Boundaries
START_DATE = "2019-01-01"
END_DATE = "2020-06-15"

## Parameters
IGNORE_RETWEETS = True

## Multiprocessing
NUM_JOBS = 8

###################
### Imports
###################

## Standard Library
import os
import sys
import gzip
import json
from glob import glob
from datetime import datetime
from collections import Counter
from functools import partial
from multiprocessing import Pool

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, vstack
from mhlib.util.logging import initialize_logger
from pandas.plotting import register_matplotlib_converters

###################
### Globals
###################

## Logging
LOGGER = initialize_logger()

## Timestamp Formatting in Plots
_ = register_matplotlib_converters()

###################
### Helpers
###################

def _format_timestamp(timestamp,
                      level="day"):
    """

    """
    level_func = {
            "hour": lambda x: (x.year, x.month, x.day, x.hour),
            "day":lambda x: (x.year, x.month, x.day),
            "month":lambda x: (x.year, x.month, 1),
            "year":lambda x: (x.year, 1, 1)
    }
    if level not in level_func:
        raise ValueError("'level' must be in {}".format(list(level_func.keys())))
    return level_func.get(level)(timestamp)

def format_timestamps(timestamps,
                      level="day"):
    """
    Args:
        timestamps (list):
        level (str): One of "hour", "day", "month", "year"
    """
    timestamps = list(map(lambda t: _format_timestamp(t, level), timestamps))
    return timestamps

def count_timestamps(filename,
                     date_res="day"):
    """

    """
    ## Load Timestamps
    timestamps = []
    with gzip.open(filename, "r") as the_file:
        for comment in json.load(the_file):
            if IGNORE_RETWEETS and comment["text"].startswith("RT"):
                continue
            timestamps.append(comment["created_utc"])
    ## Format Timestamps
    timestamps = list(map(datetime.fromtimestamp, timestamps))
    ## Extract Year, Month
    timestamps = format_timestamps(timestamps, date_res)
    ## Return Vectorized Count
    tau = np.zeros(len(date_range_map))
    for t in timestamps:
        if t not in date_range_map:
            continue
        tau[date_range_map[t]] += 1
    tau = csr_matrix(tau)
    return tau, filename

def load_timestamp_distribution(filenames,
                                date_range_map,
                                date_res="day"):
    """

    """
    ## Count Timestamps
    mp = Pool(NUM_JOBS)
    helper = partial(count_timestamps, date_res=date_res)
    res = list(tqdm(mp.imap_unordered(helper, filenames), total=len(filenames), file=sys.stdout))
    mp.close()
    ## Parse Result
    tau = vstack([r[0] for r in res])
    filenames = [r[1] for r in res]
    return filenames, tau

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

###################
### Overhead
###################

## Check for Plot Directory
if not os.path.exists(PLOT_DIR):
    _ = os.makedirs(PLOT_DIR)

###################
### Load Timestamps
###################

## Date Range
date_range = pd.date_range(START_DATE,
                           END_DATE,
                           freq="h")
date_range_simple = format_timestamps(date_range, DATE_RES)
date_range_simple = sorted(set(format_timestamps(date_range, DATE_RES)))
date_range_map = dict(zip(date_range_simple, range(len(date_range_simple))))

## Count Timestamps
filenames, tau = load_timestamp_distribution(glob(f"{DATA_DIR}*.json.gz"), date_range_map, DATE_RES)

## Format Array
tau = tau.toarray()

###################
### Visualize Distribution
###################

## Get Percentile
tau_P = bootstrap_sample(tau,
                         func=np.mean,
                         axis=0,
                         sample_percent=70,
                         samples=100)
tau_P = pd.DataFrame(tau_P.T,
                     columns=["lower","median","upper"],
                     index=list(map(lambda i: datetime(*list(i)), date_range_simple)))

## Plot Percentile Range over Time
fig, ax = plt.subplots()
ax.fill_between(tau_P.index,
                tau_P.lower,
                tau_P.upper,
                color="C0",
                alpha=0.4)
ax.plot(tau_P.index,
        tau_P["median"],
        color="C0",
        linestyle="--")
ax.set_xlabel("Month")
ax.set_ylabel("# Posts per User")
fig.autofmt_xdate()
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}{PLATFORM}_post_distribution_time.png", dpi=300)
plt.close()

## Stop Check
if DATE_RES in ["hour","day","year"]:
    exit()

## Calculate User Consistency Matrix
thresholds = [5, 10, 15, 20, 30, 40, 50]
tau_C = np.zeros((len(thresholds), len(date_range_simple)))
for t, thresh in enumerate(thresholds):
    for d, dr in enumerate(date_range_simple):
        avail = ((tau[:,d:]>=thresh).sum(axis=1) == len(date_range_simple) - d).sum()
        tau_C[t, d] = avail

## Plot User Consistency Matrix
fig, ax = plt.subplots()
m = ax.imshow(tau_C,
              interpolation="nearest",
              aspect="auto",
              cmap=plt.cm.Purples,
              alpha=1)
for i, row in enumerate(tau_C):
    for j, val in enumerate(row):
        ax.text(j, i, int(val), rotation=90, fontsize=6, ha="center", va="center")
cbar = fig.colorbar(m)
cbar.set_label("# Users Available")
ax.set_xticks([i for i in range(len(date_range_simple))])
ax.set_xticklabels(list(map(lambda d: d[0] if d[1] == 1 else "", date_range_simple)), ha="center", rotation=90)
ax.set_yticks(list(range(len(thresholds))))
ax.set_yticklabels(thresholds)
ax.set_xlabel("Start Date")
ax.set_ylabel("Minimum Post Frequency")
ax.set_ylim(len(thresholds)-.5, -.5)
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}{PLATFORM}_post_consistency_matrix.png", dpi=300)
plt.close(fig)