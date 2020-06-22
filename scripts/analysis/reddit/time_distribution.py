
## Processed Data Directory
DATA_DIR = "./data/processed/reddit/histories/"

## Date Boundaries
START_DATE = "2018-01-01"
END_DATE = "2020-06-01"

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
from functools import partial
from datetime import datetime
from collections import Counter
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

def count_timestamps(filename,
                     date_range_map):
    """

    """
    ## Load Timestamps
    timestamps = []
    with gzip.open(filename, "r") as the_file:
        for comment in json.load(the_file):
            timestamps.append(comment["created_utc"])
    ## Format Timestamps
    timestamps = list(map(datetime.fromtimestamp, timestamps))
    ## Extract Year, Month
    timestamps = list(map(lambda d: (d.year, d.month), timestamps))
    ## Return Vectorized Count
    tau = np.zeros(len(date_range_map))
    for t in timestamps:
        if t not in date_range_map:
            continue
        tau[date_range_map[t]] += 1
    tau = csr_matrix(tau)
    return tau, filename

def load_timestamp_distribution(filenames,
                                date_range_map):
    """

    """
    ## Count Timestamps
    mp_helper = partial(count_timestamps, date_range_map=date_range_map)
    mp = Pool(NUM_JOBS)
    res = list(tqdm(mp.imap_unordered(mp_helper, filenames), total=len(filenames), file=sys.stdout))
    mp.close()
    ## Parse Result
    tau = vstack([r[0] for r in res])
    filenames = [r[1] for r in res]
    return filenames, tau

###################
### Load Timestamps
###################

## Date Range
date_range = pd.date_range(START_DATE,
                           END_DATE,
                           freq="MS")
date_range_simple = list(map(lambda d: (d.year, d.month), date_range))
date_range_map = dict(zip(date_range_simple, range(len(date_range_simple))))

## Count Timestamps
filenames, tau = load_timestamp_distribution(glob(f"{DATA_DIR}*.json.gz"), date_range_map)

###################
### Visualize Distribution
###################

## Get Percentile
tau_P = []
for row in tau.T:
    tau_P.append(np.percentile(row.toarray()[0], [36,50,84]))
tau_P = pd.DataFrame(np.vstack(tau_P),
                     columns=["lower","median","upper"],
                     index=date_range)

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
plt.savefig("./plots/reddit_post_distribution_time.png", dpi=300)
plt.close()

## Calculate User Consistency Matrix
thresholds = [5, 10, 15, 20, 30, 40, 50]
tau_C = np.zeros((len(thresholds), len(date_range)))
for t, thresh in enumerate(thresholds):
    for d, dr in enumerate(date_range):
        avail = ((tau[:,d:]>=thresh).sum(axis=1) == len(date_range) - d).sum()
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
plt.savefig("./plots/reddit_post_consistency_matrix.png", dpi=300)
plt.close(fig)