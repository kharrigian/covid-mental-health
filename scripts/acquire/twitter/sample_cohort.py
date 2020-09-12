
##################
### Configuration
##################

## Count Outputs
COUNTS_DIR = "./data/processed/twitter/2018-2020/counts/"

## Recaculate Counts (if they already exist)
RERUN_COUNTS = False

## Choose Plot Directory
PLOT_DIR = "./plots/twitter/2018-2020/sample/"

## Date Filtering
START_DATE = "2018-01-01"
END_DATE = "2020-06-20"

##################
### Imports
##################

## Standard Library
import os
import sys
import json
import gzip
from glob import glob
from datetime import datetime
from collections import Counter

## External Libraries
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import hstack, vstack, csr_matrix

##################
### Globals
##################

## Initialize Plot Directory
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

##################
### Helpers
##################

def flatten(l):
    """
    Flatten a list of lists by one level.
    Args:
        l (list of lists): List of lists
    
    Returns:
        flattened_list (list): Flattened list
    """
    flattened_list = [item for sublist in l for item in sublist]
    return flattened_list

def construct_count_matrix(count_cache_file):
    """

    """
    ## Identify Count Files
    count_files = glob(f"{COUNTS_DIR}*.json")
    ## Load Counts
    counts = {}
    for f in tqdm(count_files, desc="Count File Loader", file=sys.stdout):
        with open(f, "r") as the_file:
            fcounts = json.load(the_file)
        for date, date_counts in fcounts.items():
            for user, user_count in date_counts.items():
                if user not in counts:
                    counts[user] = Counter()
                counts[user][date] += user_count
    ## Vectorize
    users = sorted(counts.keys())
    dates = sorted(list(set(flatten(counts.values()))), key = lambda x: datetime.strptime(x, "%Y_%m_%d"))
    dates_dict = dict((d, i) for i, d in enumerate(dates))
    X = []
    for user in tqdm(users, "User Vectorization", file=sys.stdout):
        x = np.zeros(len(dates))
        for date, count in counts[user].items():
            x[dates_dict[date]] += count
        X.append(csr_matrix(x))
    X = vstack(X)
    ## Cache
    _ = joblib.dump({"users":users, "dates":dates, "X":X}, count_cache_file)
    return X, users, dates

def load_count_matrix(count_cache_file):
    """

    """
    ## Load Data
    data = joblib.load(count_cache_file)
    ## Parse Values
    X = data["X"]
    users = data["users"]
    dates = data["dates"]
    return X, users, dates

##################
### Load Post Counts
##################

## Cache File
count_cache_file = f"{COUNTS_DIR}count_cache.joblib"

## Load Counts
if not os.path.exists(count_cache_file) or RERUN_COUNTS:
    print("Building Count Matrix")
    X, users, dates = construct_count_matrix(count_cache_file)
else:
    print("Loading Existing Count Matrix")
    X, users, dates = load_count_matrix(count_cache_file)

## Format Dates
dates = list(map(lambda i: datetime.strptime(i, "%Y_%m_%d"),dates))

## Get Date Mask
date_mask = []
for d, date in enumerate(dates):
    if date >= pd.to_datetime(START_DATE) and date <= pd.to_datetime(END_DATE):
        date_mask.append(d)

## Filter
X = X[:, date_mask]
dates = [dates[i] for i in date_mask]

##################
### Examine Distribution Statistics
##################

print("Summarizing Distribution")

## Compute Post Distribution (Posts per User)
post_distribution = pd.Series(Counter(np.array(X.sum(axis=1).T)[0]))

## Plot Post Distribution
fig, ax = plt.subplots()
ax.scatter(post_distribution.index, post_distribution.values)
ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.set_xlabel("# Tweets",fontweight="bold")
ax.set_ylabel("# Users",fontweight="bold")
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}twitter_post_distribution.png", dpi=300)
plt.close(fig)

## Compute Post Distribution (Posts Per Day)
post_time_distribution = pd.Series(np.array(X.sum(axis=0))[0], index=dates)
post_time_distribution = post_time_distribution.reindex(pd.date_range(post_time_distribution.index.min(),
                                                                      post_time_distribution.index.max()))
post_time_distribution.loc[post_time_distribution < 100] = np.nan
post_time_distribution = post_time_distribution / 1e5

## Plot Post Distribution
fig, ax = plt.subplots()
post_time_distribution.rolling(7).mean().plot(
                                ax=ax,
                                color="C0",
                                linestyle="-",
                                linewidth=2,
                                alpha=0.8,
                                label="7-day Average"
)
ax.scatter(post_time_distribution.index,
           post_time_distribution.values,
           color="C0",
           alpha=0.2,
           s=25)
ax.set_xlabel("Date",fontweight="bold")
ax.set_ylabel("# Posts (10k)", fontweight="bold")
ax.legend(loc="upper right",frameon=True)
fig.autofmt_xdate()
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}twitter_post_distribution_time.png", dpi=300)
plt.close(fig)

##################
### Identify Cohort
##################

print("Selecting Cohort")

## Choose Time Bin Threshold
min_time_bin_threshold = 5

## Time Bins (Month of Year)
time_bins = [(d.year, d.month) for d in dates]
time_bins_index = {}
for tb in sorted(set(time_bins)):
    time_bins_index[tb] = [i for i,b in enumerate(time_bins) if b == tb]

## Create Aggregate
X_agg = []
for i, (tb, ind) in enumerate(time_bins_index.items()):
    X_agg.append(csr_matrix(X[:,ind].sum(axis=1)))
X_agg = hstack(X_agg).tocsr()

## Identify Cohort (Filter Based on Time Bin Posts)
filled_rows = (X_agg>=min_time_bin_threshold).sum(axis=1).max()
cohort_mask = np.array((X_agg>=min_time_bin_threshold).sum(axis=1)==filled_rows).T[0]
cohort_mask = np.nonzero(cohort_mask)[0]
cohort_users = [users[i] for i in cohort_mask]
cohort_X = X[cohort_mask].toarray()
cohort_X_agg = X_agg[cohort_mask].toarray()

## Identify Cohort (Filter Out Top 5% of Posters)
total_post_threshold = np.nanpercentile(cohort_X.sum(axis=1), 95)
cohort_mask = np.nonzero(cohort_X.sum(axis=1) < total_post_threshold)[0]
cohort_users = [cohort_users[i] for i in cohort_mask]
cohort_X = cohort_X[cohort_mask]
cohort_X_agg = cohort_X_agg[cohort_mask]

## Dump
outfile = f"{COUNTS_DIR}user_sample.txt"
with open(outfile,"w") as the_file:
    for cu in cohort_users:
        the_file.write(f"{cu}\n")

print("Script complete!")