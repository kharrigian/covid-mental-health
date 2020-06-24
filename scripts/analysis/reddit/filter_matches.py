
## Processed Data Directory
DATA_DIR = "./data/processed/reddit/histories/"

## Random Sampling
SAMPLE_RATE = 0.1
SAMPLE_SEED = 42

## Date Boundaries
START_DATE = "2008-01-01"
END_DATE = "2020-05-01"

## Multiprocessing
NUM_JOBS = 8

###################
### Imports
###################

## Standard Library
import os
import re
import sys
import gzip
import json
import random
from glob import glob
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from collections import Counter

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, vstack
from mhlib.util.logging import initialize_logger
from mhlib.util.helpers import flatten
from pandas.plotting import register_matplotlib_converters

###################
### Globals
###################

## Logging
LOGGER = initialize_logger()

## Register Matplotlib Time Converters
_ = register_matplotlib_converters()

###################
### Helpers
###################

def pattern_match(text,
                  patterns):
    """

    """
    text_lower = text.lower()
    matches = []
    for p in patterns:
        if p in text_lower:
            matches.append(p)
    return matches

def match_post(post,
               match_dict):
    """

    """    
    ## Cycle Through Options
    match_found = False
    match_cache = {}
    for category in match_dict.keys():
        if post["subreddit"].lower() in match_dict[category]["subreddits"]:
            if category not in match_cache:
                match_cache[category] = {}
            match_cache[category]["subreddits"] = post["subreddit"].lower()
            match_found = True
        term_matches = pattern_match(post["text"], match_dict[category]["terms"])
        if len(term_matches) > 0:
            if category not in match_cache:
                match_cache[category] = {}
            match_cache[category]["terms"] = term_matches
            match_found = True
    ## Metadata
    if not match_found:
        return None
    post_match_cache = {"matches":match_cache}
    for val in ["user_id_str","created_utc","comment_id","text","subreddit"]:
        post_match_cache[val] = post[val]
    return post_match_cache

def find_matches(filename,
                 match_dict):
    """

    """
    ## Initialize Sampler
    sampler = random.Random(SAMPLE_SEED)
    ## Search For Matches
    matches = []
    timestamps = []
    n = 0
    n_seen = 0
    with gzip.open(filename,"r") as the_file:
        for post in json.load(the_file):
            n += 1
            if sampler.uniform(0,1) >= SAMPLE_RATE:
                continue
            else:
                n_seen += 1
                timestamps.append(datetime.fromtimestamp(post["created_utc"]))
                post_matches = match_post(post,
                                          match_dict)
                if post_matches is not None:
                    matches.append(post_matches)
    ## Format Timestamps
    timestamps = [(d.year, d.month) for d in timestamps]
    timestamps = Counter(timestamps)
    return filename, matches, n, n_seen, timestamps

def search_files(filenames,
                 match_dict):
    """

    """
    ## Parameterize Helper
    helper = partial(find_matches, match_dict=match_dict)
    ## Run Lookup
    mp = Pool(NUM_JOBS)
    res = list(tqdm(mp.imap_unordered(helper, filenames), total=len(filenames), desc="Searching For Matches"))
    mp.close()
    ## Parse
    filenames = [r[0] for r in res]
    matches = [r[1] for r in res]
    n = [r[2] for r in res]
    n_seen = [r[3] for r in res]
    timestamps = [r[4] for r in res]
    return filenames, matches, n, n_seen, timestamps

def get_match_values(post_matches,
                     category,
                     match_type):
    """

    """
    ## Look For Relevant Matches
    timestamps = []
    values = []
    for p in post_matches:
        if category in p["matches"] and match_type in p["matches"][category]:
            timestamps.append(p["created_utc"])
            values.append(p["matches"][category][match_type])
    ## Format
    timestamps = list(map(datetime.fromtimestamp, timestamps))
    timestamps = list(map(lambda d: (d.year, d.month), timestamps))
    return timestamps, values

def vectorize_timestamps(timestamps,
                         date_range_map):
    """

    """
    tau = np.zeros(len(date_range_map))
    if not isinstance(timestamps, Counter):
        timestamps = Counter(timestamps)
    for t, v in timestamps.items():
        if t not in date_range_map:
            continue
        tau[date_range_map[t]] = v
    tau = csr_matrix(tau)
    return tau

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
### MH Terms/Subreddits
###################

## Load Mental Health Resources
MH_TERM_FILE = "./data/resources/mental_health_terms.json"
MH_SUBREDDIT_FILE = "./data/resources/mental_health_subreddits.json"
with open(MH_TERM_FILE,"r") as the_file:
    MH_TERMS = json.load(the_file)
with open(MH_SUBREDDIT_FILE,"r") as the_file:
    MH_SUBREDDITS = json.load(the_file)

## Filter Out General Diagnosis Terms
if "diagnos" in MH_TERMS["terms"]["smhd"]:
    MH_TERMS["terms"]["smhd"].remove("diagnos")

###################
### COVID Terms/Subreddits
###################

## Load COVID Resources
COVID_TERM_FILE = "./data/resources/covid_terms.json"
COVID_SUBREDDIT_FILE = "./data/resources/covid_subreddits.json"
with open(COVID_TERM_FILE,"r") as the_file:
    COVID_TERMS = json.load(the_file)
with open(COVID_SUBREDDIT_FILE,"r") as the_file:
    COVID_SUBREDDITS = json.load(the_file)

###################
### Identify Matches
###################

## Match Dictionary
match_dict = {
    "mental_health":{
        "terms":set(MH_TERMS["terms"]["smhd"]),
        "subreddits":set(MH_SUBREDDITS["all"])
    },
    "covid":{
        "terms":set(COVID_TERMS["covid"]),
        "subreddits":set(COVID_SUBREDDITS["covid"])
    }
}

## Files
filenames = sorted(glob(f"{DATA_DIR}*.json.gz"))

## Look For Matches
filenames, matches, n, n_seen, timestamps = search_files(filenames,
                                                         match_dict)
 
###################
### Summarize Matches
###################

## Get Values and Timestamps for each Match Category/Type
mh_term_matches = list(map(lambda p: get_match_values(p, "mental_health", "terms"), matches))
mh_subreddit_matches = list(map(lambda p: get_match_values(p, "mental_health", "subreddits"), matches))
covid_term_matches = list(map(lambda p: get_match_values(p, "covid", "terms"), matches))
covid_subreddit_matches = list(map(lambda p: get_match_values(p, "covid", "subreddits"), matches))

###################
### Temporal Analysis
###################

## Date Range
date_range = pd.date_range(START_DATE,
                           END_DATE,
                           freq="MS")
date_range_simple = list(map(lambda d: (d.year, d.month), date_range))
date_range_map = dict(zip(date_range_simple, range(len(date_range_simple))))

## Vectorize Match Timestamps
mh_term_ts = vstack([vectorize_timestamps(i[0], date_range_map) for i in mh_term_matches]).toarray()
mh_subreddit_ts = vstack([vectorize_timestamps(i[0], date_range_map) for i in mh_subreddit_matches]).toarray()
covid_term_ts = vstack([vectorize_timestamps(i[0], date_range_map) for i in covid_term_matches]).toarray()
covid_subreddit_ts = vstack([vectorize_timestamps(i[0], date_range_map) for i in covid_subreddit_matches]).toarray()

## Vectorize General Timestamps
tau = vstack([vectorize_timestamps(i, date_range_map) for i in timestamps]).toarray()

## Normalize Match Timestamps
mh_term_ts_norm = np.divide(mh_term_ts,
                            tau,
                            out=np.zeros_like(mh_term_ts),
                            where=tau>0)
mh_subreddit_ts_norm = np.divide(mh_subreddit_ts,
                                 tau,
                                 out=np.zeros_like(mh_subreddit_ts),
                                 where=tau>0)
covid_term_ts_norm = np.divide(covid_term_ts,
                               tau,
                               out=np.zeros_like(covid_term_ts),
                               where=tau>0)
covid_subreddit_ts_norm = np.divide(covid_subreddit_ts,
                                    tau,
                                    out=np.zeros_like(covid_subreddit_ts),
                                    where=tau>0)

## Term/Subreddit Maps
mh_term_map = dict((y, x) for x, y in enumerate(sorted(match_dict["mental_health"]["terms"])))
mh_subreddit_map = dict((y, x) for x, y in enumerate(sorted(match_dict["mental_health"]["subreddits"])))
covid_term_map = dict((y, x) for x, y in enumerate(sorted(match_dict["covid"]["terms"])))
covid_subreddit_map = dict((y, x) for x, y in enumerate(sorted(match_dict["covid"]["subreddits"])))

## Vectorize Term Breakdowns
mh_term_breakdown = vstack([vectorize_timestamps(flatten(i[1]), mh_term_map) for i in mh_term_matches]).toarray()
mh_subreddit_breakdown = vstack([vectorize_timestamps(i[1], mh_subreddit_map) for i in mh_subreddit_matches]).toarray()
covid_term_breakdown = vstack([vectorize_timestamps(flatten(i[1]), covid_term_map) for i in covid_term_matches]).toarray()
covid_subreddit_breakdown = vstack([vectorize_timestamps(i[1], covid_subreddit_map) for i in covid_subreddit_matches]).toarray()

###################
### Temporal Visualization
###################

## Timeseries Fields to Plot
plot_vals = [("Mental Health Terms",mh_term_ts_norm),
             ("Mental Health Subreddits", mh_subreddit_ts_norm),
             ("COVID-19 Terms", covid_term_ts_norm),
             ("COVID-19 Subreddits", covid_subreddit_ts_norm)]

## Plot Post Poportions over Time
fig, ax = plt.subplots(len(plot_vals), 1, figsize=(10,5.8))
date_index = [i.date() for i in date_range]
for p, (pname, pmatrix) in enumerate(plot_vals):
    pci = bootstrap_sample(pmatrix,
                           func=np.mean,
                           sample_percent=70,
                           samples=100)
    ax[p].fill_between(date_index,
                       pci[0],
                       pci[2],
                       color="C0",
                       alpha=.5)
    ax[p].plot(date_index,
               pci[1],
               color="C0",
               alpha=0.8,
               linestyle="--",
               linewidth=2)
    ax[p].set_title(pname, loc="left", fontweight="bold")
    ax[p].set_ylabel("Proportion\nof Posts", fontweight="bold")
    ax[p].spines["top"].set_visible(False)
    ax[p].spines["right"].set_visible(False)
    ax[p].set_xlim(left=pd.to_datetime("2014-01-01"),right=pd.to_datetime(END_DATE))
ax[-1].set_xlabel("Date", fontweight="bold")
fig.tight_layout()
plt.savefig("plots/reddit_term_subreddit_proportions.png", dpi=300)
plt.close()

## Plot User Proportions over Time
fig, ax = plt.subplots(len(plot_vals), 1, figsize=(10,5.8))
for p, (pname, pmatrix) in enumerate(plot_vals):
    pci = bootstrap_sample(pmatrix,
                           tau,
                           func=lambda x, y: (x>0).sum(axis=0) / (y>0).sum(axis=0),
                           sample_percent=70,
                           samples=100)
    ax[p].fill_between(date_index,
                       pci[0],
                       pci[2],
                       color="C0",
                       alpha=.5)
    ax[p].plot(date_index,
               pci[1],
               color="C0",
               alpha=0.8,
               linestyle="--",
               linewidth=2)
    ax[p].set_title(pname, loc="left", fontweight="bold")
    ax[p].set_ylabel("Proportion\nof Users", fontweight="bold")
    ax[p].spines["top"].set_visible(False)
    ax[p].spines["right"].set_visible(False)
    ax[p].set_xlim(left=pd.to_datetime("2014-01-01"),right=pd.to_datetime(END_DATE))
ax[-1].set_xlabel("Date", fontweight="bold")
fig.tight_layout()
plt.savefig("plots/reddit_term_subreddit_user_proportions.png", dpi=300)
plt.close()

## Plot User Proportion Entire History
fig, ax = plt.subplots(figsize=(10,5.8))
max_val = -1
for p, (pname, pmatrix) in enumerate(plot_vals):
    val = (pmatrix>0).any(axis=1).sum() / pmatrix.shape[0] * 100
    ax.bar(p,
           val,
           color=f"C{p}",
           alpha=0.75)
    ax.text(p,
            val + 2,
            "{:.2f}%".format(val),
            ha="center",
            va="center")
    if val > max_val:
        max_val = val
ax.set_ylabel("Percentage of\nUsers", fontweight="bold")
ax.set_xticks(list(range(p+1)))
ax.set_xticklabels([i[0] for i in plot_vals])
ax.set_ylim(bottom=0, top=max_val + 4)
fig.tight_layout()
plt.savefig("plots/reddit_term_subreddit_user_proportions_overall.png", dpi=300)
plt.close()

###################
### Term/Subreddit Visualization
###################

## Term/Subreddit Fields to Plot
term_plot_vals = [("Mental Health Terms",mh_term_breakdown, mh_term_map),
                  ("Mental Health Subreddits", mh_subreddit_breakdown, mh_subreddit_map),
                  ("COVID-19 Terms", covid_term_breakdown, covid_term_map),
                  ("COVID-19 Subreddits", covid_subreddit_breakdown, covid_subreddit_map)]

## Plot Frequencies
for group, group_breakdown, group_map in term_plot_vals:
    ## Names
    group_map_r = dict((y,x) for x,y in group_map.items())
    group_index = [group_map_r[i] for i in range(len(group_map_r))]
    ## Total (Across Users)
    vals = pd.Series(group_breakdown.sum(axis=0),
                     index=group_index).sort_values().nlargest(30).iloc[::-1]
    ## Total (Per Users)
    vals_per_user = pd.Series((group_breakdown>0).sum(axis=0),
                              index=group_index).sort_values().nlargest(30).iloc[::-1]
    ## Plot
    fig, ax = plt.subplots(1,2,figsize=(10,5.8))
    vals.plot.barh(ax=ax[0],
                   color="C0",
                   alpha=.7)
    vals_per_user.plot.barh(ax=ax[1],
                            color="C0",
                            alpha=.7)
    ax[0].set_xlabel("Total Matches", fontweight="bold")
    ax[1].set_xlabel("Total Users with Match", fontweight="bold")
    fig.suptitle(group, y=.98, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=.94)
    fig.savefig("plots/top_{}.png".format(group.lower().replace(" ","_")), dpi=300)
    plt.close()
