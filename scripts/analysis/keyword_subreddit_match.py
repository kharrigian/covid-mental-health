
## Processed Data Directory
DATA_DIR = "./data/processed/reddit/2017-2020/histories/"
# DATA_DIR = "./data/processed/twitter/2018-2020/timelines/"

## Plot Directory
PLOT_DIR = "./plots/reddit/2017-2020/keywords-subreddits/"
# PLOT_DIR = "./plots/twitter/2018-2020/keywords/"

## Random Sampling
SAMPLE_RATE = 1
SAMPLE_SEED = 42

## Platform
PLATFORM = "reddit"
# PLATFORM = "twitter"

## Date Boundaries
START_DATE = "2017-01-01"
END_DATE = "2020-05-01"
# START_DATE = "2018-01-01"
# END_DATE = "2020-06-20"

## Date Resolution (hour, day, month, year)
DATE_RES = "day"

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
from functools import partial
from datetime import datetime
from multiprocessing import Pool
from collections import Counter

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from scipy.sparse import csr_matrix, vstack
from pandas.plotting import register_matplotlib_converters

## Mental Health Library
from mhlib.util.logging import initialize_logger
from mhlib.util.helpers import flatten

###################
### Globals
###################

## Logging
LOGGER = initialize_logger()

## Register Matplotlib Time Converters
_ = register_matplotlib_converters()

## Plot Directory
if not os.path.exists(PLOT_DIR):
    _ = os.makedirs(PLOT_DIR)
if not os.path.exists(f"{PLOT_DIR}timeseries/"):
    _ = os.makedirs(f"{PLOT_DIR}timeseries/")

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

def pattern_match(text,
                  patterns):
    """

    """
    if isinstance(text, str):
        text_lower = text.lower()
    else:
        text_lower = [t.lower() for t in text]
    matches = []
    for p in patterns:
        if p.isupper():
            if p in text:
                matches.append(p)
        else:
            if p in text_lower:
                matches.append(p)
    return matches

def match_post(post):
    """

    """    
    ## Cycle Through Options
    match_found = False
    match_cache = {}
    for category in MATCH_DICT.keys():
        if PLATFORM == "reddit":
            if post["subreddit"].lower() in MATCH_DICT[category]["subreddits"]:
                if category not in match_cache:
                    match_cache[category] = {}
                match_cache[category]["subreddits"] = post["subreddit"].lower()
                match_found = True
        if MATCH_DICT[category]["use_tokens"]:
            term_matches = pattern_match(post["text_tokenized"], MATCH_DICT[category]["terms"])
        else:
            term_matches = pattern_match(post["text"], MATCH_DICT[category]["terms"])
        if len(term_matches) > 0:
            if category not in match_cache:
                match_cache[category] = {}
            match_cache[category]["terms"] = term_matches
            match_found = True
    ## Metadata
    if not match_found:
        return None
    post_match_cache = {"matches":match_cache}
    meta_cols = ["user_id_str","created_utc","text"]
    if PLATFORM == "reddit":
        meta_cols.extend(["comment_id","subreddit"])
    elif PLATFORM == "twitter":
        meta_cols.append("tweet_id")
    for val in meta_cols:
        post_match_cache[val] = post[val]
    return post_match_cache

def find_matches(filename,
                 level="day"):
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
                post_matches = match_post(post)
                if post_matches is not None:
                    matches.append(post_matches)
    ## Format Timestamps
    timestamps = format_timestamps(timestamps, level)
    timestamps = Counter(timestamps)
    return filename, matches, n, n_seen, timestamps

def search_files(filenames,
                 date_res="day"):
    """

    """

    ## Run Lookup
    mp = Pool(NUM_JOBS)
    helper = partial(find_matches, level=date_res)
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
                     match_type,
                     level="day"):
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
    timestamps = format_timestamps(timestamps, level)
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

def terms_over_time(term_matches,
                    term_map,
                    date_range_map,
                    subreddit=False):
    """

    """
    ## Initialize Matrix
    term_time_matrix = np.zeros((len(term_map), len(date_range_map)))
    ## Construct Matrix
    for f, (dates, tm) in enumerate(term_matches):
        for dt, t in zip(dates, tm):
            if dt not in date_range_map:
                continue
            dt_ind = date_range_map[dt]
            if not subreddit:
                for _t in t:
                    _t_ind = term_map[_t]
                    term_time_matrix[_t_ind, dt_ind] += 1
            else:
                _t_ind = term_map[t]
                term_time_matrix[_t_ind, dt_ind] += 1
    ## Extract Row/Column Names
    date_range_map_rev = dict((y,x) for x, y in date_range_map.items())
    term_map_rev = dict((y,x) for x, y in term_map.items())
    rows = list(map(lambda i: term_map_rev[i], range(term_time_matrix.shape[0])))
    cols = list(map(lambda i: date_range_map_rev[i], range(term_time_matrix.shape[1])))
    cols = list(map(lambda i: datetime(*list(i)), cols))
    ## Format into DataFrame
    df = pd.DataFrame(term_time_matrix.T, index=cols, columns=rows)
    return df

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

## Load Georgetown Mental Health Resources
MH_TERM_FILE = "./data/resources/mental_health_terms.json"
MH_SUBREDDIT_FILE = "./data/resources/mental_health_subreddits.json"
with open(MH_TERM_FILE,"r") as the_file:
    MH_TERMS = json.load(the_file)
with open(MH_SUBREDDIT_FILE,"r") as the_file:
    MH_SUBREDDITS = json.load(the_file)
if "diagnos" in MH_TERMS["terms"]["smhd"]:
    MH_TERMS["terms"]["smhd"].remove("diagnos")

## Load Crisis Keywords
CRISIS_KEYWORD_FILES = glob("./data/resources/*crisis*.keywords")
CRISIS_KEYWORDS = set()
for f in CRISIS_KEYWORD_FILES:
    fwords = [i.strip() for i in open(f,"r").readlines()]
    fwords = list(map(lambda i: i.lower() if not i.isupper() else i, fwords))
    CRISIS_KEYWORDS.update(fwords)

## Load PMI Terms/Phrases
MH_KEYWORDS_FILE = "./data/resources/mental_health_keywords_manual_selection.csv"
MH_KEYWORDS = pd.read_csv(MH_KEYWORDS_FILE)
MH_KEYWORDS = set(MH_KEYWORDS.loc[MH_KEYWORDS["ignore_level"].isnull()]["ngram"])
MH_KEYWORDS.add("depression")
MH_KEYWORDS.add("depressed")

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
MATCH_DICT = {
    "mental_health":{
        "terms":set(MH_TERMS["terms"]["smhd"]),
        "subreddits":set(MH_SUBREDDITS["all"]),
        "use_tokens":False
    },
    "crisis":{
        "terms":CRISIS_KEYWORDS,
        "subreddits":set(),
        "use_tokens":True
    },
    "mental_health_keywords":{
        "terms":MH_KEYWORDS,
        "subreddits":set(),
        "use_tokens":True
    },
    "covid":{
        "terms":set(COVID_TERMS["covid"]),
        "subreddits":set(COVID_SUBREDDITS["covid"]),
        "use_tokens":False
    }
}

## Files
filenames = sorted(glob(f"{DATA_DIR}*.json.gz"))

## Look For Matches
filenames, matches, n, n_seen, timestamps = search_files(filenames,
                                                         date_res=DATE_RES)
 
###################
### Summarize Matches
###################

## Get Values and Timestamps for each Match Category/Type
mh_term_matches = list(map(lambda p: get_match_values(p, "mental_health", "terms", DATE_RES), matches))
mh_subreddit_matches = list(map(lambda p: get_match_values(p, "mental_health", "subreddits",DATE_RES), matches))
crisis_term_matches = list(map(lambda p: get_match_values(p, "crisis", "terms",DATE_RES), matches))
mh_keyword_term_matches = list(map(lambda p: get_match_values(p, "mental_health_keywords", "terms",DATE_RES), matches))
covid_term_matches = list(map(lambda p: get_match_values(p, "covid", "terms",DATE_RES), matches))
covid_subreddit_matches = list(map(lambda p: get_match_values(p, "covid", "subreddits",DATE_RES), matches))

###################
### Temporal Analysis
###################

## Date Range
date_range = pd.date_range(START_DATE,
                           END_DATE,
                           freq="h")
date_range_simple = format_timestamps(date_range, DATE_RES)
date_range_simple = sorted(set(format_timestamps(date_range, DATE_RES)))
date_range_map = dict(zip(date_range_simple, range(len(date_range_simple))))

## Vectorize Term Match Timestamps
mh_term_ts = vstack([vectorize_timestamps(i[0], date_range_map) for i in mh_term_matches]).toarray()
crisis_term_ts = vstack([vectorize_timestamps(i[0], date_range_map) for i in crisis_term_matches]).toarray()
mh_keyword_term_ts =   vstack([vectorize_timestamps(i[0], date_range_map) for i in mh_keyword_term_matches]).toarray()
covid_term_ts = vstack([vectorize_timestamps(i[0], date_range_map) for i in covid_term_matches]).toarray()

## Vectorize Subreddit Match Timestamps
if PLATFORM == "reddit":
    mh_subreddit_ts = vstack([vectorize_timestamps(i[0], date_range_map) for i in mh_subreddit_matches]).toarray()
    covid_subreddit_ts = vstack([vectorize_timestamps(i[0], date_range_map) for i in covid_subreddit_matches]).toarray()

## Vectorize General Timestamps
tau = vstack([vectorize_timestamps(i, date_range_map) for i in timestamps]).toarray()

## Normalize Term Match Timestamps
mh_term_ts_norm = np.divide(mh_term_ts,
                            tau,
                            out=np.zeros_like(mh_term_ts),
                            where=tau>0)
crisis_term_ts_norm = np.divide(crisis_term_ts,
                           tau,
                           out=np.zeros_like(crisis_term_ts),
                           where=tau>0)
mh_keyword_term_ts_norm = np.divide(mh_keyword_term_ts,
                               tau,
                               out=np.zeros_like(mh_keyword_term_ts),
                               where=tau>0)
covid_term_ts_norm = np.divide(covid_term_ts,
                               tau,
                               out=np.zeros_like(covid_term_ts),
                               where=tau>0)

## Normalize Subreddit Match Timestamps
if PLATFORM == "reddit":
    mh_subreddit_ts_norm = np.divide(mh_subreddit_ts,
                                    tau,
                                    out=np.zeros_like(mh_subreddit_ts),
                                    where=tau>0)
    covid_subreddit_ts_norm = np.divide(covid_subreddit_ts,
                                        tau,
                                        out=np.zeros_like(covid_subreddit_ts),
                                        where=tau>0)

## Term/Subreddit Maps
mh_term_map = dict((y, x) for x, y in enumerate(sorted(MATCH_DICT["mental_health"]["terms"])))
crisis_term_map = dict((y, x) for x, y in enumerate(sorted(MATCH_DICT["crisis"]["terms"])))
mh_keyword_term_map = dict((y, x) for x, y in enumerate(sorted(MATCH_DICT["mental_health_keywords"]["terms"])))
covid_term_map = dict((y, x) for x, y in enumerate(sorted(MATCH_DICT["covid"]["terms"])))
if PLATFORM == "reddit":
    mh_subreddit_map = dict((y, x) for x, y in enumerate(sorted(MATCH_DICT["mental_health"]["subreddits"])))
    covid_subreddit_map = dict((y, x) for x, y in enumerate(sorted(MATCH_DICT["covid"]["subreddits"])))

## Vectorize Term Breakdowns
mh_term_breakdown = vstack([vectorize_timestamps(flatten(i[1]), mh_term_map) for i in mh_term_matches]).toarray()
crisis_term_breakdown = vstack([vectorize_timestamps(flatten(i[1]), crisis_term_map) for i in crisis_term_matches]).toarray()
mh_keyword_term_breakdown = vstack([vectorize_timestamps(flatten(i[1]), mh_keyword_term_map) for i in mh_keyword_term_matches]).toarray()
covid_term_breakdown = vstack([vectorize_timestamps(flatten(i[1]), covid_term_map) for i in covid_term_matches]).toarray()
if PLATFORM == "reddit":
    mh_subreddit_breakdown = vstack([vectorize_timestamps(i[1], mh_subreddit_map) for i in mh_subreddit_matches]).toarray()
    covid_subreddit_breakdown = vstack([vectorize_timestamps(i[1], covid_subreddit_map) for i in covid_subreddit_matches]).toarray()

## Vectorize Terms Over Time
mh_term_time_df = terms_over_time(mh_term_matches, mh_term_map, date_range_map)
crisis_term_time_df = terms_over_time(crisis_term_matches, crisis_term_map, date_range_map)
mh_keyword_term_time_df = terms_over_time(mh_keyword_term_matches, mh_keyword_term_map, date_range_map)
covid_term_time_df = terms_over_time(covid_term_matches, covid_term_map, date_range_map)
if PLATFORM == "reddit":
    mh_subreddit_time_df = terms_over_time(mh_subreddit_matches, mh_subreddit_map, date_range_map, True)
    covid_subreddit_time_df = terms_over_time(covid_subreddit_matches, covid_subreddit_map, date_range_map, True)

###################
### Temporal Visualization
###################

## Timeseries Fields to Plot
plot_vals = [("SMHD Mental Health Terms",mh_term_ts_norm, mh_term_time_df),
             ("JHU Crisis Terms", crisis_term_ts_norm, crisis_term_time_df),
             ("CLSP Mental Health Terms", mh_keyword_term_ts_norm, mh_keyword_term_time_df),
             ("COVID-19 Terms", covid_term_ts_norm, covid_term_time_df)]
if PLATFORM == "reddit":
    plot_vals.insert(1, ("Mental Health Subreddits", mh_subreddit_ts_norm, mh_subreddit_time_df))
    plot_vals.insert(-1, ("COVID-19 Subreddits", covid_subreddit_ts_norm, covid_subreddit_time_df))

## Plot Post Poportions over Time
fig, ax = plt.subplots(len(plot_vals), 1, figsize=(10,5.8))
date_index = list(map(lambda i: datetime(*list(i)), date_range_simple))
for p, (pname, pmatrix, pdf) in enumerate(plot_vals):
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
    ax[p].set_xlim(left=pd.to_datetime("2020-01-01"),right=pd.to_datetime(END_DATE))
ax[-1].set_xlabel("Date", fontweight="bold")
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}{PLATFORM}_term_subreddit_proportions.png", dpi=300)
plt.close()

## Plot User Proportions over Time
fig, ax = plt.subplots(len(plot_vals), 1, figsize=(10,5.8))
for p, (pname, pmatrix, pdf) in enumerate(plot_vals):
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
    ax[p].set_xlim(left=pd.to_datetime("2020-01-01"),right=pd.to_datetime(END_DATE))
ax[-1].set_xlabel("Date", fontweight="bold")
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}{PLATFORM}_term_subreddit_user_proportions.png", dpi=300)
plt.close()

## Plot User Proportion Entire History
fig, ax = plt.subplots(figsize=(10,5.8))
max_val = -1
for p, (pname, pmatrix, pdf) in enumerate(plot_vals):
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
ax.set_xticklabels([i[0].replace("Term","\nTerm").replace("Subreddit","\nSubreddit") for i in plot_vals])
ax.set_ylim(bottom=0, top=max_val + 4)
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}{PLATFORM}_term_subreddit_user_proportions_overall.png", dpi=300)
plt.close()

## Posts Per Day
tau_series = pd.Series(index=list(map(lambda i: datetime(*list(i)), date_range_simple)), data=tau.sum(axis=0))
pd.DataFrame(tau_series, columns=["num_posts"]).to_csv(f"{PLOT_DIR}posts_per_day.csv")

## Plot Each Term Over Time
for p, (pname, _, pdf) in enumerate(plot_vals):
    ## Dump DataFrame
    pname_clean = pname.replace(" ","_")
    pdf.to_csv(f"{PLOT_DIR}matches_per_day_{pname_clean}.csv")
    ## Create Term/Subreddit Plots
    for term in pdf.columns:
        term_series = pdf[term]
        if term_series.max() <= 5 or (term_series > 0).sum() < 10:
            continue
        term_series_normed = term_series.rolling(14).mean()
        term_series_normed_std = term_series.rolling(14).std()
        fig, ax = plt.subplots(figsize=(10,5.8))
        ax.fill_between(term_series_normed.index,
                        term_series_normed-term_series_normed_std,
                        term_series_normed+term_series_normed_std,
                        alpha=0.3)
        ax.plot(term_series_normed.index,
                term_series_normed.values,
                marker="o",
                linestyle="--",
                linewidth=0.5,
                color="C0",
                ms=1,
                alpha=0.5)
        ax.set_xlabel("Date", fontweight="bold")
        ax.set_ylabel("Posts Per Day (14-day Average)", fontweight="bold")
        ax.set_title(f"{pname}: {term}", fontweight="bold", loc="left")
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        term_clean = term.replace("/","-").replace(".","-")
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}timeseries/{pname_clean}_{term_clean}.png", dpi=300)
        plt.close(fig)

###################
### Term/Subreddit Visualization
###################

## Term/Subreddit Fields to Plot
term_plot_vals = [("SMHD Mental Health Terms",mh_term_breakdown, mh_term_map),
                  ("JHU Crisis Terms", crisis_term_breakdown, crisis_term_map),
                  ("CLSP Mental Health Terms", mh_keyword_term_breakdown, mh_keyword_term_map),
                  ("COVID-19 Terms", covid_term_breakdown, covid_term_map)]
if PLATFORM == "reddit":
    term_plot_vals.insert(1, ("Mental Health Subreddits", mh_subreddit_breakdown, mh_subreddit_map))
    term_plot_vals.insert(-1, ("COVID-19 Subreddits", covid_subreddit_breakdown, covid_subreddit_map))

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
    fig.savefig("{}top_{}_{}.png".format(PLOT_DIR, group.lower().replace(" ","_"), PLATFORM), dpi=300)
    plt.close()

###################
### Summary Plots
###################

## Date Boundaries
PLOT_START = "2019-01-01"
COVID_START = "2020-03-01"

## Threshold
MIN_MATCHES = 150
PLOT_SUBREDDIT = True if PLATFORM == "reddit" else False

## Load Posts Per Day
posts_per_day = pd.read_csv(f"{PLOT_DIR}posts_per_day.csv", index_col=0)["num_posts"]
posts_per_day.index = pd.to_datetime(posts_per_day.index)
posts_per_day = posts_per_day.loc[posts_per_day.index >= pd.to_datetime(DATA_START)]

## Aggregate by Week and Month
posts_per_week = posts_per_day.resample("W-Mon").sum()
posts_per_month = posts_per_day.resample("MS").sum()

## Keywords/Subreddits Files
keyword_files = [("CLSP Mental Health Terms", "matches_per_day_CLSP_Mental_Health_Terms.csv", False),
                 ("COVID-19 Terms", "matches_per_day_COVID-19_Terms.csv", False),
                 ("SMHD Mental Health Terms", "matches_per_day_SMHD_Mental_Health_Terms.csv", False),
                 ("JHU Crisis Terms", "matches_per_day_JHU_Crisis_Terms.csv", False)]       
subreddit_files = [("Mental Health Subreddits", "matches_per_day_Mental_Health_Subreddits.csv", True),
                   ("COVID-19 Subreddits", "matches_per_day_COVID-19_Subreddits.csv", True)]
if PLOT_SUBREDDIT:
    keyword_files.extend(subreddit_files)

## Cycle Through Keywords
for k, kf, ksub in keyword_files:
    ## Load File (Daily)
    kf_df = pd.read_csv(f"{DATA_DIR}{kf}", index_col=0)
    kf_df.index = pd.to_datetime(kf_df.index)
    ## Isolate by Start Date
    kf_df = kf_df.loc[kf_df.index >= pd.to_datetime(DATA_START)]
    ## Create Aggregation by Week
    kf_df_weekly = kf_df.resample('W-Mon').sum()
    ## Create Aggregation by Month
    kf_df_monthly = kf_df.resample("MS").sum()
    ## Posts by Period
    pre_covid_matched_posts = kf_df.loc[kf_df.index < pd.to_datetime(COVID_START)].sum(axis=0)
    pre_covid_posts = posts_per_day.loc[posts_per_day.index < pd.to_datetime(COVID_START)].sum()
    post_covid_matched_posts = kf_df.loc[kf_df.index >= pd.to_datetime(COVID_START)].sum(axis=0)
    posts_covid_posts = posts_per_day.loc[posts_per_day.index >= pd.to_datetime(COVID_START)].sum()
    ## Isolate By Threshold
    good_cols = kf_df.sum(axis=0).loc[kf_df.sum(axis=0) > MIN_MATCHES].index.tolist()
    kf_df = kf_df[good_cols].copy()
    ## Relative Posts by Period
    pre_covid_prop_posts = pre_covid_matched_posts / pre_covid_posts
    post_covid_prop_posts = post_covid_matched_posts / posts_covid_posts
    period_prop_change = post_covid_prop_posts - pre_covid_prop_posts
    period_pct_change = (period_prop_change / pre_covid_prop_posts).dropna().sort_values() * 100
    period_prop_change = period_prop_change.loc[good_cols]
    period_pct_change = period_pct_change.loc[good_cols]
    period_pct_change = period_pct_change.loc[period_pct_change!=np.inf]
    ## Create Summary Plot
    fig, ax = plt.subplots(2, 2, figsize=(12,8), sharex=False, sharey=False)
    ## Matches Over Time
    ax[0][0].plot(pd.to_datetime(kf_df_weekly.index),
                  kf_df_weekly.sum(axis=1) / posts_per_week,
                  linewidth=2,
                  color="C0",
                  alpha=.7,
                  marker="o")
    ax[0][0].axvline(pd.to_datetime(COVID_START),
                     linestyle="--",
                     linewidth=2,
                     color="black",
                     alpha=0.5,
                     label="COVID-19 Start ({})".format(COVID_START))
    xticks = [i for i in kf_df_monthly.index if (i.month - 1) % 4 == 0]
    ax[0][0].set_xticks(xticks)
    ax[0][0].set_xticklabels([i.date() for i in xticks], rotation=25, ha="right")
    ax[0][0].legend(loc="upper left", frameon=True, framealpha=1)
    ax[0][0].set_ylabel("Matches Per Post (Weekly)", fontweight="bold")
    ax[0][0].set_xlabel("Week", fontweight="bold")
    ## Overall Match Rate Per Week
    ax[0][1].hist(kf_df_weekly.sum(axis=1) / posts_per_week,
                  bins=15,
                  label="$\\mu={:.2f}, \\sigma={:.3f}$".format(
                      (kf_df_weekly.sum(axis=1) / posts_per_week).mean(),
                      (kf_df_weekly.sum(axis=1) / posts_per_week).std()
                  ),
                  alpha=.7)
    ax[0][1].set_ylabel("# Weeks", fontweight="bold")
    ax[0][1].set_xlabel("Matches Per Post (Weekly)", fontweight="bold")
    ax[0][1].legend(loc="upper right", frameon=True, facecolor="white", framealpha=1)
    ## Largest Proportional Differences
    nplot = min(10, int(len(period_prop_change)/2))
    plot_data = (period_prop_change.nlargest(nplot).append(period_prop_change.nsmallest(nplot))).sort_values()
    values = list(plot_data.values[:nplot]) + [0] +  list(plot_data.values[nplot:])
    ax[1][0].barh(list(range(nplot*2 + 1)),
                  values,
                  color = list(map(lambda i: "darkred" if i <= 0 else "navy", values)),
                  alpha = 0.6)
    ax[1][0].set_yticks(list(range(nplot*2 + 1)))
    ax[1][0].set_yticklabels(list(plot_data.index[:nplot]) + ["..."] + list(plot_data.index[nplot:]),
                             ha="right", va="center")
    ax[1][0].set_ylim(-.5, nplot*2 + .5)
    ax[1][0].axvline(0, color="black", linestyle="--", alpha=0.5)
    if not ksub:
        ax[1][0].set_ylabel("Term", fontweight="bold")
    else:
        ax[1][0].set_ylabel("Subreddit", fontweight="bold")
    fmt = ticker.ScalarFormatter()
    fmt.set_powerlimits((-2,2))
    ax[1][0].xaxis.set_major_formatter(fmt)
    ax[1][0].set_xlabel("Absolute Change\n(Pre- vs. Post COVID-19 Start)", fontweight="bold")
    ax[1][0].ticklabel_format(axis="x", style="sci")
    ## Largest Percent Differences
    nplot = min(10, int(len(period_pct_change)/2))
    plot_data = (period_pct_change.nlargest(nplot).append(period_pct_change.nsmallest(nplot))).sort_values()
    values = list(plot_data.values[:nplot]) + [0] +  list(plot_data.values[nplot:])
    ax[1][1].barh(list(range(nplot*2 + 1)),
                  values,
                  color = list(map(lambda i: "darkred" if i <= 0 else "navy", values)),
                  alpha = 0.6)
    ax[1][1].set_yticks(list(range(nplot*2 + 1)))
    ax[1][1].set_yticklabels(list(plot_data.index[:nplot]) + ["..."] + list(plot_data.index[nplot:]),
                             ha="right", va="center")
    ax[1][1].set_ylim(-.5, nplot*2 + .5)
    ax[1][1].axvline(0, color="black", linestyle="--", alpha=0.5)
    if not ksub:
        ax[1][1].set_ylabel("Term", fontweight="bold")
    else:
        ax[1][1].set_ylabel("Subreddit", fontweight="bold")
    ax[1][1].set_xlabel("Percent Change\n(Pre- vs. Post COVID-19 Start)", fontweight="bold")
    fig.tight_layout()
    fig.suptitle(k, fontweight="bold", fontsize=14, y=.98)
    fig.subplots_adjust(top=.94)
    fig.savefig("{}summary_{}.png".format(PLOT_DIR, k.replace(" ","_").lower()), dpi=300)
    plt.close(fig)
    