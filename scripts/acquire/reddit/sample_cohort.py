
## Input/Output Directories
count_path = "./data/processed/reddit/2017-2020/counts/"
cache_dir = "./data/processed/reddit/2017-2020/author_comment_counts/"
plot_dir = "./plots/reddit/2017-2020/sample/"

####################
### Imports
####################

## Standard Library
import os
import sys
import gzip
import json
from glob import glob
from time import sleep
from collections import Counter
from datetime import timedelta

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from mhlib.util.helpers import chunks
from mhlib.acquire.reddit import RedditData
from mhlib.util.logging import initialize_logger

####################
### Globals
####################

## Logger
LOGGER = initialize_logger()

## Reddit API Wrapper
REDDIT_API = RedditData(False)

####################
### Helpers
####################

def get_author_comment_counts(author_list,
                              api,
                              start_date=None,
                              end_date=None,
                              ):
    """

    """
    ## Maximum Reqeust Length
    if len(author_list) > 100:
        raise ValueError("Input author_list can only have a maximum of 100 authors")
    ## Structure Request
    response = api.api.search_comments(author=author_list,
                                       before=api._get_end_date(end_date),
                                       after=api._get_start_date(start_date),
                                       aggs=["author"])
    ## Get Author Counts
    comment_counts = next(response)
    if "author" not in comment_counts:
        return Counter(dict((a,0) for a in author_list))
    comment_counts = dict((a["key"], a["doc_count"]) for a in comment_counts["author"])
    comment_counts = Counter(comment_counts)
    for a in author_list:
        if a not in comment_counts:
            comment_counts[a] = 0
    return comment_counts

####################
### Load Counts
####################

LOGGER.info("Loading Source Comment Counts")

## Identify Files
count_files = sorted(glob(f"{count_path}*/*.json.gz"))

## Load and Store Counts
counts = Counter()
for cf in tqdm(count_files, file=sys.stdout):
    with gzip.open(cf,"r") as the_file:
        cf_counts = json.load(the_file)
    cf_counts = Counter(cf_counts)
    counts += cf_counts

## Remove Moderators/Bots
IGNORE_USERS = set(["AutoModerator","MemesMod","[deleted]","[removed]"])
counts = dict((x, y) for x, y in counts.items() if x not in IGNORE_USERS)

####################
### Check Activity over Time
####################

LOGGER.info("Starting Activity Query")

## Date Range Parameters
start_date = "2019-01-01"
end_date = "2019-02-01"
freq = 7 # Number of Days or "all"

## Query Parameters
max_retries = 3
backoff = 2

## Create Date Range
if freq == "all":
    date_range = [start_date, end_date]
else:
    date_range = [pd.to_datetime(start_date)]
    while date_range[-1] < pd.to_datetime(end_date):
        date_range.append(min(date_range[-1] + timedelta(freq), pd.to_datetime(end_date)))
    date_range = [i.date().isoformat() for i in date_range]

## Cache Directory/Plot Directory
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

## Get Comment Counts over Date Range
author_chunks = list(chunks(sorted(counts.keys()), n=100))
for dstart, dstop in tqdm(zip(date_range[:-1], date_range[1:]), total=len(date_range)-1, position=0, leave=False, file=sys.stdout, desc="Date Range"):
    for a, author_chunk in tqdm(enumerate(author_chunks), total=len(author_chunks), position=1, leave=False, file=sys.stdout, desc="Author Chunk"):
        ## Check Cache
        cache_file = f"{cache_dir}{dstart}_{dstop}_chunk-{a}.json.gz"
        if os.path.exists(cache_file):
            continue
        ## Query Data
        chunk_comment_counts = None
        for r in range(max_retries):
            try:
                chunk_comment_counts = get_author_comment_counts(author_chunk,
                                                                 api=REDDIT_API,
                                                                 start_date=dstart,
                                                                 end_date=dstop)
                if chunk_comment_counts is not None:
                    break
            except:
                sleep(backoff**r)
        ## Cache Data
        if chunk_comment_counts is None:
            chunk_comment_counts = Counter()
        with gzip.open(cache_file,"wt") as the_file:
            the_file.write(json.dumps(dict(chunk_comment_counts)))
        ## Delay Next Call (120 requests / minute)
        sleep(0.5)

## Load Activity Data Results
activity_files = sorted(glob(f"{cache_dir}*.json.gz"))
dates_from_file = lambda i: tuple(os.path.basename(i).split("_chunk")[0].split("_"))
authors = sorted(counts.keys())
author_index = dict(zip(authors, list(range(len(counts)))))
date_index = dict((y, x) for x, y in enumerate(sorted(set(list(map(dates_from_file, activity_files))), key=lambda x: x[0])))
X = np.zeros((len(author_index), len(date_index)))
for chunk_file in tqdm(activity_files, desc="Loading Activity Data", file=sys.stdout):
    ## Load File Data
    with gzip.open(chunk_file,"r") as the_file:
        chunk_activity_data = json.load(the_file)
    ## Align Data for X
    chunk_authors = sorted(list(chunk_activity_data))
    chunk_row_index = [author_index[i] for i in chunk_authors]
    chunk_col_index = date_index[dates_from_file(chunk_file)]
    chunk_values = np.array([chunk_activity_data[a] for a in chunk_authors])
    ## Update X
    X[chunk_row_index, chunk_col_index] += chunk_values

####################
### Examine Distribution
####################

LOGGER.info("Visualizing Distribution")

## Plot Distribution of Posts
total_posts = X.sum(axis=1)
total_posts_vc = pd.Series(total_posts).value_counts()
fig, ax = plt.subplots()
ax.scatter(total_posts_vc.index,
           total_posts_vc.values,
           alpha=0.5)
ax.set_yscale("symlog")
ax.set_xscale("symlog")
ax.set_xlabel("# Posts", fontweight="bold")
ax.set_ylabel("# Users", fontweight="bold")
fig.tight_layout()
plt.savefig(f"{plot_dir}reddit_post_distribution.png", dpi=300)
plt.close()

####################
### Select Cohort
####################

LOGGER.info("Sampling Cohort")

## Sample Criteria
MIN_POSTS_PER_WEEK = 1
IGNORE_ACTIVE_TOP_PERCENTILE = 1

## Min Post Filter
cohort_mask = np.nonzero((X[:,:-1] >= MIN_POSTS_PER_WEEK).all(axis=1))[0]
cohort_authors = [authors[i] for i in cohort_mask]
cohort_X = X[cohort_mask]

## Outlier Filter
post_threshold =  np.percentile(cohort_X.sum(axis=1), 100-IGNORE_ACTIVE_TOP_PERCENTILE)
cohort_mask = np.nonzero(cohort_X.sum(axis=1)<post_threshold)[0]
cohort_authors = [cohort_authors[i] for i in cohort_mask]
cohort_X = cohort_X[cohort_mask]

## Dump
cohort_file = f"{cache_dir}user_sample.txt"
with open(cohort_file,"w") as the_file:
    for author in cohort_authors:
        the_file.write(f"{author}\n")

LOGGER.info("Script Complete!")