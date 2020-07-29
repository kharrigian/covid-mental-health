
## Sample File
SAMPLE_FILE = "./data/processed/reddit/2017-2020/author_comment_counts/user_sample.txt"

## Output Directory
OUTDIR = "./data/raw/reddit/2017-2020/histories/"

## Query Timeline
START_DATE = "2017-01-01"
END_DATE = "2020-06-20"
LIMIT = 10000
QUERY_FREQ = "4MS"
WAIT_TIME = 0.5

## Downsampling
SAMPLE_PERCENT = 0.2
RANDOM_STATE = 42

###################
### Imports
###################

## Standard Library
import os
import sys
import gzip
import json
from time import sleep
from random import Random

## External Libraries
import pandas as pd
from tqdm import tqdm
from mhlib.acquire.reddit import RedditData
from mhlib.util.logging import initialize_logger

###################
### Globals
###################

## Logger
LOGGER = initialize_logger()

## Reddit API Wrapper
REDDIT_API = RedditData(False)

###################
### Retrieve Data
###################

## Load User Sample
LOGGER.info("Loading User Sample")
cohort = sorted([i.strip() for i in open(SAMPLE_FILE,"r")])

## Create Output Directory
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

## Sample Cohort
sampler = Random(RANDOM_STATE)
cohort_sample = []
for c in cohort:
    if sampler.uniform(0, 1) >= SAMPLE_PERCENT:
        continue
    cohort_sample.append(c)

## Date Range
date_boundaries = [i.date() for i in pd.date_range(START_DATE, END_DATE, freq=QUERY_FREQ)]
if date_boundaries[0] > pd.to_datetime(START_DATE):
    date_boundaries = [pd.to_datetime(START_DATE).date()] + date_boundaries
if date_boundaries[-1] < pd.to_datetime(END_DATE):
    date_boundaries = date_boundaries + [pd.to_datetime(END_DATE).date()]
date_boundaries = [d.isoformat() for d in date_boundaries]

## Query Data
LOGGER.info("Starting User History Query")
for author in tqdm(cohort_sample, desc="User", file=sys.stdout, position=0, leave=False):
    ## Check Existence
    author_file = f"{OUTDIR}{author}.json.gz"
    if os.path.exists(author_file):
        continue
    ## Query Data
    author_data = []
    for dstart, dstop in tqdm(list(zip(date_boundaries[:-1], date_boundaries[1:])), position=1, leave=False, desc="Date Range", file=sys.stdout):
        ## Standard Query
        range_data = REDDIT_API.retrieve_author_comments(author,
                                                         start_date=dstart,
                                                         end_date=dstop,
                                                         limit=LIMIT)
        ## Length Fall Back (Anomalies)
        if len(range_data) >= LIMIT:
            range_data = REDDIT_API.retrieve_author_comments(author,
                                                             start_date=dstart,
                                                             end_date=dstop,
                                                             limit=LIMIT*5)
        author_data.append(range_data)
        sleep(WAIT_TIME)
    ## Dump Data
    if author_data is None or len(author_data) == 0:
        author_data = [json.dumps({})]
    else:
        author_data = pd.concat(author_data)
        author_data = author_data.drop_duplicates(subset=["id"]).reset_index(drop=True)
        author_data = [r.to_json() for _, r in author_data.iterrows()]
    with gzip.open(author_file, "wt", encoding="utf-8") as the_file:
        for row in author_data:
            the_file.write(f"{row}\n")

## Script Complete
LOGGER.info("Script complete!")