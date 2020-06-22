
## Sample File
SAMPLE_FILE = "./data/processed/reddit/author_comment_counts/user_sample.txt"

## Output Directory
OUTDIR = "./data/raw/reddit/histories/"

## Query Timeline
START_DATE = "2008-01-01"
END_DATE = "2020-06-20"

###################
### Imports
###################

## Standard Library
import os
import sys
import gzip
import json

## External Libraries
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
cohort = [i.strip() for i in open(SAMPLE_FILE,"r")]

## Create Output Directory
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

## Query Data
LOGGER.info("Starting User History Query")
for author in tqdm(cohort, desc="User", file=sys.stdout):
    ## Check Existence
    author_file = f"{OUTDIR}{author}.json.gz"
    if os.path.exists(author_file):
        continue
    ## Query Data
    author_data = REDDIT_API.retrieve_author_comments(author,
                                                      start_date=START_DATE,
                                                      end_date=END_DATE,
                                                      limit=500000,
                                                    )
    ## Dump Data
    if author_data is None or len(author_data) == 0:
        author_data = [json.dumps({})]
    else:
        author_data = [r.to_json() for _, r in author_data.iterrows()]
    with gzip.open(author_file, "wt", encoding="utf-8") as the_file:
        for row in author_data:
            the_file.write(f"{row}\n")

## Script Complete
LOGGER.info("Script complete!")