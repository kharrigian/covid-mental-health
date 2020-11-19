
## Configurable Parameters
START_DATE = "2020-05-25"
END_DATE = "2020-06-01"
DATA_FOLDER = "./data/processed/reddit/2017-2020/counts/"

#######################
### Imports
#######################

## Standard Library
import os
import sys
import gzip
import json
from datetime import datetime

## External Libraries
import pandas as pd
from retriever import Reddit as import RedditData
from mhlib.util.logging import initialize_logger

#######################
### Globals
#######################

## Logging
LOGGER = initialize_logger()

## Subreddits For Identifying User Accounts
SEED_SUBREDDITS =  ['funny',
                    'askreddit',
                    'gaming',
                    'pics',
                    'aww',
                    'science',
                    'worldnews',
                    'music',
                    'movies',
                    'videos',
                    'todayilearned',
                    'news',
                    'iama',
                    'gifs',
                    'showerthoughts',
                    'earthporn',
                    'askscience',
                    'jokes',
                    'food',
                    'explainlikeimfive',
                    'books',
                    'blog',
                    'lifeprotips',
                    'art',
                    'mildlyinteresting',
                    'diy',
                    'sports',
                    'nottheonion',
                    'space',
                    'gadgets',
                    'television',
                    'documentaries',
                    'photoshopbattles',
                    'getmotivated',
                    'listentothis',
                    'upliftingnews',
                    'tifu',
                    'internetisbeautiful',
                    'history',
                    'philosophy',
                    'futurology',
                    'oldschoolcool',
                    'dataisbeautiful',
                    'writingprompts',
                    'personalfinance',
                    'memes']

#######################
### Cycle Through Subreddits/Days
#######################

## Create Output Directory
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

## Date Range
date_range = pd.date_range(START_DATE, END_DATE, freq="D")
date_range = [i.date().isoformat() for i in date_range]

## Initialize Reddit API Wrapper
reddit = RedditData(False)

## Cycle Through Dates and Subreddits
for dstart, dstop in zip(date_range[:-1],date_range[1:]):
    LOGGER.info("~"*25 + f"\nStarting Date: {dstart}\n" + "~"*25)
    ## Output Folder Check
    date_folder = f"{DATA_FOLDER}{dstart}_{dstop}/"
    if not os.path.exists(date_folder):
        os.makedirs(date_folder)
    for s, subreddit in enumerate(SEED_SUBREDDITS):
        LOGGER.info(f"Subreddit {s+1}/{len(SEED_SUBREDDITS)}: {subreddit}")
        ## Output File Check
        outfile = f"{date_folder}{subreddit}.json.gz"
        if os.path.exists(outfile):
            continue
        ## Query Data
        activity = reddit.retrieve_subreddit_user_history(subreddit=subreddit,
                                                          start_date=dstart,
                                                          end_date=dstop,
                                                          history_type="comment")
        ## Cache (Replacing Null Data if Necessary)
        if activity is None:
            activity = pd.Series()
        with gzip.open(outfile, "wt") as the_file:
            the_file.write(activity.to_json())
    
LOGGER.info("Script complete!")