
#######################
### Imports
#######################

## Standard Library
import os
import sys
import json
import gzip
import argparse
from time import sleep

## External Libraries
import tweepy
import pandas as pd
from mhlib.util.logging import initialize_logger

#######################
### Configuration
#######################

## API Requests
MAX_RETRIES = 3
SLEEP_TIME = 1 

## Load Twitter Credentials
ROOT_DIR =os.path.dirname(os.path.abspath(__file__)) + "/../../../"
with open(f"{ROOT_DIR}config.json","r") as the_file:
    CREDENTIALS = json.load(the_file)

## Initialize Twitter API
TWITTER_AUTH = tweepy.OAuthHandler(CREDENTIALS.get("twitter").get("api_key"), 
                                   CREDENTIALS.get("twitter").get("api_secret_key"))
TWITTER_AUTH.set_access_token(CREDENTIALS.get("twitter").get("access_token"),
                              CREDENTIALS.get("twitter").get("access_secret_token"))
TWITTER_API = tweepy.API(TWITTER_AUTH,
                         wait_on_rate_limit=True,
                         wait_on_rate_limit_notify=True)

## Logger
LOGGER = initialize_logger()

#######################
### Functions
#######################

def parse_arguments():
    """
    Parse command-line to identify configuration filepath.
    Args:
        None
    
    Returns:
        args (argparse Object): Command-line argument holder.
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Pull all available tweets on a User's Timeline")
    ## Generic Arguments
    parser.add_argument("user_list",
                        type=str,
                        help="Path to list of user IDs (.txt file, newline delimited")
    parser.add_argument("output_dir",
                        type=str,
                        help="Where to store tweets")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if not os.path.exists(args.user_list):
        raise ValueError(f"Could not find user list file {args.user_list}")
    return args

def _pull_timeline(user_id,
                   include_rts=False,
                   exclude_replies=True):
    """

    """
    ## Initialize Cursor
    cursor = tweepy.Cursor(TWITTER_API.user_timeline,
                           user_id=user_id,
                           include_rts=include_rts,
                           exclude_replies=exclude_replies,
                           tweet_mode="extended",
                           trim_user=False,
                           count=200)
    ## Cycle Through Pages
    response_jsons = []
    for page in cursor.pages():
        response_jsons.extend([r._json for r in page])
    dates = pd.to_datetime([r["created_at"] for r in response_jsons])
    LOGGER.info("Found {} Tweets (Start: {}, End: {})".format(
                len(response_jsons),
                dates.min().date(),
                dates.max().date()
    ))
    ## Return
    return response_jsons

def pull_timeline(user_id,
                  max_retries=MAX_RETRIES,
                  sleep_time=SLEEP_TIME):
    """

    """
    response = []
    for r in range(max_retries):
        try:
            response = _pull_timeline(user_id)
            break
        except Exception as e:
            if e.response.reason in ['Not Found',"Forbidden"]:
                return []
            else:
                LOGGER.info(e.response)
            sleep_time = (sleep_time + 1) ** 2
            sleep(sleep_time * r)
    return response

def main():
    """

    """
    ## Parse Command Line
    args = parse_arguments()
    ## Create Output Directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    ## Load User List
    users = [i.strip() for i in open(args.user_list,"r").readlines()]
    ## Pull Data
    for u, user_id in enumerate(users):
        ## Check For User File
        outfile = f"{args.output_dir}{user_id}.json.gz"
        if os.path.exists(outfile):
            LOGGER.info(f"Skipping User {user_id} (Already Downloaded)")
            continue
        else:
            LOGGER.info(f"Pulling Tweets for User {u+1}/{len(users)}: {user_id}")
        ## Query
        response = pull_timeline(user_id)
        ## Cache
        with gzip.open(outfile, "wt", encoding="utf-8") as the_file:
            json.dump(response, the_file)
    ## Done
    LOGGER.info("Script Complete!")

#####################
### Run
#####################

if __name__ == "__main__":
    _ = main()