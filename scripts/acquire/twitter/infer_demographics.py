

## Inputs
# RAW_DIR = "./data/raw/twitter/2013-2014/timelines/"
# OUTFILE = "./data/processed/twitter/2013-2014/demographics.csv"
# RAW_DIR = "./data/raw/twitter/2016/timelines/"
# OUTFILE = "./data/processed/twitter/2016/demographics.csv"
RAW_DIR = "./data/raw/twitter/2018-2020/timelines/"
OUTFILE = "./data/processed/twitter/2018-2020/demographics.csv"

## Parameters
EXCLUDE_RETWEETS=True
EXCLUDE_NON_ENGLISH=False
MIN_RES_THRESHOLD=0.3

## Multiprocessing
JOBS = 8

#######################
### Imports
#######################

## Standard Library
import os
import sys
import gzip
import json
from glob import glob
from functools import partial
from multiprocessing import Pool

## External Libraries
import pandas as pd
from tqdm import tqdm
from mhlib.util.logging import initialize_logger
from mhlib.util.helpers import flatten

## Lab Resources
import carmen # https://github.com/mdredze/carmen-python
from demographer.gender import CensusGenderDemographer # https://bitbucket.org/mdredze/demographer/src/master/
from demographer.indorg import IndividualOrgDemographer
from demographer import process_tweet

#######################
### Globals
#######################

## Logging
LOGGER = initialize_logger()

## Geolocation Resolver
GEO_RESOLVER = carmen.get_resolver(order=["place","geocode","profile"])
GEO_RESOLVER.load_locations()

## Demographers
DEMOGRAPHERS = [
    CensusGenderDemographer(),
    IndividualOrgDemographer()
]

## Column Map
COLUMN_MAP = {
    "location":[
        "longitude",
        "latitude",
        "country",
        "state",
        "county",
        "city",
    ],
    "indorg_balanced":[
        "indorg_balanced"
    ],
    "gender":[
        "gender"
    ]
}

#######################
### Helpers
#######################

def load_tweets(filename,
                exclude_retweets=True,
                exclude_non_english=False):
    """

    """
    ## Load
    if filename.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open
    try:
        with opener(filename,"r") as the_file:
            tweets = json.load(the_file)
    except:
        with opener(filename,"r") as the_file:
            tweets = []
            for line in the_file:
                tweets.append(json.loads(line))
    if isinstance(tweets, dict):
        tweets = [tweets]
    ## Retweet Filter
    if exclude_retweets:
        tweets = list(filter(lambda t: not t["text"].startswith("RT ") or " RT " in t["text"], tweets))
    ## English Filter
    if exclude_non_english:
        tweets = list(filter(lambda t: t["lang"]=="en", tweets))
    return tweets

def get_location_info(place):
    """

    """
    atts = dict((i, None) for i in ["city",
                                    "county",
                                    "state",
                                    "country",
                                    "latitude",
                                    "longitude",
                                    "resolution_method"])
    for a in atts:
        if hasattr(place, a):
            atts[a] = getattr(place, a)
    return atts

def infer_demographics(filename,
                       exclude_retweets=True,
                       exclude_non_english=False):
    """

    """
    ## Load Filtered Tweets
    tweets = load_tweets(filename,
                         exclude_retweets,
                         exclude_non_english)
    ## Apply Demographic Classifiers
    demographics = []
    for tweet in tweets:
        ## Get Demographics
        demos = process_tweet(tweet, DEMOGRAPHERS)
        ## Get Locations and Update
        place = GEO_RESOLVER.resolve_tweet(tweet)
        if place is None:
            place = (None, None)
        demos["location"] = {**{"provisional":place[0]}, **get_location_info(place[1])}
        ## Cache
        demographics.append(demos)
    return demographics

def resolve_places_df(demographics,
                      min_threshold=0.25):
    """

    """
    places_df = pd.DataFrame([i["location"] for i in demographics])
    levels = ["city","county","state","country"]
    loc_tuple = None
    for l, level in enumerate(levels):
        level_tuple = places_df[levels[l:]].apply(tuple, axis=1)
        level_vc = level_tuple.value_counts(normalize=True)
        level_avail = level_vc.index.map(lambda i: i[0] != "")
        if sum(level_avail) > 0 and level_vc.loc[level_avail].max() > min_threshold:
            loc_tuple = level_vc.loc[level_avail].idxmax()
            break
    if loc_tuple is None:
        return None
    else:
        sol = places_df.loc[(places_df[levels[l:]].apply(tuple,axis=1)==loc_tuple)].copy()
        lat, lon = sol[["latitude","longitude"]].apply(tuple, axis=1).value_counts().idxmax()
        n = len(sol)
        sol = json.loads(sol.iloc[0].to_json())
        for c in levels[:l]:
            sol[c] = ""
        sol["latitude"] = lat
        sol["longitude"] = lon
        sol["resolution_level"] = level
        sol["resolution_matches"] = n
        sol["resolution_percent"] = n / len(places_df)
        return sol

def resolve_demographer_fields(demographics):
    """

    """
    fields = {}
    for demo in DEMOGRAPHERS:
        demo_set = [i[demo.name_key] for i in demographics]
        demo_set = [i for i in demo_set if i["value"] is not None]
        if len(demo_set) == 0:
            fields[demo.name_key] = {}
        else:
            demo_set = pd.DataFrame(demo_set)["value"].value_counts().idxmax()
            fields[demo.name_key] = {demo.name_key:demo_set}
    return fields

def resolve_file(filename,
                 exclude_retweets=True,
                 exclude_non_english=False,
                 min_res_threshold=0.5):
    """

    """
    ## Get Demographics 
    demographics = infer_demographics(filename,
                                      exclude_retweets,
                                      exclude_non_english)
    ## Extract Locations and Select Most Likely
    resolved_location = resolve_places_df(demographics, min_res_threshold)
    ## Extract Demographer Fields and Select Most Likely
    resolved_demos = resolve_demographer_fields(demographics)
    ## Combine
    resolutions = {
        **{"location":resolved_location},
        **resolved_demos
    }
    return filename, resolutions

def main():
    """

    """
    ## Get Raw Files
    filenames = glob(f"{RAW_DIR}*.json.gz")
    LOGGER.info(f"Found {len(filenames)} Files for Demographic Inference")
    ## Multiprocessing
    LOGGER.info("Starting Resolution")
    mp = Pool(JOBS)
    helper = partial(resolve_file,
                     exclude_retweets=EXCLUDE_RETWEETS,
                     exclude_non_english=EXCLUDE_NON_ENGLISH,
                     min_res_threshold=MIN_RES_THRESHOLD)
    res = list(tqdm(mp.imap_unordered(helper, filenames),
                    desc="Demographic Resolver",
                    total=len(filenames),
                    file=sys.stdout))
    mp.close()
    ## Concatenate
    LOGGER.info("Concatenating Results")
    res_df = pd.DataFrame(data=[r[1] for r in res], index=[r[0] for r in res])
    for c, cvals in COLUMN_MAP.items():
        shape = res_df.shape[1]
        for cv in cvals:
            res_df[cv] = res_df[c].map(lambda i: i.get(cv, None) if isinstance(i, dict) else None)
        if res_df.shape[1] != shape:
            res_df = res_df.drop([c],axis=1)
    res_df = res_df.reset_index().rename(columns={"index":"source"})
    res_df.rename(columns={"indorg_balanced":"indorg"},inplace=True)
    ## Dump
    LOGGER.info(f"Dumping Inferences to '{OUTFILE}'")
    res_df.to_csv(OUTFILE, index=False)
    LOGGER.info("Script Complete!")

####################
### Run
####################

if __name__ == "__main__":
    _ = main()
