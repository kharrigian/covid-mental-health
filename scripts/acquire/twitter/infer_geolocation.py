

## Inputs
RAW_DIR = "./data/raw/twitter/timelines/"

## Outputs
LOC_DIR = "./data/processed/twitter/geolocation/"
LOC_OUTFILE = "./data/processed/twitter/user_geolocation.csv"

## Resolving Parameters
EXCLUDE_RETWEETS=True
EXCLUDE_NON_ENGLISH=False
EXCLUDE_PROVISIONAL=False
MIN_RES_THRESHOLD = 0.3

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
import carmen # https://github.com/mdredze/carmen-python
import pandas as pd
from tqdm import tqdm
from mhlib.util.logging import initialize_logger

#######################
### Globals
#######################

## Logging
LOGGER = initialize_logger()

## Geolocation Resolver
GEO_RESOLVER = carmen.get_resolver(order=["place","geocode","profile"])
GEO_RESOLVER.load_locations()

#######################
### Helpers
#######################

def load_tweets(filename,
                exclude_retweets=True,
                exclude_non_english=False):
    """

    """
    ## Load
    with gzip.open(filename,"r") as the_file:
        tweets = json.load(the_file)
    ## Retweet Filter
    if exclude_retweets:
        tweets = list(filter(lambda t: not t["text"].startswith("RT "), tweets))
    ## English Filter
    if exclude_non_english:
        tweets = list(filter(lambda t: t["lang"]=="en", tweets))
    return tweets

def get_location_info(place):
    """

    """
    atts = dict((i, None) for i in ["city","county","state","country","latitude","longitude","resolution_method"])
    for a in atts:
        if hasattr(place, a):
            atts[a] = getattr(place, a)
    return atts

def geocode_file(filename,
                 exclude_retweets=True,
                 exclude_non_english=False,
                 exclude_provisional=True):
    """

    """
    ## Load Filtered Tweets
    tweets = load_tweets(filename,
                         exclude_retweets,
                         exclude_non_english)
    ## Resolve
    places = list(map(lambda i: (i["id_str"], GEO_RESOLVER.resolve_tweet(i)), tweets))
    ## Filter Out Nulls
    places = list(filter(lambda p: p[1] is not None, places))
    ## Filter Out Provisional
    if exclude_provisional:
        places = list(filter(lambda p: not p[1][0], places))
    ## Format
    places = list(map(lambda j: (j[0], get_location_info(j[1][1])), places))
    return places

def resolve_places_df(places_df,
                      min_threshold=0.25):
    """

    """
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

def resolve_file(filename,
                 exclude_retweets=True,
                 exclude_non_english=False,
                 exclude_provisional=True,
                 min_res_threshold=0.5):
    """

    """
    ## Output File
    file_id = os.path.basename(filename)
    outfile = f"{LOC_DIR}{file_id}"
    ## Get Places 
    places = geocode_file(filename,
                          exclude_retweets,
                          exclude_non_english,
                          exclude_provisional)
    ## Cache
    if len(places) == 0:
        with gzip.open(outfile,"wt") as the_file:
            the_file.write("{}\n")
        return None
    else:
        places = dict((p[0], p[1]) for p in places)
        with gzip.open(outfile,"wt") as the_file:
            for x, y in places.items():
                the_file.write(json.dumps({x:y})+"\n")
    ## Resolve Places Across User Tweets
    places_df = pd.DataFrame(places).T
    try:
        places_res = resolve_places_df(places_df,
                                       min_res_threshold)
    except:
        LOGGER.info(f"Issue with {filename}")
        places_res = None
    return filename, places_res

def main():
    """

    """
    ## Get Raw Files
    filenames = glob(f"{RAW_DIR}*.json.gz")
    LOGGER.info(f"Found {len(filenames)} Files for Geolocation")
    ## Output Directory
    if not os.path.exists(LOC_DIR):
        os.makedirs(LOC_DIR)
    ## Multiprocessing
    LOGGER.info("Starting Resolution")
    mp = Pool(JOBS)
    helper = partial(resolve_file,
                     exclude_retweets=EXCLUDE_RETWEETS,
                     exclude_non_english=EXCLUDE_NON_ENGLISH,
                     exclude_provisional=EXCLUDE_PROVISIONAL,
                     min_res_threshold=MIN_RES_THRESHOLD)
    res = list(tqdm(mp.imap_unordered(helper, filenames),
                    desc="Geolocation Resolver",
                    total=len(filenames),
                    file=sys.stdout))
    mp.close()
    ## Concatenate
    LOGGER.info("Concatenating Results")
    res_df = pd.DataFrame(dict((r[0], r[1]) for r in res if r is not None)).T
    res_df = res_df.reset_index().rename(columns={"index":"source"})
    ## Dump
    LOGGER.info(f"Dumping User-level Resolutions to '{LOC_OUTFILE}'")
    res_df.to_csv(LOC_OUTFILE, index=False)
    LOGGER.info("Script Complete!")

####################
### Run
####################

if __name__ == "__main__":
    _ = main()
