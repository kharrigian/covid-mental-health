
## Where temp directory of data files currently lives
DATA_DIR = "./data/raw/twitter/timelines/"

##################
### Imports
##################

## Standard Library
import os
import sys
import json
import gzip
from glob import glob

## External 
from tqdm import tqdm

##################
### Concatenate User Files
##################

## Find Files in Temp Directory
files = glob(f"{DATA_DIR}temp/*/*.gz")

## Map Users to their Files
get_user_id = lambda i: os.path.basename(i).rstrip(".json.gz")
user_files = {}
for f in files:
    uid = get_user_id(f)
    if uid not in user_files:
        user_files[uid] = []
    user_files[uid].append(f)

## Cycle Through Users
for user, files in tqdm(user_files.items(),
                        total=len(user_files),
                        desc="Concatenating User",
                        file=sys.stdout):
    ## Combine Tweets (Unique)
    user_cache = []
    tweet_ids = set()
    for f in files:
        with gzip.open(f, "r") as the_file:
            fdata = json.load(the_file)
            fdata = [t for t in fdata if t["id_str"] not in tweet_ids]
            user_cache.extend(fdata)
            tweet_ids.update([t["id_str"] for t in fdata])
    ## Cache
    outfile = f"{DATA_DIR}{user}.json.gz"
    with gzip.open(outfile, "wt", encoding="utf-8") as the_file:
        json.dump(user_cache, the_file)

## Remove Temp Directory
_ = os.system("rm -rf {DATA_DIR}temp/")