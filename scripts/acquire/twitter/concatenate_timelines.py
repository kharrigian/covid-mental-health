
## Where temp directory of data files currently lives
DATA_DIR = "./data/raw/twitter/timelines/"

## Whether or not to remove the temporary data directory
REMOVE_TEMP = False

## Multiprocessing Jobs
NUM_JOBS = 16

##################
### Imports
##################

## Standard Library
import os
import sys
import json
import gzip
from glob import glob
from multiprocessing import Pool

## External 
from tqdm import tqdm
from mhlib.util.logging import initialize_logger

##################
### Globals
##################

## Logger
LOGGER = initialize_logger()

##################
### Helpers
##################

## Concatenation
def concatenate_user(user):
    """

    """
    ## Get Files
    files = user_files[user]
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

##################
### Concatenate User Files
##################

## Find Files in Temp Directory
LOGGER.info("Identifying Temporary User Files")
files = glob(f"{DATA_DIR}temp/*/*.gz")

## Map Users to their Files
get_user_id = lambda i: os.path.basename(i).rstrip(".json.gz")
user_files = {}
for f in tqdm(files, desc="Grouping User Files"):
    uid = get_user_id(f)
    if uid not in user_files:
        user_files[uid] = []
    user_files[uid].append(f)

## Cycle Through Users
mp = Pool(NUM_JOBS)
_ = list(tqdm(mp.imap_unordered(concatenate_user, sorted(user_files.keys())),
              total=len(user_files),
              desc="Concatenating User Files",
              file=sys.stdout))
mp.close()

## Remove Temp Directory
if REMOVE_TEMP:
    LOGGER.info("Removing Temporary Directory")
    folders = glob(f"{DATA_DIR}temp/*")
    for f in tqdm(folders, file=sys.stdout):
        _ = os.system("rm -rf {}".format(f))
    _ = os.system(f"rm -rf {DATA_DIR}temp/")

LOGGER.info("Script complete!")