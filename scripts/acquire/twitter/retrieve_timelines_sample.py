
#######################
### Imports
#######################

## Standard Library
import os
import json
import gzip
import argparse

## External Libraries
from mhlib.util.logging import initialize_logger

#######################
### Configuration
#######################

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
    parser.add_argument("file_list",
                        type=str,
                        help="Path to list of files containing Tweet samples (.txt file, newline delimited)")
    parser.add_argument("output_dir",
                        type=str,
                        help="Where to store tweets")
    parser.add_argument("file_ind",
                        type=int,
                        help="Which file to process")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if not os.path.exists(args.user_list):
        raise FileNotFoundError(f"Could not find user list file {args.user_list}")
    if not os.path.exists(args.file_list):
        raise FileNotFoundError(f"Could not find file list file {args.file_list}")
    return args

def main():
    """

    """
    ## Parse Command Line
    args = parse_arguments()
    ## Create Output Directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    ## Load User List
    users = set([i.strip() for i in open(args.user_list,"r").readlines()])
    ## Load File List
    filelist = [i.strip() for i in open(args.file_list,"r").readlines()]
    ## Get Desired File
    if args.file_ind >= len(filelist):
        LOGGER.info("File Index Out of Range. Exiting.")
        exit()
    filename = filelist[args.file_ind-1]
    ## Load and Cache Matched Tweets
    tweet_cache = []
    with gzip.open(filename, "r") as the_file:
        try:
            for line in the_file:
                try:
                    line_data = json.loads(line)
                except:
                    continue
                if "user" in line_data and line_data["user"]["id_str"] in users:
                    tweet_cache.append(line_data)
        except:
            pass
    ## Early Exit (No-matches)
    if len(tweet_cache) == 0:
        exit()
    ## Sort Tweet Cache Based on User ID
    tweet_cache = sorted(tweet_cache, key = lambda x: x["user"]["id_str"])
    ## Create Cache Directory
    outdir = "{}temp/{}/".format(args.output_dir, os.path.basename(filename).rstrip(".gz"))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    ## Cache Users
    user_id = tweet_cache[0]["user"]["id_str"]
    user_cache = [tweet_cache[0]]
    for t in tweet_cache[1:]:
        if t["user"]["id_str"] == user_id:
            user_cache.append(t)
        else:
            user_outfile = f"{outdir}{user_id}.json.gz"
            with gzip.open(user_outfile,"wt",encoding="utf-8") as the_file:
                json.dump(user_cache, the_file)
            user_cache = [t]
            user_id = t["user"]["id_str"]
    user_outfile = f"{outdir}{user_id}.json.gz"
    with gzip.open(user_outfile,"wt",encoding="utf-8") as the_file:
        json.dump(user_cache, the_file)
    ## Done
    LOGGER.info("Script Complete!")

#####################
### Run
#####################

if __name__ == "__main__":
    _ = main()