

###################
### Configuration
###################

## Data Directories
DATA_DIR = "./data/results/twitter/2018-2020/falconet-keywords/"

## Random Sampling
SKIP_COVID=True
SAMPLE_RATE = 0.1
SAMPLE_SEED = 42

## Multiprocessing
NUM_JOBS = 4

###################
### Imports
###################

## Standard Library
import os
import re
import sys
import gzip
import json
import random
import string
from glob import glob
from functools import partial
from datetime import datetime
from multiprocessing import Pool
from collections import Counter

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

## Mental Health Library
from mhlib.util.logging import initialize_logger
from mhlib.util.helpers import flatten

###################
### Globals
###################

## Logging
LOGGER = initialize_logger()

## Date Resolution (Changing May Break Expected Visualizations/Analysis)
DATE_RES = "day"

## Special Characters
SPECIAL = "“”…‘’´"

###################
### Helpers
###################

def _format_timestamp(timestamp,
                      level="day"):
    """

    """
    level_func = {
            "hour": lambda x: (x.year, x.month, x.day, x.hour),
            "day":lambda x: (x.year, x.month, x.day),
            "month":lambda x: (x.year, x.month, 1),
            "year":lambda x: (x.year, 1, 1)
    }
    if level not in level_func:
        raise ValueError("'level' must be in {}".format(list(level_func.keys())))
    return level_func.get(level)(timestamp)

def format_timestamps(timestamps,
                      level="day"):
    """
    Args:
        timestamps (list):
        level (str): One of "hour", "day", "month", "year"
    """
    timestamps = list(map(lambda t: _format_timestamp(t, level), timestamps))
    return timestamps

###################
### Functions
###################

def create_regex_dict(terms,
                      include_hashtag=True):
    """

    """
    regex_dict = {}
    for t in terms:
        is_stem = t.endswith("*")
        t_stem = re.escape(t.rstrip("*"))
        if t_stem.isupper():
            regex_dict[t] = (re.compile(t_stem), is_stem)
            if include_hashtag:
                regex_dict["#"+t] = (re.compile("#" + t_stem), is_stem)
        else:
            regex_dict[t] = (re.compile(t_stem, re.IGNORECASE), is_stem)
            if include_hashtag:
                regex_dict["#"+t] = (re.compile("#"+t_stem, re.IGNORECASE), is_stem)
    return regex_dict

def _starts_with(text,
                 index,
                 prefix):
    """

    """
    i = index
    while i > 0:
        if text[i] == " ":
            return False
        if text[i] == prefix:
            return True
        i -= 1
    return False

def _search_full_words(text,
                       regex,
                       stem=False,
                       include_mentions=False):
    """

    """
    matches = []
    L = len(text)
    for match in regex.finditer(text):
        match_span = match.span()
        if include_mentions:
            starts_text = match_span[0] == 0 or \
                          text[match_span[0]-1] == " " or \
                          text[match_span[0]-1] in (string.punctuation + SPECIAL)
        else:
            starts_text =  match_span[0] == 0 or \
                           text[match_span[0]-1] == " " or \
                           (text[match_span[0]-1] in (string.punctuation + SPECIAL) and not _starts_with(text,match_span[0],"@"))
        ends_text = match_span[1] == L or text[match_span[1]] == " " or text[match_span[1]] in (string.punctuation + SPECIAL)
        if starts_text:
            if stem or ends_text:
                matches.append((text[match_span[0]:match_span[1]], match_span))
    return matches


def pattern_match(text,
                  pattern_re,
                  include_mentions=False):
    """

    """
    matches = []
    for keyword, (pattern, is_stem) in pattern_re.items():
        keyword_matches = _search_full_words(text, pattern, stem=is_stem, include_mentions=include_mentions)
        keyword_matches = [(keyword, k[0], k[1]) for k in keyword_matches]
        matches.extend(keyword_matches)
    return matches

def filter_substrings(term_matches):
    """

    """
    if len(term_matches) == 1:
        return term_matches
    term_matches_filtered = []
    for t1, term_match_1 in enumerate(term_matches):
        is_subinterval = False
        for t2, term_match_2 in enumerate(term_matches):
            if t1 == t2:
                continue
            if term_match_1[2][0] >= term_match_2[2][0] and term_match_1[2][1] <= term_match_2[2][1]:
                is_subinterval = True
                break
        if not is_subinterval:
            term_matches_filtered.append(term_match_1)
    return term_matches_filtered

def match_post(post,
               include_mentions=False):
    """

    """    
    ## Cycle Through Options
    match_cache = {}
    for category in MATCH_DICT.keys():
        term_matches = pattern_match(post["text"].replace("\n"," "), MATCH_DICT[category]["terms"], include_mentions=include_mentions)
        if len(term_matches) > 0:
            if category not in match_cache:
                match_cache[category] = {}
            match_cache[category]["terms"] = term_matches
    ## Filter Out Substrings
    if match_cache:
        for category, category_dict in match_cache.items():
            match_cache[category]["terms"] = filter_substrings(match_cache[category]["terms"])
    ## Metadata
    if not match_cache:
        return None
    post_match_cache = {"matches":match_cache}
    meta_cols = ["user_id","date","text","tweet_id"]
    for val in meta_cols:
        post_match_cache[val] = post[val]
    return post_match_cache

def find_matches(filename,
                 level=DATE_RES,
                 include_mentions=False):
    """

    """
    ## Initialize Sampler
    sampler = random.Random(SAMPLE_SEED)
    ## Search For Matches
    matches = []
    timestamps = []
    n = 0
    n_seen = 0
    with gzip.open(filename,"r") as the_file:
        for post in the_file:
            n += 1
            if sampler.uniform(0,1) >= SAMPLE_RATE:
                continue
            else:
                n_seen += 1
                ## Load Data
                post_data = json.loads(post)
                ## Cache Timestamp
                timestamps.append(pd.to_datetime(post_data["date"]))
                ## Regex Version
                post_regex_matches = match_post(post_data, include_mentions=include_mentions)
                ## Falconet Matches
                falconet_terms = post_data.get("keywords")
                ## Continue if None
                if post_regex_matches is None and falconet_terms is None:
                    continue
                else:
                    if post_regex_matches:
                        if SKIP_COVID and "covid" in post_regex_matches.get("matches"):
                            continue
                        regex_terms = [list(row) for _, row in pd.DataFrame(flatten([i["terms"] for i in post_regex_matches.get("matches").values()])).drop_duplicates(subset=[1,2]).iterrows()]
                        regex_terms = filter_substrings(regex_terms)
                        regex_terms = sorted([f[0] for f in regex_terms])
                    else:
                        regex_terms = []
                    if not falconet_terms:
                        falconet_terms = []
                    else:
                        falconet_terms = sorted(falconet_terms)
                    matches.append({
                        "tweet_id":post_data.get("tweet_id"),
                        "date":pd.to_datetime(post_data.get("date")),
                        "text":post_data.get("text"),
                        "regex_keywords":regex_terms,
                        "falconet_keywords":falconet_terms
                    })
    ## Format Timestamps
    timestamps = format_timestamps(timestamps, level)
    timestamps = Counter(timestamps)
    return filename, matches, n, n_seen, timestamps

def search_files(filenames,
                 date_res="day",
                 include_mentions=False):
    """

    """

    ## Run Lookup
    mp = Pool(NUM_JOBS)
    helper = partial(find_matches, level=date_res, include_mentions=include_mentions)
    res = list(tqdm(mp.imap_unordered(helper, filenames), total=len(filenames), desc="Searching For Matches", file=sys.stdout))
    mp.close()
    ## Parse
    filenames = [r[0] for r in res]
    matches = [r[1] for r in res]
    n = [r[2] for r in res]
    n_seen = [r[3] for r in res]
    timestamps = [r[4] for r in res]
    return filenames, matches, n, n_seen, timestamps

###################
### Terms + Subreddits
###################

## Load Georgetown Mental Health Resources (SMHD)
MH_TERM_FILE = "./data/resources/mental_health_terms.json"
with open(MH_TERM_FILE,"r") as the_file:
    MH_TERMS = json.load(the_file)

## Load Crisis Keywords (JHU Behavioral Health)
CRISIS_KEYWORD_FILES = glob("./data/resources/*crisis*.keywords")
CRISIS_KEYWORDS = set()
for f in CRISIS_KEYWORD_FILES:
    fwords = [i.strip() for i in open(f,"r").readlines()]
    fwords = list(map(lambda i: i.lower() if not i.isupper() else i, fwords))
    CRISIS_KEYWORDS.update(fwords)

## Load JHU CLSP Mental Health Keywords
MH_KEYWORDS_FILE = "./data/resources/mental_health_keywords_manual_selection.csv"
MH_KEYWORDS = pd.read_csv(MH_KEYWORDS_FILE)
MH_KEYWORDS = set(MH_KEYWORDS.loc[MH_KEYWORDS["ignore_level"].isnull()]["ngram"])
MH_KEYWORDS.add("depression")
MH_KEYWORDS.add("depressed")

## Load COVID Terms/Subreddits
COVID_TERM_FILE = "./data/resources/covid_terms.json"
with open(COVID_TERM_FILE,"r") as the_file:
    COVID_TERMS = json.load(the_file)

###################
### Identify Matches
###################

## Create Match Dictionary (Subreddit Lists + Term Regular Expressions)
MATCH_DICT = {
    "mental_health":{
        "terms":create_regex_dict(MH_TERMS["terms"]["smhd"]),
        "name":"SMHD"
    },
    "crisis":{
        "terms":create_regex_dict(CRISIS_KEYWORDS),
        "name":"JHU Crisis"
    },
    "mental_health_keywords":{
        "terms":create_regex_dict(MH_KEYWORDS),
        "name":"JHU CLSP"
    },
    "covid":{
        "terms":create_regex_dict(COVID_TERMS["covid"]),
        "name":"COVID-19"
    }
}

## Find Procesed Files
filenames = sorted(glob(f"{DATA_DIR}*.json.gz"))

## Search For Keyword/Subreddit Matches
filenames, matches, n, n_seen, timestamps = search_files(filenames,
                                                         date_res=DATE_RES,
                                                         include_mentions=False)

## Disagreement
disagreements = [i for i in flatten(matches) if i["regex_keywords"] != i["falconet_keywords"]]
disagreement_rate = len(disagreements) / len(flatten(matches)) * 100
print("Disagreement Rate: {:.3f}%".format(disagreement_rate))

## Count Comparision
regex_counts = pd.Series(Counter(flatten([i["regex_keywords"] for i in flatten(matches)]))).to_frame("regex")
falconet_counts =  pd.Series(Counter(flatten([i["falconet_keywords"] for i in flatten(matches)]))).to_frame("falconet")
merged_counts = pd.concat([regex_counts, falconet_counts],axis=1,sort=True).fillna(0)
outliers = np.log((merged_counts["regex"] + 0.01) / (merged_counts["falconet"] + 0.01)).sort_values()

## Plot Comparison
fig, ax = plt.subplots(1, 2, figsize=(10,5.8))
merged_counts.plot.scatter("regex","falconet",ax=ax[0])
ax[0].plot([0, regex_counts.max().item()],[0, regex_counts.max().item()], color="black", zorder=10)
outliers.head(15).append(outliers.tail(15)).plot.barh(ax=ax[1], alpha=0.8)
ax[1].axvline(0, color="black", alpha=0.8, linestyle="--")
ax[0].set_xlabel("Regex Count")
ax[0].set_ylabel("Falconet Count")
ax[0].set_yscale("symlog")
ax[0].set_xscale("symlog")
ax[1].set_xlabel("Regex to Falconet Ratio (log-scale)")
fig.tight_layout()
plt.close()