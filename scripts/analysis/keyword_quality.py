

##########################
### Configuration
##########################

## Cache Directory
# CACHE_DIR = "./data/results/reddit/2017-2020/keywords-subreddits/"
CACHE_DIR = "./data/results/twitter/2018-2020/keywords/"

## Platform
# PLATFORM = "reddit"
PLATFORM = "twitter"

## Language Date Boundaries
START_DATE = "2019-01-01"
END_DATE = "2020-06-15"

## Sampling
NUM_SAMPLES_PER_TERM = 10
RANDOM_STATE = 42

##########################
### Imports
##########################

## Standard Lbrary
import os
import sys
import json
import gzip
from glob import glob
from datetime import datetime
from copy import deepcopy
from pprint import PrettyPrinter
from collections import Counter

## External
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from mhlib.util.helpers import flatten

##########################
### Globals
##########################

PRINTER = PrettyPrinter(width=80)

##########################
### Helpers
##########################

def load_keyword_search_results(match_cache_dir):
    """

    """
    ## Load Keyword Search Results
    filenames = []
    matches = []
    n = []
    n_seen = []
    timestamps = []
    ## Load Matches
    match_files = glob(f"{match_cache_dir}*.joblib")
    for match_file in tqdm(match_files, desc="Filename"):
        match_data = joblib.load(match_file)
        filenames.append(match_data.get("filename"))
        matches.append(match_data.get("matches"))
        n.append(match_data.get("n"))
        n_seen.append(match_data.get("n_seen"))
        timestamps.append(match_data.get("timestamps"))
    n = np.array(n)
    n_seen = np.array(n_seen)
    return filenames, matches, n, n_seen, timestamps

def get_unique_terms(matches):
    """

    """
    terms = {}
    for m in flatten(matches):
        for match_key, match_dict in m.get("matches").items():
            if "terms" not in match_dict:
                continue
            if match_key not in terms:
                terms[match_key] = set()
            for (term, _, _) in match_dict.get("terms"):
                terms[match_key].add(term)
    return terms

def examine_matches(matches,
                    match_key,
                    match_type,
                    query):
    """

    """
    relevant_matches = []
    for m in flatten(matches):
        if match_key not in m["matches"]:
            continue
        if match_type not in m["matches"][match_key]:
            continue
        m_res = m.get("matches").get(match_key).get(match_type)
        m_res_present = [(x[0], x[-1]) for x in m_res if x[0] == query]
        if len(m_res_present) > 0:
            m_text = m["text"]
            m_text_highlighted = ""
            start = 0
            end = len(m_text)
            for _, (term_start, term_end) in m_res_present:
                m_text_highlighted += m_text[start:term_start] + "<" + m_text[term_start:term_end] + ">"
                start = term_end
            if start != end:
                m_text_highlighted += m_text[start:end]
            m_copy = deepcopy(m)
            _ = m_copy.pop("matches",None)
            m_copy["text"] = m_text_highlighted
            relevant_matches.append(m_copy)
    return relevant_matches

def label_row_indication_strength(row):
    """

    """
    term = row["term"]
    text = row["text"]
    date = row["date"]
    print_data = {"DATE":date,"TEXT":text}
    print_str = "{}\n{}".format(f"\n~~ TERM: {term} ~~\n", PRINTER.pformat(print_data))
    label_opt_str = "Label Selection:\n[1] No Indication\n[2] Possible Indication\n[3] Strong Indication"
    print(print_str); print(label_opt_str)
    label_str = int(input("Label Selection: "))
    return (term, row["post_id"], label_str)

###################
### Load/Parse Matches
###################

## Sample Cache File
sample_cache_file = f"{CACHE_DIR}{PLATFORM}_keyword_samples_k-{NUM_SAMPLES_PER_TERM}.json"

## Run Sampling If Necessary
if not os.path.exists(sample_cache_file):

    ## Load Matches
    match_cache_dir = f"{CACHE_DIR}{PLATFORM}_{START_DATE}_{END_DATE}_matches/"
    filenames, matches, n, n_seen, timestamps = load_keyword_search_results(match_cache_dir)

    ## Unique Query Terms
    unique_terms = get_unique_terms(matches)
    unique_terms_df = pd.DataFrame(data=sorted(set(flatten(unique_terms.values()))),
                                   columns=["term"])
    unique_terms_df["keyword_group"] = [[]] * len(unique_terms_df)
    for term_group, terms in unique_terms.items():
        unique_terms_df["keyword_group"] = unique_terms_df.apply(lambda row: [term_group] + row["keyword_group"] if row["term"] in terms else row["keyword_group"], axis=1)
        
    ## Get Match Sizes
    match_sizes = {term:0 for term in unique_terms_df["term"]}
    for match_set in matches:
        for post in match_set:
            for match_key, match_values in post.get("matches").items():
                if "terms" not in match_values:
                    continue
                terms_present = [t[0] for t in match_values.get("terms")]
                for t in terms_present:
                    if t not in match_sizes:
                        continue
                    match_sizes[t] += 1
    unique_terms_df["num_matches"] = unique_terms_df["term"].map(lambda i: match_sizes.get(i))

    ## Sample Matches (Leverage Temporal Distribution)
    samples = []
    np.random.seed(RANDOM_STATE)
    for _, term_data in tqdm(unique_terms_df.iterrows(), total=len(unique_terms_df), desc="Term Sampling"):
        term_matches = pd.DataFrame(examine_matches(matches, term_data.loc["keyword_group"][0], "terms", term_data.loc["term"]))
        term_matches["month"] = term_matches["created_utc"].map(datetime.fromtimestamp).map(lambda i: (i.year, i.month))
        month_dist = term_matches["month"].value_counts(normalize=True)
        term_matches["sample_prob"] = (month_dist.loc[term_matches.month]).values
        term_matches_sample = term_matches.sample(min(NUM_SAMPLES_PER_TERM, len(term_matches)), weights="sample_prob", replace=False)
        term_matches_sample_simple = []
        for _,sample in term_matches_sample.iterrows():
            sample_simple = {
                        "term":term_data.loc["term"],
                        "text":sample.loc["text"],
                        "date":datetime.fromtimestamp(sample.loc["created_utc"]).date(),
                        "user_id_str":sample.loc["user_id_str"],
                        "post_id":sample.loc["tweet_id"] if PLATFORM == "twitter" else sample.loc["comment_id"]
            }
            term_matches_sample_simple.append(sample_simple)
        samples.extend(term_matches_sample_simple)

    ## Cache Samples
    with open(sample_cache_file, "w") as the_file:
        for sample in samples:
            sample["date"] = sample["date"].isoformat()
            the_file.write("{}\n".format(json.dumps(sample)))

    ## Cache Metadata
    unique_terms_df["keyword_group"]= unique_terms_df["keyword_group"].map(lambda i: ", ".join(i))
    unique_terms_df.to_csv(f"{CACHE_DIR}{PLATFORM}_keyword_samples_metadata.csv", index=False)

## Load Existing Samples if Already Complete
else:

    ## Load Samples From Cache
    samples = []
    with open(sample_cache_file, "r") as the_file:
        for sample in the_file:
            samples.append(json.loads(sample))
    
    ## Load Metadata From Cache
    unique_terms_df = pd.read_csv(f"{CACHE_DIR}{PLATFORM}_keyword_samples_metadata.csv")
    unique_terms_df["keyword_group"] = unique_terms_df["keyword_group"].str.split(", ")

## Format Samples
samples = pd.DataFrame(samples)
samples = samples.sort_values("term", ascending=True).reset_index(drop=True)

###################
### Annotate Data (First Pass: Indication of Mental Health Status)
###################

## Load/Initialize Annotation Cahce
indicator_annotation_file = f"{CACHE_DIR}{PLATFORM}_keyword_indictor_annotations_k-{NUM_SAMPLES_PER_TERM}.json"
indicator_annotations = {}
if os.path.exists(indicator_annotation_file):
    with open(indicator_annotation_file, "r") as the_file:
        indicator_annotations = json.load(the_file)

## Run Labelling Procedure
last_seen_term = ""
for _, sample_row in samples.iterrows():
    need_to_label = False
    if sample_row["term"] not in indicator_annotations:
        indicator_annotations[sample_row["term"]] = {}
        need_to_label = True
    if sample_row["post_id"] not in indicator_annotations[sample_row["term"]]:
        need_to_label = True
    if need_to_label:
        term, post_id, lbl = label_row_indication_strength(sample_row)
        indicator_annotations[term][post_id] = lbl
        if term != last_seen_term:
            with open(indicator_annotation_file, "w") as the_file:
                json.dump(indicator_annotations, the_file)
            last_seen_term = term

## Merge Annotations With Samples
samples["label"] = samples.apply(lambda row: indicator_annotations.get(row["term"]).get(row["post_id"]) \
                                 if indicator_annotations.get(row["term"]) is not None else None, axis=1)

###################
### Review Distribution
###################

## Format Indicator Counts
indicator_counts = dict((term, Counter(anots.values())) for term, anots in indicator_annotations.items())
indicator_counts = pd.DataFrame(indicator_counts).T[[1,2,3]].fillna(0)

## Compute Expected Scores
exp_indicator_count = ((indicator_counts[1] + 2 * indicator_counts[2] + 3 * indicator_counts[3]) / indicator_counts.sum(axis=1)).sort_values(ascending=False)