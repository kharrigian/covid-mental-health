
"""
Robustness Metrics:
- Number of Matches (Overall)
- Match Rate (Overall)
- Percentage of Matches with COVID Terms
- Largest Single Day Increase
- Neighbor N-Grams
"""

#######################
### Configuration
#######################

## Path to Falconet Output
DATA_DIR = "./data/results/twitter/2018-2020/falconet-keywords/"

## Path to Cache Directory/Plot Directory
CACHE_DIR = "./data/results/twitter/2018-2020/falconet-keywords/summary/"
PLOT_DIR = "./plots/twitter/2018-2020/falconet-keywords/"

## Filters
INDORGS = ["ind"]
GENDERS = ["man","woman"]
LOCATIONS = {"country":["United States"]}

## Context Analysis Parameters
CONTEXT_WINDOW = None
NGRAMS = (1,2)
MIN_FREQ = 10
MIN_CONTEXT_FREQ = 5
VOCAB_CUTOFF = 3
VOCAB_SIZE = 250000
SMOOTHING = 0.01
SMOOTHING_WINDOW = 30
PRE_COVID_WINDOW = ["2019-03-19","2019-08-01"]
POST_COVID_WINDOW = ["2020-03-19","2020-08-01"]

## Meta Parameters
NUM_JOBS = 8
RERUN = True

#######################
### Imports
#######################

## Standard Library
import os
import re
import sys
import gzip
import json
import string
import textwrap
from glob import glob
from datetime import datetime
from dateutil.parser import parse
from collections import Counter
from multiprocessing import Pool
from functools import partial

## External Libraries
import demoji
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.stats.proportion import proportion_confint

## Mental Health Specific
from mhlib.util.logging import initialize_logger
from mhlib.preprocess.tokenizer import Tokenizer, get_ngrams
from mhlib.model.data_loaders import LoadProcessedData
from mhlib.util.helpers import flatten, chunks

#######################
### Globals
#######################

## Logger
LOGGER = initialize_logger()

## Plot/Cache Directories
for d in [PLOT_DIR, CACHE_DIR]:
    if not os.path.exists(d):
        _ = os.makedirs(d)
for i in ["timeseries","context"]:
    if not os.path.exists(f"{PLOT_DIR}{i}/"):
        _ = os.makedirs(f"{PLOT_DIR}{i}/")
if not os.path.exists(f"{CACHE_DIR}examples/"):
    _ = os.makedirs(f"{CACHE_DIR}examples/")

## Tokenizer
TOKENIZER = Tokenizer(stopwords=set(),
                      keep_case=False,
                      negate_handling=False,
                      negate_token=False,
                      keep_punctuation=False,
                      keep_numbers=False,
                      expand_contractions=True,
                      keep_user_mentions=False,
                      keep_pronouns=True,
                      keep_url=False,
                      keep_hashtags=True,
                      keep_retweets=False,
                      emoji_handling=None,
                      strip_hashtag=False)

## Regex Rules
SPECIAL = "“”…‘’´"
INCLUDE_HASHTAGS = True

## Logical
COVID_START = pd.to_datetime(POST_COVID_WINDOW[0]).date()

#######################
### Reference Keywords
#######################

## Mental Health/Coronavirus Keywords
falconet_keywords = {}
falconet_keyword_dir="./data/resources/falconet/"
for mhlist, mhfile in [("Crisis (Level 1)", "crisis_level1.keywords"),
                       ("Crisis (Level 2)", "crisis_level2.keywords"),
                       ("Crisis (Level 3)", "crisis_level3.keywords"),
                       ("SMHD", "smhd.keywords"),
                       ("CLSP", "pmi.keywords"),
                       ("Coronavirus", "corona_virus.keywords")]:
    mhkeys = list(map(lambda i: i.strip(), open(f"{falconet_keyword_dir}{mhfile}","r").readlines()))
    mhkeys = sorted(set(flatten([[i, i.lower()] for i in mhkeys])))
    mhkeys = flatten([i, f"#{i}"] for i in mhkeys)
    falconet_keywords[mhlist] = mhkeys

## Reverse Mental Health Keyword List
falconet_keywords_reverse = dict()
for mhlist, terms in falconet_keywords.items():
    for t in terms:
        if t not in falconet_keywords_reverse:
            falconet_keywords_reverse[t] = []
        falconet_keywords_reverse[t].append(mhlist)

#######################
### Helpers
#######################

def filter_post(post,
                indorgs,
                genders,
                locations):
    """
    Args:
        post (dict): Post data (JSON format)
        indorgs (list or None): Restrict to these indorg classifications if desired
        genders (list or None): Restrict to these gender classifications if desired
        locations (dict or None): Restrict to {"country":[],...} if desired. Empty set 
                                  signifies any value
    
    Returns:
        filter_post (bool): True if post should be ignored else False
    """
    ## Get Fields
    post_demos = post.get("demographics")
    post_locs = post.get("location")
    ## Filtering
    if indorgs and post_demos.get("indorg") not in indorgs:
        return True
    if genders and post_demos.get("gender") not in genders:
        return True
    if locations:
        if post_locs is None:
            return True
        no_loc = False
        for key, values in locations.items():
            if key not in post_locs:
                no_loc = True
                break
            if len(values) != 0 and post_locs.get(key) not in values:
                no_loc = True
                break
        if no_loc:
            return True
    return False

def load_file(filename,
              indorgs=None,
              genders=None,
              locations=None):
    """
    Args:
        filename (str): Path to Falconet output
        indorgs (list or None): Restrict to these indorg classifications if desired
        genders (list or None): Restrict to these gender classifications if desired
        locations (dict or None): Restrict to {"country":[],...} if desired. Empty set 
                                  signifies any value
    """
    ## Prepare Filters
    if indorgs is not None:
        indorgs = set(indorgs)
    if genders is not None:
        genders = set(genders)
    if locations is not None:
        for key in locations.keys():
            locations[key] = set(locations[key])
    ## Load Data
    data = []
    with gzip.open(filename,"r") as the_file:
        for line in the_file:
            post = json.loads(line)
            if filter_post(post,
                           indorgs,
                           genders,
                           locations):
                continue
            data.append(post)
    return data

def jaccard(a, b):
    """

    """
    if not isinstance(a, list) or not isinstance(b, list):
        return 0
    if len(a) == 0 and len(b) == 0:
        return 0
    js = len(set(a) & set(b)) / len(set(a) | set(b))
    return js

def count_keywords_in_file(filename,
                           indorgs=None,
                           genders=None,
                           locations=None):
    """
    Args:
        filename (str): Path to output file from Falconet pipeline
        indorgs (list or None): Restrict to these indorg classifications if desired
        genders (list or None): Restrict to these gender classifications if desired
        locations (dict or None): Restrict to {"country":[],...} if desired. Empty set 
                                  signifies any value
    """
    ## Storage
    n_posts = Counter()
    keywords_by_date = dict()
    ## Load File
    f_data = load_file(filename, indorgs, genders, locations)
    ## Count Keywords in Posts
    for post in f_data:
        ## Count Date
        post_date = parse(post.get("date")).date()
        n_posts[post_date] += 1
        ## Count Keywords
        post_keywords = post.get("keywords")
        if post_keywords is not None:
            post_keyword_counts = Counter(list(set(post_keywords)))
            if post_date not in keywords_by_date:
                keywords_by_date[post_date] = []
            keywords_by_date[post_date].append(post_keyword_counts)
    ## Sum Keywords
    for date, keyword_list in keywords_by_date.items():
        temp_date_count = Counter()
        for l in keyword_list:
            temp_date_count += l
        keywords_by_date[date] = temp_date_count
    return n_posts, keywords_by_date

def _count_fields(filename,
                  fields=[],
                  indorgs=None,
                  genders=None,
                  locations=None,
                  keys=["date","demographics","location"]):
    """

    """
    ## counts
    timestamp_counts = Counter()
    field_counts = {field:{} for field in fields}
    ## Parse File
    f_data = load_file(filename, indorgs, genders, locations)
    for line in f_data:
        ## Extract Timestamp At Desired Resoulution
        post_date = parse(line.get("date")).date().isoformat()
        ## Identify Line Key
        line_key = []
        if "date" in keys:
            line_key.append(post_date)
        if "demographics" in keys:
            line_key.append(line.get("demographics").get("gender","MISSING"))
            line_key.append(line.get("demographics").get("indorg","MISSING"))
        if "location" in keys:
            if line.get("location") is None:
                line_key.append("MISSING")
                line_key.append("MISSING")
                line_key.append("MISSING")
            else:
                country = line.get("location").get("country")
                state = line.get("location").get("state")
                if country is None:
                    country = "MISSING"
                if state is None:
                    state = "MISSING"
                line_key.append(country)
                line_key.append(state)
                line_key.append("U.S." if country is not None and country == "United States" else "Non-U.S.")
        line_key = tuple(line_key)
        ## Update Counts
        timestamp_counts[line_key] += 1
        for field in fields:
            if line_key not in field_counts[field]:
                field_counts[field][line_key] = Counter()
            if field == "keywords" and line.get("keywords") is not None:
                line_key_counts = Counter(list(set(line.get("keywords"))))
                field_counts[field][line_key] += line_key_counts
    return field_counts, timestamp_counts

def count_fields(filenames,
                 fields=[],
                 indorgs=None,
                 genders=None,
                 locations=None,
                 keys=["date","demographics","location"]):
    """

    """
    ## Initialize Helper
    helper = partial(_count_fields,
                     keys=keys,
                     fields=fields,
                     indorgs=indorgs,
                     genders=genders,
                     locations=locations)
    ## Initialize and Execute Multiprocessing of Counts
    pool = Pool(NUM_JOBS)
    results = list(tqdm(pool.imap_unordered(helper, filenames), total=len(filenames), desc="Counter", file=sys.stdout))
    pool.close()
    ## Parse Results
    field_counts = {}
    timestamps = Counter()
    for fieldcount, timecount in results:
        timestamps += timecount
        for field, field_dict in fieldcount.items():
            if field not in field_counts:
                field_counts[field] = {}
            for key, counts in field_dict.items():
                if key not in field_counts[field]:
                    field_counts[field][key] = Counter()
                field_counts[field][key] += counts
    ## Column Order
    columns = []
    for k in keys:
        if k == "date":
            columns.append("date")
        if k == "demographics":
            columns.extend(["gender","indorg"])
        if k == "location":
            columns.extend(["country","state","is_united_states"])
    ## Format Timestamps
    timestamps = pd.Series(timestamps).reset_index().rename(columns=dict((f"level_{c}", col) for c, col in enumerate(columns)))
    timestamps = timestamps.rename(columns={0:"count"})
    ## Format Fields
    field_dfs = {}
    for field, field_count in field_counts.items():
        field_df = pd.DataFrame(field_count).T.reset_index().rename(columns=dict((f"level_{c}", col) for c, col in enumerate(columns)))
        field_df = field_df.set_index(columns)
        field_dfs[field] = field_df
    field_dfs = pd.concat(field_dfs)
    return field_dfs, timestamps 

def keyword_comorbidity(filename,
                        comparison_set,
                        indorgs=None,
                        genders=None,
                        locations=None):
    """

    """
    ## Storage
    comorbidity = Counter()
    morbidity = Counter()
    ## Update Set Type
    if not isinstance(comparison_set, set):
        comparison_set = set(comparison_set)
    ## Load File
    f_data = load_file(filename, indorgs, genders, locations)
    ## Filter No Keywords
    f_data = list(filter(lambda p: p.get("keywords"), f_data))
    ## Get Comorbidity
    for post in f_data:
        post_keywords = set(post.get("keywords"))
        for i, p in enumerate(post_keywords):
            morbidity[p] += 1
            comparison_present = False
            for j, pj in enumerate(post_keywords):
                if pj in comparison_set:
                    comparison_present = True
            if comparison_present:
                comorbidity[p] += 1
    return morbidity, comorbidity

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
            starts_text = match_span[0] == 0 or text[match_span[0]-1] == " " or text[match_span[0]-1] in (string.punctuation + SPECIAL)
        else:
            starts_text =  match_span[0] == 0 or text[match_span[0]-1] == " " or (text[match_span[0]-1] in (string.punctuation + SPECIAL) and not _starts_with(text,match_span[0],"@"))
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
    term_matches = pattern_match(post["text"].replace("\n"," "),
                                 KEYWORD_REGEX,
                                 include_mentions=include_mentions)
    if len(term_matches) == 0:
        return None
    ## Filter Out Substrings (Or Early Return)
    term_matches = filter_substrings(term_matches)
    ## Format
    post_match_cache = {"matches":term_matches}
    meta_cols = ["user_id","tweet_id","date","text"]
    for val in meta_cols:
        post_match_cache[val] = post[val]
    return post_match_cache

def get_posts(filename,
              keywords,
              indorgs=None,
              genders=None,
              locations=None):
    """

    """
    data = load_file(filename, indorgs, genders, locations)
    data = list(filter(lambda i: i.get("keywords"), data))
    all_matches = []
    for k in keywords:
        k_data = list(filter(lambda i: k in set(i.get("keywords")), data))
        k_data = [{"keyword":k, "tweet_id":d.get("tweet_id"),"user_id":d.get("user_id"),"date":d.get("date"),"text":d.get("text")} for d in k_data]
        all_matches.extend(k_data)
    return all_matches

def find_keyword_examples(filenames,
                          keywords,
                          n=10,
                          indorgs=None,
                          genders=None,
                          locations=None):
    """

    """
    ## Find Matches
    mp = Pool(NUM_JOBS)
    helper = partial(get_posts, keywords=keywords, indorgs=indorgs, genders=genders, locations=locations)
    mp_matches = list(tqdm(mp.imap_unordered(helper, filenames),
                        total=len(filenames),
                        leave=False,
                        file=sys.stdout))
    mp.close()
    ## Sample
    mp_matches = pd.DataFrame(flatten(mp_matches))
    sample = []
    for keyword in keywords:
        mp_keyword_matches = mp_matches.loc[mp_matches["keyword"]==keyword]
        mp_keyword_matches = mp_keyword_matches.drop_duplicates("text")
        mp_keyword_matches = mp_keyword_matches.sample(min(n, len(mp_keyword_matches)),
                                                       random_state=42,
                                                       replace=False)
        sample.append(mp_keyword_matches)
    sample = pd.concat(sample).reset_index(drop=True)
    return sample

def get_context(filename,
                ngrams=(1,1),
                window=None,
                include_mentions=False,
                min_date=None,
                max_date=None,
                indorgs=None,
                genders=None,
                locations=None):
    """

    """
    ## Load File Data
    data = load_file(filename, indorgs, genders, locations)
    ## Filter Out Non-Matches
    data = list(filter(lambda x: x.get("keywords") is not None, data))
    ## Date Filter
    dates = list(map(lambda i: parse(i.get("date")).date(), data))
    dates_accept = list(range(len(dates)))
    if min_date is not None:
        dates_accept = list(filter(lambda x: dates[x] >= min_date, dates_accept))
    if max_date is not None:
        dates_accept = list(filter(lambda x: dates[x] <= max_date, dates_accept))
    data = [data[i] for i in dates_accept]
    ## Run Matcher
    matches = list(map(lambda p: match_post(p, include_mentions),data))
    ## Get Context
    context = dict()
    term_counts = Counter()
    for m in matches:
        if m is None:
            continue
        text = m.get("text")
        for keyword, keyword_raw, keyword_span in m.get("matches"):
            ## Add Keyword to Context
            if keyword not in context:
                context[keyword] = Counter()
            term_counts[keyword] += 1
            ## Tokenize
            left_window = TOKENIZER.tokenize(text[:keyword_span[0]])
            right_window = TOKENIZER.tokenize(text[keyword_span[1]:])
            ## Window
            if window is not None:
                left_window = left_window[max(0,len(left_window)-window):]
                right_window = right_window[:window]
            ## N-Grams
            left_window = get_ngrams(left_window, ngrams[0], ngrams[1])
            right_window = get_ngrams(right_window, ngrams[0], ngrams[1])
            ## Cache
            window_counts = Counter(left_window + right_window)
            context[keyword] += window_counts
    return context, term_counts

def get_vocab_usage(filename,
                    vocab,
                    ngrams=(1,1),
                    min_date=None,
                    max_date=None,
                    indorgs=None,
                    genders=None,
                    locations=None):
    """

    """
    ## Load File Data
    data = load_file(filename, indorgs, genders, locations)
    ## Date Filter
    dates = list(map(lambda i: parse(i.get("date")).date(), data))
    dates_accept = list(range(len(dates)))
    if min_date is not None:
        dates_accept = list(filter(lambda x: dates[x] >= min_date, dates_accept))
    if max_date is not None:
        dates_accept = list(filter(lambda x: dates[x] <= max_date, dates_accept))
    data = [data[i] for i in dates_accept]
    ## Acceptance
    vocab = set(vocab)
    ## Initialize Counter
    vocab_counts = Counter()
    for p in data:
        ptoks = TOKENIZER.tokenize(p["text"])
        pngrams = get_ngrams(ptoks, ngrams[0], ngrams[1])
        pngrams = [i for i in pngrams if i in vocab]
        vocab_counts += Counter(pngrams)
    return vocab_counts

def replace_emojis(features):
    """

    """
    features_clean = []
    for f in features:
        f_res = demoji.findall(f)
        if len(f_res) > 0:
            for x,y in f_res.items():
                f = f.replace(x,f"<{y}>")
            features_clean.append(f)
        else:
            features_clean.append(f)
    return features_clean

#######################
### Keyword Identification/Counts
#######################

LOGGER.info("Counting Keyword Matches")

## Get Processed Files
filenames = sorted(glob(f"{DATA_DIR}*_minimal.json.gz"))

## Count Cache Files
keyword_count_cache = f"{CACHE_DIR}keyword_counts.csv"
post_count_cache = f"{CACHE_DIR}post_counts.csv"

## Load Counts
if (not os.path.exists(keyword_count_cache) and not os.path.exists(post_count_cache)) or RERUN:
    ## Get Keyword Counts Over Time (Parallel Processing)
    mp = Pool(NUM_JOBS)
    counter_func = partial(count_keywords_in_file,
                           indorgs=INDORGS,
                           genders=GENDERS,
                           locations=LOCATIONS)
    mp_results = list(tqdm(mp.imap_unordered(counter_func, filenames),
                        total=len(filenames),
                        file=sys.stdout))
    mp.close()
    ## Parse Keyword Counts
    n_posts = Counter()
    keyword_counts = dict()
    for n, kc in mp_results:
        n_posts += n
        for date, date_counts in kc.items():
            if date not in keyword_counts:
                keyword_counts[date] = date_counts
            else:
                keyword_counts[date] += date_counts
    ## Format Counts
    n_posts = pd.Series(n_posts).to_frame("count")
    keyword_counts = pd.DataFrame(keyword_counts).T.sort_index()
    keyword_counts = keyword_counts.fillna(0).astype(int)
    ## Cache
    n_posts.to_csv(post_count_cache)
    keyword_counts.to_csv(keyword_count_cache)
else:
    ## Load From Cache
    n_posts = pd.read_csv(post_count_cache, index_col=0)
    keyword_counts = pd.read_csv(keyword_count_cache, index_col=0)
    ## Format Indices
    n_posts.index = pd.to_datetime(n_posts.index)
    keyword_counts.index = pd.to_datetime(keyword_counts.index)

## Sort
n_posts = n_posts.sort_index()
keyword_counts = keyword_counts.sort_index()

## Relative Matches
keyword_counts_normed = keyword_counts.apply(lambda row: row / n_posts["count"], axis=0)

## Rolling Statistics
window_size = SMOOTHING_WINDOW
rolling_keyword_counts = keyword_counts.rolling(window_size,axis=0).sum()
rolling_n_posts = n_posts["count"].rolling(window_size).sum()
rolling_keyword_counts_normed = rolling_keyword_counts.apply(lambda x: x / rolling_n_posts, axis=0).iloc[window_size:]

## Filtering Helper
date_filter = lambda df, dts, window: df.loc[(dts >= pd.to_datetime(window[0]))&(dts<pd.to_datetime(window[1]))]

## Counts
overall_num_matches = keyword_counts.sum(axis=0)
overall_posts = n_posts["count"].sum()
pre_covid_num_matches = date_filter(keyword_counts, keyword_counts.index, PRE_COVID_WINDOW).sum(axis=0)
post_covid_num_matches = date_filter(keyword_counts, keyword_counts.index, POST_COVID_WINDOW).sum(axis=0)
pre_covid_posts = date_filter(n_posts, n_posts.index, PRE_COVID_WINDOW)["count"].sum()
post_covid_posts = date_filter(n_posts, n_posts.index, POST_COVID_WINDOW)["count"].sum()

## Window Rates
overall_match_rate = overall_num_matches / overall_posts
pre_covid_match_rate = pre_covid_num_matches / pre_covid_posts
post_covid_match_rate = post_covid_num_matches / post_covid_posts
match_rate_covid_change = (post_covid_match_rate - pre_covid_match_rate)
match_rate_covid_relative_change = (post_covid_match_rate - pre_covid_match_rate) / pre_covid_match_rate * 100
match_rate_covid_relative_change[(pre_covid_num_matches == 0)|(post_covid_num_matches==0)] = np.nan

## Variability over Time
max_single_day_change = keyword_counts.diff(axis=0).max(axis=0)
max_single_day_rate_change = keyword_counts_normed.diff(axis=0).max(axis=0)
match_rate_coefficient_of_variation = keyword_counts_normed.std(axis=0) / keyword_counts_normed.mean(axis=0)

#######################
### Demographic Breakdown
#######################

## Keyword Breakdown by Group
LOGGER.info("Counting Keywords by Group")
keyword_breakdown, timestamp_breakdown = count_fields(filenames,
                                                fields=["keywords"],
                                                indorgs=INDORGS,
                                                genders=GENDERS,
                                                locations={"country":[]},
                                                keys=["date","demographics","location"])

## Format Breakdown
keyword_breakdown = keyword_breakdown.loc["keywords"].reset_index()
keyword_breakdown["date"] = pd.to_datetime(keyword_breakdown["date"])
timestamp_breakdown["date"] = pd.to_datetime(timestamp_breakdown["date"])

## Aggregate by Gender (US Only)
gender_keyword_breakdown = pd.pivot_table(keyword_breakdown.loc[keyword_breakdown["country"]=="United States"],
                                          index="date",
                                          columns="gender",
                                          values=keyword_counts.columns,
                                          aggfunc=np.nansum)
gender_timestamp_breakdown = pd.pivot_table(timestamp_breakdown.loc[timestamp_breakdown["country"]=="United States"],
                                            index="date",
                                            columns="gender",
                                            values="count",
                                            aggfunc=np.nansum)

## Overall Aggregation
num_matches_gender = pd.pivot_table(gender_keyword_breakdown.sum(axis=0).reset_index(), index="level_0", columns="gender", values=0)
posts_gender = gender_timestamp_breakdown.sum(axis=0)
match_rate_gender = num_matches_gender / posts_gender

## Aggregation by COVID Period
pre_covid_num_matches_gender = date_filter(gender_keyword_breakdown, gender_keyword_breakdown.index, PRE_COVID_WINDOW).sum(axis=0)
pre_covid_num_matches_gender = pd.pivot_table(pre_covid_num_matches_gender.reset_index(), index="level_0", columns="gender", values=0)
post_covid_num_matches_gender = date_filter(gender_keyword_breakdown, gender_keyword_breakdown.index, POST_COVID_WINDOW).sum(axis=0)
post_covid_num_matches_gender = pd.pivot_table(post_covid_num_matches_gender.reset_index(), index="level_0", columns="gender", values=0)
pre_covid_posts_gender = date_filter(gender_timestamp_breakdown, gender_timestamp_breakdown.index, PRE_COVID_WINDOW).sum(axis=0)
post_covid_posts_gender = date_filter(gender_timestamp_breakdown, gender_timestamp_breakdown.index, POST_COVID_WINDOW).sum(axis=0)
pre_covid_match_rate_gender = pre_covid_num_matches_gender / pre_covid_posts_gender
post_covid_match_rate_gender = post_covid_num_matches_gender / post_covid_posts_gender

## Gender Skew (Male)
match_rate_gender_skew = match_rate_gender["man"] - match_rate_gender["woman"]
pre_covid_match_rate_gender_skew = pre_covid_match_rate_gender["man"] - pre_covid_match_rate_gender["woman"]
post_covid_match_rate_gender_skew = post_covid_match_rate_gender["man"] - post_covid_match_rate_gender["woman"]

## Change
match_rate_change_gender = post_covid_match_rate_gender - pre_covid_match_rate_gender
match_rate_percent_change_gender = (post_covid_match_rate_gender - pre_covid_match_rate_gender) / pre_covid_match_rate_gender * 100
for col in match_rate_percent_change_gender.columns:
    match_rate_percent_change_gender[col][(pre_covid_num_matches_gender[col] == 0)|(pre_covid_num_matches_gender[col]==0)] = np.nan

#######################
### Representative Posts
#######################

LOGGER.info("Identifying Representative Examples")

## Post Cache
rep_cache_file = f"{CACHE_DIR}representative_examples.json"

## Load Representatives
if not os.path.exists(rep_cache_file) or RERUN:
    ## Find Representative Posts
    representative_examples = []
    keyword_chunks = list(chunks(keyword_counts.columns.tolist(), 40))
    for keyword_chunk in tqdm(keyword_chunks,position=0,desc="Keyword Chunk",file=sys.stdout):
        keyword_examples = find_keyword_examples(filenames,
                                                 keyword_chunk,
                                                 n=100,
                                                 indorgs=INDORGS,
                                                 genders=GENDERS,
                                                 locations=LOCATIONS)
        representative_examples.append(keyword_examples)
    representative_examples = pd.concat(representative_examples).reset_index(drop=True)
    ## Cache
    with open(rep_cache_file, "w") as the_file:
        for _, row in representative_examples.iterrows():
            the_file.write(row.to_json()+"\n")
else:
    ## Load 
    representative_examples = []
    with open(rep_cache_file, "r") as the_file:
        for line in the_file:
            representative_examples.append(json.loads(line))
    representative_examples = pd.DataFrame(representative_examples)

#######################
### Keyword Comorbidity
#######################

LOGGER.info("Constructing Keyword Comorbidity Matrix")

## Comorbidity Cache
comorbidity_cache = f"{CACHE_DIR}keyword_comorbidity.csv"

## Load Comorbidity Matrix
if not os.path.exists(comorbidity_cache) or RERUN:
    ## Get Keyword Comorbidity
    mp = Pool(NUM_JOBS)
    chelper = partial(keyword_comorbidity,
                      comparison_set=set(falconet_keywords["Coronavirus"]),
                      indorgs=INDORGS,
                      genders=GENDERS,
                      locations=LOCATIONS)
    mp_results = list(tqdm(mp.imap_unordered(chelper, filenames),
                        total=len(filenames),
                        file=sys.stdout))
    mp.close()
    ## Format into Matrix
    comorbidity = np.zeros((keyword_counts.shape[1], 2))
    keyword2ind = dict(zip(keyword_counts.columns.tolist(), range(keyword_counts.shape[1])))
    for kc, kc_co in mp_results:
        for _k, _c in kc.items():
            comorbidity[keyword2ind[_k], 1] += _c
        for _k, _c in kc_co.items():
            comorbidity[keyword2ind[_k], 0] += _c
    comorbidity = comorbidity.astype(int)
    ## Format into DataFrame
    comorbidity = pd.DataFrame(comorbidity, columns=["coronavirus","total"], index=keyword_counts.columns)
    comorbidity["coronavirus_match_rate"] = comorbidity["coronavirus"] / comorbidity["total"]
    comorbidity = comorbidity.drop([i for i in falconet_keywords["Coronavirus"] if i in comorbidity.index])
    ## Cache
    comorbidity.to_csv(comorbidity_cache)    
else:
    ## Load from Cache
    comorbidity = pd.read_csv(comorbidity_cache,index_col=0)

#######################
### Keyword Context (Neighbors)
#######################

LOGGER.info("Calculating Contextual Keyword Usage")

## Initialize Regex for Keywords (To Get Spans)
KEYWORD_REGEX = create_regex_dict(keyword_counts.columns.tolist(), include_hashtag=INCLUDE_HASHTAGS)

## Cache File
context_cache_file = "{}context_{}_{}-{}_{}_{}.joblib".format(CACHE_DIR,
                                                        CONTEXT_WINDOW,
                                                        NGRAMS[0],
                                                        NGRAMS[1],
                                                        VOCAB_CUTOFF,
                                                        VOCAB_SIZE)
## Load Context
if os.path.exists(context_cache_file) and not RERUN:
    LOGGER.info("Loading {}".format(os.path.basename(context_cache_file)))
    context = joblib.load(context_cache_file)
else:
    ## Use Multiprocessing to Get Context
    mp = Pool(NUM_JOBS)
    con_helper = partial(get_context,
                         ngrams=NGRAMS,
                         window=CONTEXT_WINDOW,
                         min_date=None,
                         max_date=None,
                         indorgs=INDORGS,
                         genders=GENDERS,
                         locations=LOCATIONS)
    win_context = list(tqdm(mp.imap_unordered(con_helper, filenames),
                            desc="Context Calculator",
                            file=sys.stdout,
                            total=len(filenames)))
    mp.close()
    ## Filter Out Nulls
    win_context = list(filter(lambda i: i, win_context))
    ## Concatenate Results
    win_context_concat = dict()
    win_keyword_counts = Counter()
    for wc, kc in win_context:
        win_keyword_counts += kc
        for term, term_context in wc.items():
            if term not in win_context_concat:
                win_context_concat[term] = Counter()
            win_context_concat[term] += term_context
    ## Get Vocab Within Window
    context_vocab = pd.Series(flatten(win_context_concat.values())).value_counts()
    context_vocab = context_vocab.loc[context_vocab > VOCAB_CUTOFF].nlargest(VOCAB_SIZE)
    context_vocab = context_vocab.index.tolist()
    ## Use Multiprocessing to Get General Vocab Usage
    mp = Pool(NUM_JOBS)
    voc_helper = partial(get_vocab_usage,
                        vocab=context_vocab,
                        ngrams=NGRAMS,
                        min_date=None,
                        max_date=None,
                        indorgs=INDORGS,
                        genders=GENDERS,
                        locations=LOCATIONS)
    win_vocab = list(tqdm(mp.imap_unordered(voc_helper, filenames),
                        desc="COVID 19 Vocab Counter",
                        file=sys.stdout,
                        total=len(filenames)))
    mp.close()
    ## Concatenate Counts
    win_vocab = list(filter(lambda i: sum(i.values()) > 0, win_vocab))
    win_vocab_counts = sum(win_vocab, Counter())
    ## Cache Locally
    context = {
                            "context_counts":win_context_concat,
                            "keyword_counts":win_keyword_counts,
                            "vocab_counts":win_vocab_counts
                            }
    ## Cache on Disk
    _ = joblib.dump(context, context_cache_file)

## Get Relative Keyword Frequencies
p_keyword = pd.Series(context.get("keyword_counts"))
p_keyword = (p_keyword + SMOOTHING) / (p_keyword + SMOOTHING).sum()

## Get Relative N-Gram Frequencies
p_ngram = pd.DataFrame(context.get("vocab_counts").most_common(), columns=["ngram","freq"])
p_ngram["nlen"] = p_ngram["ngram"].map(len)
p_ngram["p_x"] = np.nan
for n in range(min(NGRAMS), max(NGRAMS)+1):
    p_ngram.loc[p_ngram["nlen"]==n,"p_x"] = (p_ngram.loc[p_ngram["nlen"]==n,"freq"] + SMOOTHING) / \
                                            (p_ngram.loc[p_ngram["nlen"]==n,"freq"] + SMOOTHING).sum()

## Concatenate Frequencies/Probabilities
context_df = []
for keyword, keyword_context in context.get("context_counts").items():
    keyword_context_df = pd.DataFrame(keyword_context.most_common(), columns=["ngram","context_freq"])
    keyword_context_df["freq"] = keyword_context_df["ngram"].map(lambda i: context.get("vocab_counts").get(i, None))
    keyword_context_df["keyword"] = keyword
    keyword_context_df["p_keyword"] = p_keyword[keyword]
    keyword_context_df["nlen"] = keyword_context_df["ngram"].map(len)
    keyword_context_df["p_x_keyword"] = np.nan
    for n in range(min(NGRAMS), max(NGRAMS)+1):
        keyword_context_df.loc[keyword_context_df["nlen"]==n,"p_x_keyword"] = \
                                        (keyword_context_df.loc[keyword_context_df["nlen"]==n,"context_freq"] + SMOOTHING) / \
                                        (keyword_context_df.loc[keyword_context_df["nlen"]==n,"context_freq"] + SMOOTHING).sum()
    context_df.append(keyword_context_df)
context_df = pd.concat(context_df).reset_index(drop=True)
context_df = context_df.dropna()

## Merge General Probabilities
context_df = pd.merge(context_df, p_ngram[["ngram","p_x"]])

## Compute PMI
context_df["pmi"] = np.log(context_df["p_x_keyword"] / context_df["p_x"])

## Formatting
pmi_df_pivot = pd.pivot_table(context_df.loc[(context_df["context_freq"]>=MIN_CONTEXT_FREQ) & 
                                             (context_df["freq"]>=MIN_FREQ)],
                                columns="keyword",
                                index="ngram",
                                values="pmi")
neighbors = []
for keyword in pmi_df_pivot.columns:
    keyword_neighbors = pmi_df_pivot[keyword].dropna().sort_values(ascending=False)
    keyword_neighbor_cache = {"keyword":keyword}
    for n in range(min(NGRAMS), max(NGRAMS)+1):
        keyword_neighbor_cache[f"neighbors_n_{n}"] = keyword_neighbors.loc[keyword_neighbors.index.map(len)==n].index.tolist()[:100]
    neighbors.append(keyword_neighbor_cache)
neighbors = pd.DataFrame(neighbors)

## Format Comparison
neighbor_cols = [f"neighbors_n_{n}" for n in range(min(NGRAMS), max(NGRAMS)+1)]
for nc in neighbor_cols:
    neighbors[nc] = neighbors[nc].map(lambda i: ", ".join("_".join(k) for k in i[:25]) if isinstance(i, list) else None)
neighbors = neighbors.set_index("keyword")

## Filter by Keyword Frequency
neighbors = neighbors.loc[[i for i in overall_num_matches.loc[overall_num_matches > 100].index if i in neighbors.index]]

#######################
### Combine Statistics
#######################

LOGGER.info("Concatenating Metrics")

## Combine Match Rates
summary = pd.concat([
    overall_num_matches.to_frame("num_matches"),
    overall_match_rate.to_frame("match_rate"),
    pre_covid_num_matches.to_frame("pre_covid_num_matches"),
    post_covid_num_matches.to_frame("post_covid_num_matches"),
    pre_covid_match_rate.to_frame("pre_covid_match_rate"),
    post_covid_match_rate.to_frame("post_covid_match_rate"),
    comorbidity[["coronavirus_match_rate"]],
    match_rate_covid_change.to_frame("post_covid_match_rate_change"),
    match_rate_covid_relative_change.to_frame("post_covid_match_rate_percent_change"),
    max_single_day_change.to_frame("max_single_day_change"),
    max_single_day_rate_change.to_frame("max_single_day_rate_change"),
    match_rate_coefficient_of_variation.to_frame("match_rate_coefficient_of_variation"),
    num_matches_gender.rename(columns={"man":"male_num_matches","woman":"female_num_matches"}),
    match_rate_gender.rename(columns={"man":"male_match_rate","woman":"female_match_rate"}),
    pre_covid_num_matches_gender.rename(columns={"man":"male_pre_covid_num_matches","woman":"female_pre_covid_num_matches"}),
    post_covid_num_matches_gender.rename(columns={"man":"male_post_covid_num_matches","woman":"female_post_covid_num_matches"}),
    pre_covid_match_rate_gender.rename(columns={"man":"male_pre_covid_match_rate","woman":"female_pre_covid_match_rate"}),
    post_covid_match_rate_gender.rename(columns={"man":"male_post_covid_match_rate","woman":"female_post_covid_match_rate"}),
    match_rate_gender_skew.to_frame("male_skew"),
    pre_covid_match_rate_gender_skew.to_frame("pre_covid_male_skew"),
    post_covid_match_rate_gender_skew.to_frame("post_covid_male_skew"),
    neighbors],
    axis=1)

## Format
summary = summary.sort_values("num_matches",ascending=False)

## Cache
summary.to_csv(f"{CACHE_DIR}summary.csv")

#######################
### Visualize
#######################

LOGGER.info("Creating Summary Visuals")

def visualize_timeseries(keyword,
                         window=SMOOTHING_WINDOW,
                         alpha=0.05):
    """

    """
    ## Get Timeseries (Smoothed and Confidence Intervals)
    posts = n_posts["count"].rolling(window).sum()
    timeseries = keyword_counts[keyword].rolling(window).sum()
    timeseries = timeseries.reindex(posts.index)
    ci_median = timeseries / posts
    ci_lower, ci_upper = proportion_confint(timeseries, posts, alpha=alpha, method="normal")
    ## Get Raw Rate
    raw_rate = keyword_counts.reindex(posts.index)[keyword] / n_posts["count"]
    ## Plot
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.scatter(raw_rate.index,
               raw_rate.values,
               color="C0",
               alpha=0.2,
               zorder=-1)
    ax.fill_between(ci_lower.index, ci_lower, ci_upper, color="C0",alpha=0.5, label="{}-day Average".format(window), zorder=1)
    ax.plot(ci_median.index, ci_median.values, color="C0", alpha=0.8, zorder=2)
    ax.axvline(pd.to_datetime(POST_COVID_WINDOW[0]),
                  color="black",
                  linestyle="--",
                  alpha=0.8,
                  label="COVID-19 Lockdown")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_ylabel("Proportion\nof Tweets", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    for t in ax.get_xticklabels():
        t.set_horizontalalignment("right")
    ax.set_ylim(bottom=0)
    ax.set_xlim(ci_lower.dropna().index.min(), ci_lower.dropna().index.max())
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    term_lists = falconet_keywords_reverse.get(keyword, None)
    if term_lists:
        ax.set_title("Keyword: {} [{}]".format(keyword, ", ".join(term_lists)), fontweight="bold")
    else:
        ax.set_title(f"Keyword: {keyword}", fontweight="bold")
    fig.tight_layout()
    return fig, ax    

def visualize_pmi(keyword,
                  min_context_freq=MIN_CONTEXT_FREQ,
                  min_freq=MIN_FREQ,
                  k_top=40):
    """

    """
    ## Get PMI
    keyword_pmi = context_df.loc[context_df["keyword"]==keyword]
    keyword_pmi = keyword_pmi.loc[keyword_pmi["context_freq"]>=min_context_freq]
    keyword_pmi = keyword_pmi.loc[keyword_pmi["freq"]>=min_freq]
    keyword_pmi = keyword_pmi.set_index("ngram")["pmi"].copy()
    keyword_pmi = keyword_pmi.nlargest(k_top)
    keyword_pmi.index = keyword_pmi.index.map(lambda i: " ".join(i))
    keyword_pmi.index = replace_emojis(keyword_pmi.index)
    ## Generate Figure
    fig, ax = plt.subplots(figsize=(10,5.8))
    keyword_pmi.iloc[::-1].plot.barh(ax=ax, color="C0", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_ylabel("N-Gram", fontweight="bold")
    ax.set_xlabel("Context Strength", fontweight="bold")
    term_lists = falconet_keywords_reverse.get(keyword, None)
    if term_lists:
        ax.set_title("Keyword: {} [{}]".format(keyword, ", ".join(term_lists)), fontweight="bold")
    else:
        ax.set_title(f"Keyword: {keyword}", fontweight="bold")
    fig.tight_layout()
    return fig, ax

## Choose Keywords to Plot
to_plot = summary.dropna(subset=neighbor_cols).index.tolist()

## Generate Figures
errors = []
for keyword in tqdm(to_plot):
    ## Clean Keyword for Filename
    keyword_clean = keyword.replace(" ","_").replace("/","-").replace(":","-")
    ## Cache Sample Text
    text = []
    for _, row in representative_examples.loc[representative_examples["keyword"]==keyword].iterrows():
        text.append(row["text"])
    keyword_clean = keyword.replace("/","-").replace(":","-")
    with open(f"{CACHE_DIR}examples/{keyword_clean}.txt","w") as the_file:
        for t in text:
            the_file.write(t.replace("\n","") + "\n\n")
    ## Generate Plots
    try:
        ## Timeseries
        fig, ax = visualize_timeseries(keyword,
                                       window=SMOOTHING_WINDOW)
        fig.savefig(f"{PLOT_DIR}timeseries/{keyword_clean}.png", dpi=300)
        plt.close(fig)
        ## PMI
        fig, ax = visualize_pmi(keyword,
                                min_context_freq=MIN_CONTEXT_FREQ,
                                min_freq=MIN_FREQ,
                                k_top=40)
        fig.savefig(f"{PLOT_DIR}context/{keyword_clean}.png", dpi=300)
        plt.close(fig)
    except IndexError:
        errors.append(keyword)
        plt.close()
        continue
    except KeyboardInterrupt:
        break
