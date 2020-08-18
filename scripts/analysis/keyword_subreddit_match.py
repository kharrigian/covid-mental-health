
###################
### Configuration
###################

## Rerun (Re-run existing matches, vocabulary generation)
RERUN = False

## Processed Data Directory
DATA_DIR = "./data/processed/reddit/2017-2020/histories/"
# DATA_DIR = "./data/processed/twitter/2018-2020/timelines/"

## Plot Directory
PLOT_DIR = "./plots/reddit/2017-2020/keywords-subreddits/"
# PLOT_DIR = "./plots/twitter/2018-2020/keywords/"

## Cache Directory
CACHE_DIR = "./data/results/reddit/2017-2020/keywords-subreddits/"
# CACHE_DIR = "./data/results/twitter/2018-2020/keywords/"

## Random Sampling
SAMPLE_RATE = 1
SAMPLE_SEED = 42

## Platform
PLATFORM = "reddit"
# PLATFORM = "twitter"

## Language Date Boundaries
START_DATE = "2017-01-01"
END_DATE = "2020-05-01"
# START_DATE = "2018-01-01"
# END_DATE = "2020-06-20"

## Analysis Date Boundaries
ANALYSIS_START = "2019-01-01"
COVID_START = "2020-03-01"

## Multiprocessing
NUM_JOBS = 8

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
import joblib
import demoji
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from scipy.sparse import csr_matrix, vstack
from pandas.plotting import register_matplotlib_converters

## Mental Health Library
from mhlib.util.logging import initialize_logger
from mhlib.util.helpers import flatten
from mhlib.model.vocab import Vocabulary
from mhlib.model.file_vectorizer import File2Vec
from mhlib.preprocess.preprocess import tokenizer

###################
### Globals
###################

## Logging
LOGGER = initialize_logger()

## Register Matplotlib Time Converters
_ = register_matplotlib_converters()

## Plot Directory
if not os.path.exists(PLOT_DIR):
    _ = os.makedirs(PLOT_DIR)
if not os.path.exists(f"{PLOT_DIR}timeseries/"):
    _ = os.makedirs(f"{PLOT_DIR}timeseries/")
if not os.path.exists(f"{PLOT_DIR}context/"):
    _ = os.makedirs(f"{PLOT_DIR}context/")

## Cache Directory
if not os.path.exists(CACHE_DIR):
    _ = os.makedirs(CACHE_DIR)

## Date Resolution (Changing May Break Expected Visualizations/Analysis)
DATE_RES = "day"

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

def bootstrap_sample(X,
                     Y=None,
                     func=np.mean,
                     axis=0,
                     sample_percent=70,
                     samples=100):
    """

    """
    sample_size = int(sample_percent / 100 * X.shape[0])
    estimates = []
    for sample in range(samples):
        sample_ind = np.random.choice(X.shape[0], size=sample_size, replace=True)
        X_sample = X[sample_ind]
        if Y is not None:
            Y_sample = Y[sample_ind]
            sample_est = func(X_sample, Y_sample)
        else:
            sample_est = func(X_sample, axis=axis)
        estimates.append(sample_est)
    estimates = np.vstack(estimates)
    ci = np.percentile(estimates, [2.5, 50, 97.5], axis=axis)
    return ci

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

###################
### Functions
###################

def create_regex_dict(terms):
    """

    """
    regex_dict = {}
    for t in terms:
        is_stem = t.endswith("*")
        t_stem = re.escape(t.rstrip("*"))
        if t_stem.isupper():
            regex_dict[t] = (re.compile(t_stem), is_stem)
        else:
            regex_dict[t] = (re.compile(t_stem, re.IGNORECASE), is_stem)
    return regex_dict

def _search_full_words(text,
                       regex,
                       stem=False):
    """

    """
    matches = []
    L = len(text)
    for match in regex.finditer(text):
        match_span = match.span()
        starts_text = match_span[0] == 0 or text[match_span[0]-1] == " " or text[match_span[0]-1] in string.punctuation
        ends_text = match_span[1] == L or text[match_span[1]] == " " or text[match_span[1]] in string.punctuation
        if starts_text:
            if stem or ends_text:
                matches.append((text[match_span[0]:match_span[1]], match_span))
    return matches

def pattern_match(text,
                  pattern_re):
    """

    """
    matches = []
    for keyword, (pattern, is_stem) in pattern_re.items():
        keyword_matches = _search_full_words(text, pattern, stem=is_stem)
        keyword_matches = [(keyword, k[0], k[1]) for k in keyword_matches]
        matches.extend(keyword_matches)
    return matches

def match_post(post):
    """

    """    
    ## Cycle Through Options
    match_found = False
    match_cache = {}
    for category in MATCH_DICT.keys():
        if PLATFORM == "reddit":
            if post["subreddit"].lower() in MATCH_DICT[category]["subreddits"]:
                if category not in match_cache:
                    match_cache[category] = {}
                match_cache[category]["subreddits"] = post["subreddit"].lower()
                match_found = True
        term_matches = pattern_match(post["text"], MATCH_DICT[category]["terms"])
        if len(term_matches) > 0:
            if category not in match_cache:
                match_cache[category] = {}
            match_cache[category]["terms"] = term_matches
            match_found = True
    ## Metadata
    if not match_found:
        return None
    post_match_cache = {"matches":match_cache}
    meta_cols = ["user_id_str","created_utc","text"]
    if PLATFORM == "reddit":
        meta_cols.extend(["comment_id","subreddit"])
    elif PLATFORM == "twitter":
        meta_cols.append("tweet_id")
    for val in meta_cols:
        post_match_cache[val] = post[val]
    return post_match_cache

def find_matches(filename,
                 level="day"):
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
        for post in json.load(the_file):
            n += 1
            if sampler.uniform(0,1) >= SAMPLE_RATE:
                continue
            else:
                n_seen += 1
                timestamps.append(datetime.fromtimestamp(post["created_utc"]))
                post_matches = match_post(post)
                if post_matches is not None:
                    matches.append(post_matches)
    ## Format Timestamps
    timestamps = format_timestamps(timestamps, level)
    timestamps = Counter(timestamps)
    return filename, matches, n, n_seen, timestamps

def search_files(filenames,
                 date_res="day"):
    """

    """

    ## Run Lookup
    mp = Pool(NUM_JOBS)
    helper = partial(find_matches, level=date_res)
    res = list(tqdm(mp.imap_unordered(helper, filenames), total=len(filenames), desc="Searching For Matches", file=sys.stdout))
    mp.close()
    ## Parse
    filenames = [r[0] for r in res]
    matches = [r[1] for r in res]
    n = [r[2] for r in res]
    n_seen = [r[3] for r in res]
    timestamps = [r[4] for r in res]
    return filenames, matches, n, n_seen, timestamps

def load_keyword_search_results(match_cache_file):
    """

    """
    match_data = joblib.load(match_cache_file)
    filenames = match_data.get("filenames")
    matches = match_data.get("matches")
    n = match_data.get("n")
    n_seen = match_data.get("n_seen")
    timestamps = match_data.get("timestamps")
    return filenames, matches, n, n_seen, timestamps

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
        m_res_present = set(map(lambda x: x[0], m_res))
        if query in m_res_present:
            relevant_matches.append(m)
    return relevant_matches

def get_match_values(post_matches,
                     category,
                     match_type,
                     date_res="day"):
    """

    """
    ## Look For Relevant Matches
    timestamps = []
    values = []
    for p in post_matches:
        if category in p["matches"] and match_type in p["matches"][category]:
            timestamps.append(p["created_utc"])
            values.append(p["matches"][category][match_type])
    ## Format
    timestamps = list(map(datetime.fromtimestamp, timestamps))
    timestamps = format_timestamps(timestamps, date_res)
    return timestamps, values

def vectorize_timestamps(timestamps,
                         date_range_map):
    """

    """
    tau = np.zeros(len(date_range_map))
    if not isinstance(timestamps, Counter):
        timestamps = Counter(timestamps)
    for t, v in timestamps.items():
        if t not in date_range_map:
            continue
        tau[date_range_map[t]] = v
    tau = csr_matrix(tau)
    return tau

def terms_over_time(term_matches,
                    term_map,
                    date_range_map,
                    subreddit=False):
    """

    """
    ## Initialize Matrix
    term_time_matrix = np.zeros((len(term_map), len(date_range_map)))
    ## Construct Matrix
    for f, (dates, tm) in enumerate(term_matches):
        for dt, t in zip(dates, tm):
            if dt not in date_range_map:
                continue
            dt_ind = date_range_map[dt]
            if not subreddit:
                for _t in t:
                    _t_ind = term_map[_t[0]]
                    term_time_matrix[_t_ind, dt_ind] += 1
            else:
                _t_ind = term_map[t]
                term_time_matrix[_t_ind, dt_ind] += 1
    ## Extract Row/Column Names
    date_range_map_rev = dict((y,x) for x, y in date_range_map.items())
    term_map_rev = dict((y,x) for x, y in term_map.items())
    rows = list(map(lambda i: term_map_rev[i], range(term_time_matrix.shape[0])))
    cols = list(map(lambda i: date_range_map_rev[i], range(term_time_matrix.shape[1])))
    cols = list(map(lambda i: datetime(*list(i)), cols))
    ## Format into DataFrame
    df = pd.DataFrame(term_time_matrix.T, index=cols, columns=rows)
    return df

def learn_vocabulary(filenames,
                     date_boundaries,
                     vocab_kwargs=dict(filter_negate=True,
                                       filter_upper=True,
                                       filter_punctuation=True,
                                       filter_numeric=True,
                                       filter_user_mentions=True,
                                       filter_url=True,
                                       filter_stopwords=False,
                                       keep_pronouns=True,
                                       preserve_case=False,
                                       emoji_handling=None,
                                       filter_hashtag=False,
                                       strip_hashtag=False,
                                       max_vocab_size=250000,
                                       min_token_freq=10,
                                       max_token_freq=None,
                                       ngrams=(1,1),
                                       keep_retweets=False,
                                       random_state=SAMPLE_SEED),
                    ):
    """
    Learn Vocabulary ~ Time

    Args:
        filenames (list of str)
        vocab_kwargs (dict)
        date_range_map (dict)

    """
    ## Learn Vocabulary
    vocabulary = Vocabulary(**vocab_kwargs)
    vocabulary = vocabulary.fit(filenames,
                                chunksize=50,
                                min_date=date_boundaries[0],
                                max_date=date_boundaries[-1],
                                randomized=True,
                                n_samples=None if SAMPLE_RATE == 1 else SAMPLE_RATE)
    ## Initialize Vectorizer
    f2v = File2Vec(); f2v.vocab = vocabulary; f2v._initialize_dict_vectorizer()
    return f2v

def load_vocabulary(vocab_cache_file):
    """

    """
    f2v = joblib.load(vocab_cache_file)
    return f2v

def vectorize_files(filenames,
                    date_boundaries,
                    f2v):
    """

    """
    ## Apply Vectorization
    X = np.zeros((len(date_boundaries)-1, len(f2v.vocab.ngram_to_idx)))
    for i, (i_start, i_end) in tqdm(enumerate(zip(date_boundaries[:-1],date_boundaries[1:])), desc="Vectorization", total=len(date_boundaries)-1, file=sys.stdout):
        ## Vectorize
        files_vec, vec = f2v._vectorize_files(filenames,
                                              jobs=4,
                                              min_date=i_start,
                                              max_date=i_end,
                                              randomized=True,
                                              n_samples=None if SAMPLE_RATE == 1 else SAMPLE_RATE)
        ## Cache Total Usage
        X[i] += vec.sum(axis=0)
    return X

def load_X(X_cache_file):
    """

    """
    X = joblib.load(X_cache_file)
    return X

def _get_context(text,
                 tspan,
                 f2v,
                 ngram_window):
    """

    """
    ## Get Context
    text_before = text[:tspan[0]]
    text_after = text[tspan[1]:]
    ## Tokenize
    tok_before = tokenizer.tokenize(text_before)
    tok_after = tokenizer.tokenize(text_after)
    ## Filter
    try:
        tok_before = f2v.vocab._loader.filter_user_data([{"text_tokenized":tok_before}])[0]["text_tokenized"]
    except IndexError:
        tok_before = []
    try:
        tok_after = f2v.vocab._loader.filter_user_data([{"text_tokenized":tok_after}])[0]["text_tokenized"]
    except IndexError:
        tok_after = []
    ## Get Window
    if ngram_window is not None:
        tok_before = tok_before[max(0, len(tok_before)-ngram_window):]
        tok_after = tok_after[:min(ngram_window, len(tok_after))]
    ## Get N-Grams
    tok_before = f2v.vocab.get_ngrams(tok_before, f2v.vocab._ngrams[0], f2v.vocab._ngrams[1])
    tok_after = f2v.vocab.get_ngrams(tok_after, f2v.vocab._ngrams[0], f2v.vocab._ngrams[1])
    ## Combine
    context = tok_before + tok_after
    return context

def construct_matched_ngram_matrix(matches,
                                   term_list,
                                   f2v,
                                   date_range_map,
                                   date_res,
                                   ngram_window=None):
    """
    Isolate N-grams around keyword matches

    Args:
        matches (list of dict)
        term_list (str)
        f2v (File2Vec)
        date_range_map
        date_res
        ngram_window (int): Size to left and right of matched token

    Returns:
        M,
        D,
        C,
        categories
    """
    ## Target Encoding
    categories=sorted(MATCH_DICT[term_list]["terms"])
    categories_map=dict(zip(categories, range(len(categories))))
    ## Initialize Cache
    M = []
    D = []
    C = []
    ## Cycle Through Matches
    for mlist in tqdm(matches, desc="Match Set", file=sys.stdout):
        ## Isolate Relevant Matches
        term_matches = list(filter(lambda i: term_list in i["matches"] and "terms" in i["matches"][term_list], mlist))
        if len(term_matches) == 0:
            continue
        ## Get N-Grams, Terms, and Dates
        ngrams, dates, terms = [], [], []
        for doc in term_matches:
            doc_terms = doc["matches"][term_list]["terms"]
            for tkey, tmatch, tspan in doc_terms:
                t_contexts = _get_context(doc["text"],
                                          tspan,
                                          f2v,
                                          ngram_window)

                ngrams.append(t_contexts)
                dates.append(doc["created_utc"])
                terms.append(tkey)
        ngrams = list(map(Counter, ngrams))
        dates = list(map(lambda d: _format_timestamp(datetime.fromtimestamp(d), date_res), dates))
        ## Vectorize
        mvec = f2v._count2vec.transform(ngrams)
        dvec = np.array(list(map(lambda i: date_range_map.get(i), dates)))
        cvec = np.zeros((mvec.shape[0], len(categories)))
        for i, t in enumerate(terms):
            cvec[i, categories_map[t]] += 1
        ## Drop Out-of-Time Window
        mask = np.nonzero(~pd.isnull(dvec))[0]
        M.append(mvec[mask])
        D.append(dvec[mask])
        C.append(csr_matrix(cvec[mask]))
    ## Stack Results
    M = vstack(M)
    D = np.hstack(D)
    C = vstack(C)
    return M, D, C, categories

def assign_dates_to_bins(dates,
                         date_boundaries):
    """
    Convert date tuples into bins within string boundaries

    Args:
        dates (list of tuple): e.g. [(2019,1,23),...]
        date_boundaries (list of str): Date boundaries e.g. ["2019-01-01","2020-01-01","2021-01-01"]
    
    Returns:
        date_bins (list of int): Index of appropriate date bin
    """
    date_boundaries = pd.to_datetime(date_boundaries)
    date_bins = []
    for d in dates:
        if d < date_boundaries[0] or d >= date_boundaries[-1]:
            date_bins.append(-1)
        else:
            for b, (dl, dr) in enumerate(zip(date_boundaries[:-1], date_boundaries[1:])):
                if d >= dl and d < dr:
                    date_bins.append(b)
                    break
    assert len(date_bins) == len(dates)
    return date_bins
        
def compute_pmi_matrix(X,
                       context_dict,
                       feature_names,
                       date_bin,
                       alpha=0.001,
                       min_support=30):
    """
    Compute a Keyword x Vocabulary PMI Matrix

    Args:
        X (2d-array): Date Boundary x Vocabulary
        context_dict (dict): Outputs from construct_matched_ngram_matrix
        feature_names (list): List of vocabulary terms
        date_bin (int): Which date bin to compute PMI within
        alpha (float): Smoothing parameter
        min_support (int): Maximimum number of matches to compute PMI
    
    Returns:
        pmi (pandas DataFrame): Keyword x Vocabulary PMI scores
        term_counts (pandas Series): Keyword match counts in date bin
    """
    ## Apply Date Filtering to X
    X_filt_sum = X[date_bin]
    ## Compute P(X)
    px = np.divide(X_filt_sum + alpha,
                  (X_filt_sum + alpha).sum())
    ## Apply Date filtering to Context Date
    context_date_mask = [i for i, d in enumerate(context_dict["dates"]) if d == date_bin]
    ngrams_filt = context_dict.get("ngrams")[context_date_mask]
    terms_filt = context_dict.get("terms")[context_date_mask]

    ## Initialize Matrix
    pmi = np.zeros((terms_filt.shape[1], X_filt_sum.shape[0]))
    ngram_counts = np.zeros((terms_filt.shape[1], X_filt_sum.shape[0]))
    term_counts = []
    for t, term in enumerate(terms_filt.T):
        ## Get Term Mask
        term_mask = np.nonzero(term)[1]
        term_counts.append(len(term_mask))
        if term_counts[-1] < min_support:
            continue 
        ## Isolate Relevant N-Grams
        ngram_filt_sum = np.array(ngrams_filt[term_mask].sum(axis=0))[0]
        ngram_counts[t] = ngram_filt_sum
        ## Compute Probability Estimate
        px_t = np.divide(ngram_filt_sum + alpha,
                        (ngram_filt_sum + alpha).sum())
        ## Compute PMI
        pmi[t] = np.log(px_t / px)
    ## Format Matrices
    pmi = pd.DataFrame(pmi, index=context_dict.get("categories"), columns=feature_names)
    ngram_counts = pd.DataFrame(ngram_counts, index=context_dict.get("categories"), columns=feature_names)
    ## Drop Rows Lacking Support
    pmi = pmi.loc[~(pmi==0).all(axis=1)].copy()
    ngram_counts = ngram_counts.loc[~(ngram_counts==0).all(axis=1)].copy()
    ## Format Counts
    term_counts = pd.Series(index=context_dict.get("categories"), data=term_counts)
    return pmi, term_counts, ngram_counts

def plot_context_change(query_term,
                        term_list,
                        pmi_dict,
                        min_tok_freq=5,
                        top_k=15):
    """

    """
    ## Get Desired List
    pmi_pre = pmi_dict.get(term_list).get(0).get("pmi")
    pmi_post = pmi_dict.get(term_list).get(1).get("pmi")
    ngram_counts_pre = pmi_dict.get(term_list).get(0).get("ngram_counts")
    ngram_counts_post = pmi_dict.get(term_list).get(1).get("ngram_counts")
    ## Get Relevant Data
    q_pre = (ngram_counts_pre.loc[query_term].loc[ngram_counts_pre.loc[query_term]>=min_tok_freq]).index
    q_pre_plot = pmi_pre.loc[query_term, q_pre].sort_values().copy()
    q_pre_plot.index = replace_emojis(q_pre_plot.index.map(lambda i: "_".join(i)))
    q_post = (ngram_counts_post.loc[query_term].loc[ngram_counts_post.loc[query_term]>=min_tok_freq]).index
    q_post_plot = pmi_post.loc[query_term, q_post].sort_values().copy()
    q_post_plot.index = replace_emojis(q_post_plot.index.map(lambda i: "_".join(i)))
    q_compare = pd.concat([q_pre_plot.to_frame("pre"), q_post_plot.to_frame("post")], axis=1).dropna()
    q_compare["delta"] = q_compare["post"] - q_compare["pre"]
    q_compare = q_compare.sort_values("delta")
    ## Initialize Figure + Axes
    f = plt.figure(constrained_layout=False,figsize=(8,6))
    gs = gridspec.GridSpec(2, 2, figure=f)
    top_pre = f.add_subplot(gs[0,0])
    top_post = f.add_subplot(gs[1,0])
    top_diff = f.add_subplot(gs[:,1])
    ## Plot Top
    q_pre_plot.nlargest(top_k).iloc[::-1].plot.barh(ax=top_pre, alpha=0.8)
    q_post_plot.nlargest(top_k).iloc[::-1].plot.barh(ax=top_post, alpha=0.8)
    ## Plot Diff
    q_diff_plot = (q_compare["delta"].nlargest(top_k).append(q_compare["delta"].nsmallest(top_k))).drop_duplicates().sort_values()
    q_diff_plot.plot.barh(ax=top_diff, alpha=0.8)
    top_diff.axvline(0, color="black", linestyle=":", alpha=0.5)
    top_diff.yaxis.tick_right()
    ## Format Plot
    top_pre.set_title("Pre COVID-19", fontweight="bold", fontstyle="italic", loc="left", fontsize=10)
    top_post.set_title("Post COVID-19", fontweight="bold", fontstyle="italic", loc="left", fontsize=10)
    top_post.set_xlabel("PMI", fontweight="bold")
    top_diff.set_title("Largest Changes", fontweight="bold", fontstyle="italic", loc="left", fontsize=10)
    top_diff.set_xlabel("PMI Delta", fontweight="bold")
    for ax in [top_pre, top_post, top_diff]:
        ax.tick_params(axis="y", labelsize=9)
    f.suptitle("Keyword: '{}'".format(query_term.title()), fontweight="bold", fontsize=12)
    f.tight_layout()
    f.subplots_adjust(hspace=0.2)
    return f

###################
### Terms + Subreddits
###################

## Load Georgetown Mental Health Resources (SMHD)
MH_TERM_FILE = "./data/resources/mental_health_terms.json"
MH_SUBREDDIT_FILE = "./data/resources/mental_health_subreddits.json"
with open(MH_TERM_FILE,"r") as the_file:
    MH_TERMS = json.load(the_file)
with open(MH_SUBREDDIT_FILE,"r") as the_file:
    MH_SUBREDDITS = json.load(the_file)

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
COVID_SUBREDDIT_FILE = "./data/resources/covid_subreddits.json"
with open(COVID_TERM_FILE,"r") as the_file:
    COVID_TERMS = json.load(the_file)
with open(COVID_SUBREDDIT_FILE,"r") as the_file:
    COVID_SUBREDDITS = json.load(the_file)

###################
### Identify Matches
###################

## Create Match Dictionary (Subreddit Lists + Term Regular Expressions)
MATCH_DICT = {
    "mental_health":{
        "terms":create_regex_dict(MH_TERMS["terms"]["smhd"]),
        "subreddits":set(MH_SUBREDDITS["all"]) if PLATFORM == "reddit" else set(),
        "name":"SMHD"
    },
    "crisis":{
        "terms":create_regex_dict(CRISIS_KEYWORDS),
        "subreddits":set(),
        "name":"JHU Crisis"
    },
    "mental_health_keywords":{
        "terms":create_regex_dict(MH_KEYWORDS),
        "subreddits":set(),
        "name":"JHU CLSP"
    },
    "covid":{
        "terms":create_regex_dict(COVID_TERMS["covid"]),
        "subreddits":set(COVID_SUBREDDITS["covid"]) if PLATFORM == "reddit" else set(),
        "name":"COVID-19"
    }
}

## Find or Load Matches
match_cache_file = f"{CACHE_DIR}{PLATFORM}_{START_DATE}_{END_DATE}_matches.joblib"
if not os.path.exists(match_cache_file) or RERUN:
    LOGGER.info("Starting Keyword Search")
    ## Find Procesed Files
    filenames = sorted(glob(f"{DATA_DIR}*.json.gz"))
    ## Search For Keyword/Subreddit Matches
    filenames, matches, n, n_seen, timestamps = search_files(filenames,
                                                             date_res=DATE_RES)
    ## Cache
    _ = joblib.dump({"filenames":filenames,
                     "matches":matches,
                     "n":n,
                     "n_seen":n_seen,
                     "timestamps":timestamps}, match_cache_file)
else:
    LOGGER.info("Loading Keyword Search Results")
    filenames, matches, n, n_seen, timestamps = load_keyword_search_results(match_cache_file)

## Isolate Matches by Category
LOGGER.info("Isolating Matches by List and Type")
match_values = {}
for match_key, match_key_vals in MATCH_DICT.items():
    match_values[match_key] = {}
    for match_type in match_key_vals.keys():
        if len(match_key_vals[match_type]) == 0:
            continue
        if match_type not in ["subreddits","terms"]:
            continue
        match_values[match_key][match_type] = list(map(lambda p: get_match_values(p,
                                                                                  match_key,
                                                                                  match_type,
                                                                                  DATE_RES),
                                                        matches))

###################
### Temporal Analysis
###################

LOGGER.info("Vectorizing Matches")

## Identify Date Range
date_range = sorted(set(format_timestamps(pd.date_range(START_DATE, END_DATE, freq="h"), DATE_RES)))
date_range_map = dict(zip(date_range, range(len(date_range))))

## Vectorize General Timestamps (Filenames x Dates)
tau = vstack([vectorize_timestamps(i, date_range_map) for i in timestamps]).toarray()

## Vectorize Term Match Timestamps (Filenames x Dates)
term_vectors = {}
for match_key, match_key_vals in match_values.items():
    term_vectors[match_key] = {}
    for match_type, match_vals in match_key_vals.items():
        term_vectors[match_key][match_type] = vstack([vectorize_timestamps(i[0], date_range_map) for i in match_vals]).toarray()

## Compute Term Match Proportions (Filenames x Dates)
term_vectors_normed = {}
for match_key, match_key_vals in term_vectors.items():
    term_vectors_normed[match_key] = {}
    for match_type, match_vals in match_key_vals.items():
        term_vectors_normed[match_key][match_type] = np.divide(match_vals,
                                                               tau,
                                                               out=np.zeros_like(match_vals),
                                                               where=tau>0)

## Term/Subreddit Maps (Value to Index Mapping)
term_maps = {}
for match_key, match_key_vals in MATCH_DICT.items():
    term_maps[match_key] = {}
    for match_type, match_vals in match_key_vals.items():
        if match_type not in ["subreddits","terms"]:
            continue
        term_maps[match_key][match_type] = dict((y, x) for x, y in enumerate(sorted(match_vals)))

## Vectorize Term Breakdowns (Filename x Term)
term_breakdowns = {}
for match_key, match_key_vals in match_values.items():
    term_breakdowns[match_key] = {}
    for match_type, match_vals in match_key_vals.items():
        if match_type == "subreddits":
            breakdown = vstack([vectorize_timestamps(i[1], term_maps[match_key][match_type]) for i in match_vals]).toarray()
        else:
            breakdown = vstack([vectorize_timestamps([j[0] for j in flatten(i[1])], term_maps[match_key][match_type]) for i in match_vals]).toarray()
        term_breakdowns[match_key][match_type] = breakdown

## Vectorize Terms Over Time (Time x Term)
term_time_df = {}
for match_key, match_key_vals in match_values.items():
    term_time_df[match_key] = {}
    for match_type, match_vals in match_key_vals.items():
        term_time_df[match_key][match_type] = terms_over_time(match_vals,
                                                  term_maps[match_key][match_type],
                                                  date_range_map,
                                                  match_type=="subreddits")

###################
### Temporal Visualization
###################

LOGGER.info("Beginning Visualization")

## Date Range (As Datetimes)
date_index = list(map(lambda i: datetime(*list(i)), date_range))

## Count Unique Sets
term_identifiers = []
for match_key, match_vals in MATCH_DICT.items():
    for match_type, vals in match_vals.items():
        if len(vals) == 0 or match_type not in ["terms","subreddits"]:
            continue
        term_identifiers.append((match_key, match_type))

## Plot Post Poportions over Time
LOGGER.info("Plotting Post Proportions Over Time")
fig, ax = plt.subplots(len(term_identifiers), 1, figsize=(10,5.8))
for p, (pkey, ptype) in enumerate(term_identifiers):
    pci = bootstrap_sample(term_vectors_normed[pkey][ptype],
                           func=np.mean,
                           sample_percent=70,
                           samples=100)
    ax[p].fill_between(date_index,
                       pci[0],
                       pci[2],
                       color="C0",
                       alpha=.5)
    ax[p].plot(date_index,
               pci[1],
               color="C0",
               alpha=0.8,
               linestyle="--",
               linewidth=2)
    pname = "{} {}".format(MATCH_DICT[pkey]["name"], "Terms" if ptype == "terms" else "Subreddits")
    ax[p].set_title(pname, loc="left", fontweight="bold")
    ax[p].set_ylabel("Proportion\nof Posts", fontweight="bold")
    ax[p].spines["top"].set_visible(False)
    ax[p].spines["right"].set_visible(False)
    ax[p].set_xlim(left=pd.to_datetime(ANALYSIS_START),right=pd.to_datetime(END_DATE))
ax[-1].set_xlabel("Date", fontweight="bold")
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}{PLATFORM}_term_subreddit_proportions.png", dpi=300)
plt.close()

## Plot User Proportions over Time
LOGGER.info("Plotting User Proportions Over Time")
fig, ax = plt.subplots(len(term_identifiers), 1, figsize=(10,5.8))
for p, (pkey, ptype) in enumerate(term_identifiers):
    pci = bootstrap_sample(term_vectors_normed[pkey][ptype],
                           tau,
                           func=lambda x, y: (x>0).sum(axis=0) / (y>0).sum(axis=0),
                           sample_percent=70,
                           samples=100)
    ax[p].fill_between(date_index,
                       pci[0],
                       pci[2],
                       color="C0",
                       alpha=.5)
    ax[p].plot(date_index,
               pci[1],
               color="C0",
               alpha=0.8,
               linestyle="--",
               linewidth=2)
    pname = "{} {}".format(MATCH_DICT[pkey]["name"], "Terms" if ptype == "terms" else "Subreddits")
    ax[p].set_title(pname, loc="left", fontweight="bold")
    ax[p].set_ylabel("Proportion\nof Users", fontweight="bold")
    ax[p].spines["top"].set_visible(False)
    ax[p].spines["right"].set_visible(False)
    ax[p].set_xlim(left=pd.to_datetime(ANALYSIS_START),right=pd.to_datetime(END_DATE))
ax[-1].set_xlabel("Date", fontweight="bold")
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}{PLATFORM}_term_subreddit_user_proportions.png", dpi=300)
plt.close()

## Plot User Proportion Entire History
LOGGER.info("Plotting Overall Match Proportions")
fig, ax = plt.subplots(figsize=(10,5.8))
max_val = -1
xticklabels = []
for p, (pkey, ptype) in enumerate(term_identifiers):
    pmatrix = term_vectors_normed[pkey][ptype]
    val = (pmatrix>0).any(axis=1).sum() / pmatrix.shape[0] * 100
    ax.bar(p,
           val,
           color=f"C{p}",
           alpha=0.75)
    ax.text(p,
            val + 2,
            "{:.2f}%".format(val),
            ha="center",
            va="center")
    xticklabels.append("{}\n{}".format(MATCH_DICT[pkey]["name"], "Terms" if ptype == "terms" else "Subreddits"))
    if val > max_val:
        max_val = val
ax.set_ylabel("Percentage of\nUsers", fontweight="bold")
ax.set_xticks(list(range(p+1)))
ax.set_xticklabels(xticklabels)
ax.set_ylim(bottom=0, top=max_val + 4)
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}{PLATFORM}_term_subreddit_user_proportions_overall.png", dpi=300)
plt.close()

## Posts Per Day (Compile and Cache)
posts_per_day = pd.Series(index=date_index, data=tau.sum(axis=0))
posts_per_day = pd.DataFrame(posts_per_day, columns=["num_posts"])
posts_per_day.to_csv(f"{CACHE_DIR}{PLATFORM}_{START_DATE}_{END_DATE}_posts_per_day.csv")

## Plot Each Term Over Time
for p, (pkey, ptype) in enumerate(term_identifiers):
    LOGGER.info(f"Plotting Individual Matches: {pkey} ({ptype})")
    ## Get Clean Name
    pname = "{} {}".format(MATCH_DICT[pkey]["name"], "Terms" if ptype == "terms" else "Subreddits")
    pname_clean = pname.replace(" ","_").lower()
    ## Matches Per Day
    matches_per_day = term_time_df[pkey][ptype]
    matches_per_day.to_csv(f"{CACHE_DIR}{PLATFORM}_{START_DATE}_{END_DATE}_matches_per_day_{pname_clean}.csv")
    ## Create Term/Subreddit Plots
    for term in tqdm(matches_per_day.columns, desc="Match", file=sys.stdout):
        ## Get Data and Check For Significance
        term_series = matches_per_day[term]
        if term_series.max() <= 5 or (term_series > 0).sum() < 10:
            continue
        ## Get Rolling Window
        term_series_normed = term_series.rolling(14).median()
        term_series_normed_lower = term_series.rolling(14).apply(lambda x: np.percentile(x, 25))
        term_series_normed_upper = term_series.rolling(14).apply(lambda x: np.percentile(x, 75))
        ## Generate Plot
        fig, ax = plt.subplots(figsize=(10,5.8))
        ax.fill_between(term_series_normed.index,
                        term_series_normed_lower,
                        term_series_normed_upper,
                        alpha=0.3)
        ax.plot(term_series_normed.index,
                term_series_normed.values,
                marker="o",
                linestyle="--",
                linewidth=1,
                color="C0",
                ms=3,
                alpha=0.5)
        ax.set_xlabel("Date", fontweight="bold")
        ax.set_ylabel("Posts Per Day (14-day Average)", fontweight="bold")
        ax.set_title(f"{pname}: {term}", fontweight="bold", loc="left")
        ax.set_xlim(left=pd.to_datetime(ANALYSIS_START),right=pd.to_datetime(END_DATE))
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        term_clean = term.replace("/","-").replace(".","-")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}timeseries/{pname_clean}_{term_clean}.png", dpi=300)
        plt.close(fig)

## Plot Top Match Frequencies
LOGGER.info("Plotting Top Match Frequencies")
for p, (pkey, ptype) in enumerate(term_identifiers):
    ## Get Clean Name
    pname = "{} {}".format(MATCH_DICT[pkey]["name"], "Terms" if ptype == "terms" else "Subreddits")
    pname_clean = pname.replace(" ","_").lower()
    ## Names
    group_map_r = dict((y,x) for x,y in term_maps[pkey][ptype].items())
    group_index = [group_map_r[i] for i in range(len(group_map_r))]
    ## Total (Across Users)
    vals = pd.Series(term_breakdowns[pkey][ptype].sum(axis=0),
                     index=group_index).sort_values().nlargest(30).iloc[::-1]
    ## Total (Per Users)
    vals_per_user = pd.Series((term_breakdowns[pkey][ptype]>0).sum(axis=0),
                              index=group_index).sort_values().nlargest(30).iloc[::-1]
    ## Plot
    fig, ax = plt.subplots(1,2,figsize=(10,5.8))
    vals.plot.barh(ax=ax[0],
                   color="C0",
                   alpha=.7)
    vals_per_user.plot.barh(ax=ax[1],
                            color="C0",
                            alpha=.7)
    ax[0].set_xlabel("Total Matches", fontweight="bold")
    ax[1].set_xlabel("Total Users with Match", fontweight="bold")
    fig.suptitle(pname, y=.98, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=.94)
    fig.savefig("{}top_{}_{}.png".format(PLOT_DIR, pname_clean, PLATFORM), dpi=300)
    plt.close()

###################
### Summary Plots
###################

LOGGER.info("Generating Summary Plots")

## Threshold
MIN_MATCHES = 150

## Aggregate by Week and Month
posts_per_day_filt = posts_per_day.loc[posts_per_day.index >= pd.to_datetime(ANALYSIS_START)]
posts_per_week_filt = posts_per_day_filt.resample("W-Mon").sum()
posts_per_month_filt = posts_per_day_filt.resample("MS").sum()

## Cycle Through Keywords
for p, (pkey, ptype) in enumerate(term_identifiers):
    ## Naming
    pname = "{} {}".format(MATCH_DICT[pkey]["name"], "Terms" if ptype == "terms" else "Subreddits")
    pname_clean = pname.replace(" ","_").lower()
    ## Isolate Daily Frequencies, Filtered By Plot Start
    kf_df = term_time_df[pkey][ptype].copy()
    kf_df = kf_df.loc[kf_df.index >= pd.to_datetime(ANALYSIS_START)]
    ## Create Aggregation by Week and Month
    kf_df_weekly = kf_df.resample('W-Mon').sum()
    kf_df_monthly = kf_df.resample("MS").sum()
    ## Posts by Period
    pre_covid_matched_posts = kf_df.loc[kf_df.index < pd.to_datetime(COVID_START)].sum(axis=0)
    pre_covid_posts = posts_per_day_filt.loc[posts_per_day_filt.index < pd.to_datetime(COVID_START)].sum()
    post_covid_matched_posts = kf_df.loc[kf_df.index >= pd.to_datetime(COVID_START)].sum(axis=0)
    posts_covid_posts = posts_per_day_filt.loc[posts_per_day_filt.index >= pd.to_datetime(COVID_START)].sum()
    ## Isolate By Threshold
    good_cols = kf_df.sum(axis=0).loc[kf_df.sum(axis=0) > MIN_MATCHES].index.tolist()
    kf_df = kf_df[good_cols].copy()
    ## Relative Posts by Period
    pre_covid_prop_posts = pre_covid_matched_posts / pre_covid_posts.item()
    post_covid_prop_posts = post_covid_matched_posts / posts_covid_posts.item()
    period_prop_change = post_covid_prop_posts - pre_covid_prop_posts
    period_pct_change = (period_prop_change / pre_covid_prop_posts).dropna().sort_values() * 100
    period_prop_change = period_prop_change.loc[good_cols]
    period_pct_change = period_pct_change.loc[good_cols]
    period_pct_change = period_pct_change.loc[period_pct_change!=np.inf]
    ## Create Summary Plot
    fig, ax = plt.subplots(2, 2, figsize=(12,8), sharex=False, sharey=False)
    ## Matches Over Time
    ax[0][0].plot(pd.to_datetime(kf_df_weekly.index),
                  kf_df_weekly.sum(axis=1) / posts_per_week_filt["num_posts"],
                  linewidth=2,
                  color="C0",
                  alpha=.7,
                  marker="o")
    ax[0][0].axvline(pd.to_datetime(COVID_START),
                     linestyle="--",
                     linewidth=2,
                     color="black",
                     alpha=0.5,
                     label="COVID-19 Start ({})".format(COVID_START))
    xticks = [i for i in kf_df_monthly.index if (i.month - 1) % 4 == 0]
    ax[0][0].set_xticks(xticks)
    ax[0][0].set_xticklabels([i.date() for i in xticks], rotation=25, ha="right")
    ax[0][0].legend(loc="upper left", frameon=True, framealpha=1)
    ax[0][0].set_ylabel("Matches Per Post (Weekly)", fontweight="bold")
    ax[0][0].set_xlabel("Week", fontweight="bold")
    ## Overall Match Rate Per Week
    ax[0][1].hist(kf_df_weekly.sum(axis=1) / posts_per_week_filt["num_posts"],
                  bins=15,
                  label="$\\mu={:.2f}, \\sigma={:.3f}$".format(
                      (kf_df_weekly.sum(axis=1) / posts_per_week_filt["num_posts"]).mean(),
                      (kf_df_weekly.sum(axis=1) / posts_per_week_filt["num_posts"]).std()
                  ),
                  alpha=.7)
    ax[0][1].set_ylabel("# Weeks", fontweight="bold")
    ax[0][1].set_xlabel("Matches Per Post (Weekly)", fontweight="bold")
    ax[0][1].legend(loc="upper right", frameon=True, facecolor="white", framealpha=1)
    ## Largest Proportional Differences
    nplot = min(10, int(len(period_prop_change)/2))
    plot_data = (period_prop_change.nlargest(nplot).append(period_prop_change.nsmallest(nplot))).sort_values()
    values = list(plot_data.values[:nplot]) + [0] +  list(plot_data.values[nplot:])
    ax[1][0].barh(list(range(nplot*2 + 1)),
                  values,
                  color = list(map(lambda i: "darkred" if i <= 0 else "navy", values)),
                  alpha = 0.6)
    ax[1][0].set_yticks(list(range(nplot*2 + 1)))
    ax[1][0].set_yticklabels(list(plot_data.index[:nplot]) + ["..."] + list(plot_data.index[nplot:]),
                             ha="right", va="center")
    ax[1][0].set_ylim(-.5, nplot*2 + .5)
    ax[1][0].axvline(0, color="black", linestyle="--", alpha=0.5)
    if ptype == "terms":
        ax[1][0].set_ylabel("Term", fontweight="bold")
    else:
        ax[1][0].set_ylabel("Subreddit", fontweight="bold")
    fmt = ticker.ScalarFormatter()
    fmt.set_powerlimits((-2,2))
    ax[1][0].xaxis.set_major_formatter(fmt)
    ax[1][0].set_xlabel("Absolute Change\n(Pre- vs. Post COVID-19 Start)", fontweight="bold")
    ax[1][0].ticklabel_format(axis="x", style="sci")
    ## Largest Percent Differences
    nplot = min(10, int(len(period_pct_change)/2))
    plot_data = (period_pct_change.nlargest(nplot).append(period_pct_change.nsmallest(nplot))).sort_values()
    values = list(plot_data.values[:nplot]) + [0] +  list(plot_data.values[nplot:])
    ax[1][1].barh(list(range(nplot*2 + 1)),
                  values,
                  color = list(map(lambda i: "darkred" if i <= 0 else "navy", values)),
                  alpha = 0.6)
    ax[1][1].set_yticks(list(range(nplot*2 + 1)))
    ax[1][1].set_yticklabels(list(plot_data.index[:nplot]) + ["..."] + list(plot_data.index[nplot:]),
                             ha="right", va="center")
    ax[1][1].set_ylim(-.5, nplot*2 + .5)
    ax[1][1].axvline(0, color="black", linestyle="--", alpha=0.5)
    if ptype =="terms":
        ax[1][1].set_ylabel("Term", fontweight="bold")
    else:
        ax[1][1].set_ylabel("Subreddit", fontweight="bold")
    ax[1][1].set_xlabel("Percent Change\n(Pre- vs. Post COVID-19 Start)", fontweight="bold")
    fig.tight_layout()
    fig.suptitle(pname, fontweight="bold", fontsize=14, y=.98)
    fig.subplots_adjust(top=.94)
    fig.savefig("{}summary_{}.png".format(PLOT_DIR, pname_clean), dpi=300)
    plt.close(fig)

###################
### Contextual Analysis
###################

LOGGER.info("Beginning Contextual Keyword Analysis")

## Contextual Analysis Parameters
DATE_BOUNDARIES = [START_DATE, COVID_START, END_DATE]
CONTEXT_WINDOW = None
PMI_ALPHA = 0.001
PMI_MIN_SUPPORT = 30

## Learn General Vocabulary Over Time (Date x Vocabulary Size)
vocab_cache_file = f"{CACHE_DIR}{PLATFORM}_{START_DATE}_{END_DATE}_vocabulary.joblib"
if not os.path.exists(vocab_cache_file) or RERUN:
    LOGGER.info("Learning Vocabulary")
    f2v = learn_vocabulary(filenames,
                           date_boundaries=DATE_BOUNDARIES)
    _ = joblib.dump(f2v, vocab_cache_file)
else:
    LOGGER.info("Loading Existing Vocabulary")
    f2v = load_vocabulary(vocab_cache_file)

## Vectorize Files
X_cache_file = f"{CACHE_DIR}{PLATFORM}_{START_DATE}_{COVID_START}_{END_DATE}_X.joblib"
if not os.path.exists(X_cache_file) or RERUN:
    LOGGER.info("Vectorizing Files")
    X = vectorize_files(filenames,
                        date_boundaries=DATE_BOUNDARIES,
                        f2v=f2v)
    _ = joblib.dump(X, X_cache_file)
else:
    LOGGER.info("Loading Vectorized Files")
    X = load_X(X_cache_file)

## Get Context-Matrices and Compute PMI
LOGGER.info("Computing PMI")
term_lists = [i for i in MATCH_DICT.keys() if MATCH_DICT.get(i).get("terms") is not None]
contexts = {}
pmi = {}
for tl in term_lists:
    LOGGER.info(f"Processing List: {tl}")
    ## Get Contexts
    tl_contexts = construct_matched_ngram_matrix(matches,
                                   term_list=tl,
                                   f2v=f2v,
                                   date_range_map=date_range_map,
                                   date_res=DATE_RES,
                                   ngram_window=CONTEXT_WINDOW)
    ## Re-Assign Dates to Desired Boundaries
    context_dates = [datetime(*date_range[i]) for i in tl_contexts[1]]
    context_dates = assign_dates_to_bins(context_dates, DATE_BOUNDARIES)
    ## Cache Contexts
    contexts[tl] = {
                "ngrams":tl_contexts[0],
                "dates":context_dates,
                "terms":tl_contexts[2],
                "categories":tl_contexts[3]
    }
    ## Compute PMI
    tl_pmi = {}
    for db in range(X.shape[0]):
        context_db_pmi, context_db_counts, context_ngram_count = compute_pmi_matrix(X,
                                            contexts[tl],
                                            f2v.vocab.get_ordered_vocabulary(),
                                            db,
                                            alpha=PMI_ALPHA,
                                            min_support=PMI_MIN_SUPPORT)
        tl_pmi[db] = {"pmi":context_db_pmi, "counts":context_db_counts, "ngram_counts":context_ngram_count}
    pmi[tl] = tl_pmi

## Get Analyzable Terms
context_analysis_terms = {}
for tl in term_lists:
    pmi_pre = pmi.get(tl).get(0).get("pmi")
    pmi_post = pmi.get(tl).get(1).get("pmi")
    context_analysis_terms[tl] = sorted(set(pmi_pre.index) & set(pmi_post.index))

## Visualize Context Change
LOGGER.info("Visualizing Context Change")
for tl, terms in tqdm(context_analysis_terms.items(), total=len(context_analysis_terms), position=0, leave=True, file=sys.stdout):
    for query_term in tqdm(terms, total=len(terms), position=1, leave=False, file=sys.stdout):
        fig = plot_context_change(query_term,
                                  term_list=tl,
                                  pmi_dict=pmi,
                                  min_tok_freq=10,
                                  top_k=20)
        qtclean = query_term.replace("/","-").replace(".","-")
        fig.savefig(f"{PLOT_DIR}context/{PLATFORM}_{tl}-{qtclean}.png", dpi=300)
        plt.close(fig)

LOGGER.info("Script complete!")