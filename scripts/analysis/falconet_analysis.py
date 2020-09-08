
##############################
### Configuration
##############################

## Paths
DATA_DIR = "./data/results/twitter/2018-2020/falconet-2020/"
PLOT_DIR = "./plots/twitter/2018-2020/falconet-2020/"

## Analysis Flags
KEYWORD_COUNTS = True
KEYWORD_DYNAMICS = True
MENTAL_HEALTH_COUNTS = True
MENTAL_HEALTH_TOPICS = True
MENTAL_HEALTH_REPRESENTATIVES = True

## Analysis Parameters
K_MATCHES_PER_WINDOW=2
SMOOTHING_WINDOW=30
COVID_START="2020-02-01"
MH_FIELDS=["anxiety","depression"]
MH_POSITIVE_THRESHOLD=0.9
MH_NUM_TOPICS=50
MH_TOPIC_NGRAM=(1,2)
MH_TOPIC_SAMPLE_RATE=0.1
MH_REP_NGRAM=(1,2)
MH_REP_SAMPLE_RATE=0.1
MH_FILTER_SET = { ## Filtering Applied to All Analysis
                "indorg":["ind"],
                "gender":["man","woman"],
                "is_united_states":["U.S.","Non-U.S."],
}

## Meta Parameters
NUM_JOBS = 8
RANDOM_STATE = 42

##############################
### Imports
##############################

## Standard Library
import os
import sys
import json
import gzip
import math
import random
import string
import textwrap
from glob import glob
from datetime import datetime
from multiprocessing import Pool
from collections import Counter
from functools import partial

## External Library
import joblib
import numpy as np
import pandas as pd
import demoji
from tqdm import tqdm
import matplotlib.pyplot as plt
from shap import LinearExplainer
from scipy.sparse import vstack, csr_matrix
from sklearn.feature_extraction import DictVectorizer
from pandas.plotting import register_matplotlib_converters
from statsmodels.stats.proportion import proportion_confint as bci
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer

## Local
from mhlib.util.helpers import flatten
from mhlib.util.logging import initialize_logger
from mhlib.preprocess.preprocess import tokenizer
from mhlib.preprocess.tokenizer import get_ngrams

##############################
### Globals
##############################

## Initialize Logger
LOGGER = initialize_logger()

## Probability Bins
MH_BINS=40
BIN_PREC=4

## Special Characters
SPECIAL = "“”…‘’´"

## Register Timestamp Converters for Plotting
_ = register_matplotlib_converters()

## Aggregations to Examine In Depth
AGGS = [(
            ["date"], ## Groups
            MH_FILTER_SET, ## Field Filter Set
            MH_FILTER_SET, ## Timestamp Filter Set
            False ## Group-level Normalization
        ),
        (
            ["date","gender"],
            MH_FILTER_SET,
            MH_FILTER_SET,
            True
        ),
        (
            ["date","is_united_states"],
            MH_FILTER_SET,
            MH_FILTER_SET,
            True
        ),
        (
            ["date","state"],
            {**MH_FILTER_SET, **{"country":["United States"], "state":["California","New York","Florida","Maryland"]}},
            {**MH_FILTER_SET, **{"country":["United States"]}},
            False
)]

##############################
### Helpers
##############################

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

def reverse_format_timestamp(time_tuple):
    """

    """
    return datetime(*list(time_tuple))

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
    for sample in tqdm(range(samples), total=samples, file=sys.stdout):
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

def load_file(filename,
              fields=[]):
    """
    Args:
        filename (str): Path to processed GZIP File
        fields (list of str): Which keys to extract. Extract all
                              if nothing passed. Options include:
                                - tweet_id
                                - user_id
                                - date
                                - text
                                - location
                                - keywords
                                - depression
                                - anxiety
                                - demographics
    
    Returns:
        data (list of dict): Desired Data
    """
    ## Load Desired Fields
    with gzip.open(filename, "r") as the_file:
        for line in the_file:
            line_data = json.loads(line)
            if not fields or "date" in fields:
                line_data["date"] = pd.to_datetime(line_data["date"])
            if not fields:
                yield line_data
            else:
                line_data_filtered = {}
                for f in fields:
                    line_data_filtered[f] = line_data.get(f)
                yield line_data_filtered

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

##############################
### Functions
##############################

def create_plot_dirs(plot_dir):
    """

    """
    subdirs = ["models/topics/","models/representative/","keywords/timeseries/"]
    for d in subdirs:
        if not os.path.exists(f"{plot_dir}/{d}/"):
            _ = os.makedirs(f"{plot_dir}/{d}/")

def _prob_bin_assigner(p,
                       bin_boundaries,
                       bin_size):
    """

    """        
    b = math.floor(p / bin_size)
    if p == 1:
        b -= 1
    return (round(bin_boundaries[b],BIN_PREC), round(bin_boundaries[b+1],BIN_PREC))

def _establish_bins(bins):
    """

    """
    bin_boundaries = [0]
    bin_size = 1/bins
    while bin_boundaries[-1] < 1:
        bin_boundaries.append(bin_boundaries[-1] + bin_size)
    return bin_size, bin_boundaries

def _count_fields(filename,
                  frequency="day",
                  fields=[],
                  bins=40,
                  keys=["date","demographics","location"]):
    """

    """
    ## counts
    timestamp_counts = Counter()
    field_counts = {field:{} for field in fields}
    ## Establish Bins
    bin_size, bin_boundaries = _establish_bins(bins)
    ## Parse File
    for line in load_file(filename, fields+keys):
        ## Extract Timestamp At Desired Resoulution
        line_date = _format_timestamp(line.get("date"), frequency)
        ## Identify Line Key
        line_key = []
        if "date" in keys:
            line_key.append(_format_timestamp(line.get("date"), frequency))
        if "demographics" in keys:
            line_key.append(line.get("demographics").get("gender"))
            line_key.append(line.get("demographics").get("indorg"))
        if "location" in keys:
            if line.get("location") is None:
                line_key.append(None)
                line_key.append(None)
                line_key.append(None)
            else:
                line_key.append(line.get("location").get("country"))
                line_key.append(line.get("location").get("state"))
                country = line.get("location").get("country")
                line_key.append("U.S." if country is not None and country == "United States" else "Non-U.S.")
        line_key = tuple(line_key)
        ## Update Counts
        timestamp_counts[line_key] += 1
        for field in fields:
            if line_key not in field_counts and line.get(field) is not None:
                field_counts[field][line_key] = Counter()
            if field == "keywords" and line.get("keywords") is not None:
                for keyword in line.get("keywords"):
                    field_counts[field][line_key][keyword] += 1
            elif field in set(MH_FIELDS) and line.get(field) is not None:
                pbin = _prob_bin_assigner(line.get(field), bin_boundaries, bin_size)
                field_counts[field][line_key][pbin] += 1
    return field_counts, timestamp_counts

def count_fields(filenames,
                 frequency="day",
                 fields=[],
                 bins=40,
                 keys=["date","demographics","location"]):
    """

    """
    ## Initialize Helper
    helper = partial(_count_fields, frequency=frequency, keys=keys, fields=fields, bins=bins)
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

def visualize_trends(field_counts,
                     timestamp_counts,
                     subset=None,
                     aggs=["date"],
                     visualize=True,
                     field_filter_sets={},
                     timestamp_filter_sets={},
                     group_normalize=False,
                     dropna=False,
                     smoothing_window=None,
                     min_support=100):
    """

    """
    ## Check Inputs
    if "date" not in aggs or aggs[0] != "date":
        raise ValueError("Expected date to be first aggs argument")
    if len(aggs) > 3 and visualize:
        LOGGER.warning("Asked for too many aggregations (>3)")    
    ## Copy
    field_counts_df = field_counts.copy()
    timestamp_counts_df = timestamp_counts.copy()
    ## Data Filtering
    if field_filter_sets:
        ## Apply Filtering to Keyword Set
        field_index = list(field_counts_df.index.names)
        field_counts_df.reset_index(inplace=True)
        for field, values in field_filter_sets.items():
            field_counts_df = field_counts_df.loc[field_counts_df[field].isin(set(values))].copy()
        field_counts_df = field_counts_df.set_index(field_index)
    if timestamp_filter_sets:
        ## Apply Filtering To Timestamp Set
        for field, values in timestamp_filter_sets.items():
            timestamp_counts_df = timestamp_counts_df.loc[timestamp_counts_df[field].isin(set(values))].copy()
    ## Keyword Selection (If Desired)
    if subset is not None:
        field_counts_df = field_counts_df[subset]
    ## Cleaning (Unknown / Missing Aggregation Fields)
    if dropna:
        field_counts_filt = field_counts_df.sum(axis=1).reset_index().dropna(subset=aggs,how="any")
        timestamp_counts_filt = timestamp_counts_df.dropna(subset=aggs,how="any")
    else:
        field_counts_filt = field_counts_df.sum(axis=1).reset_index().fillna("unk")
        timestamp_counts_filt = timestamp_counts_df.copy()
    ## Aggregate
    field_counts_aggs = field_counts_filt.groupby(aggs)[0].sum().reset_index().rename(columns={0:"count"})
    timestamp_counts_aggs = timestamp_counts_filt.fillna("unk").groupby(aggs)["count"].sum().reset_index()
    ## Format Date Index
    field_counts_aggs["date"] = field_counts_aggs["date"].map(reverse_format_timestamp)
    timestamp_counts_aggs["date"] = timestamp_counts_aggs["date"].map(reverse_format_timestamp)
    dates = pd.date_range(timestamp_counts_aggs["date"].min(), timestamp_counts_aggs["date"].max())
    date_counts = timestamp_counts_aggs.groupby(["date"])["count"].sum().reindex(dates)
    ## Return If Desired
    if not visualize:
        return field_counts_aggs, timestamp_counts_aggs
    ## Generate Plot
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.axvline(pd.to_datetime(COVID_START),
               linestyle="--",
               color="black",
               alpha=0.5,
               label="COVID-19 Start ({})".format(COVID_START),
               zorder=10)
    if len(aggs) == 1:
        fielddata = field_counts_aggs.set_index("date")["count"]
        denom = timestamp_counts_aggs.set_index("date")["count"] if group_normalize else date_counts
        fielddata = fielddata.reindex(dates).fillna(0)
        denom = denom.reindex(dates).fillna(0)
        if smoothing_window is None:
            plot_props_avg = fielddata / denom
        else:
            successes = fielddata.rolling(smoothing_window).sum().dropna()
            nobs = denom.rolling(smoothing_window).sum().dropna()
            plot_props_avg = successes / nobs
            plot_props_low, plot_props_high = bci(count=successes,
                                                  nobs=nobs + 1, ## Add 1 In Case of Missing Data
                                                  alpha=0.05,
                                                  method="normal")
        plot_props_avg.loc[denom<min_support] = np.nan
        plot_props_low.loc[denom<min_support] = np.nan
        plot_props_high.loc[denom<min_support] = np.nan
        if smoothing_window is not None:
            ax.fill_between(plot_props_low.index,
                            plot_props_low,
                            plot_props_high,
                            alpha=0.2)
        ax.plot(plot_props_avg.index,
                plot_props_avg,
                label = "Keyword Proportion",
                linewidth = 2,
                alpha = 0.8)
    else:
        ## Groups
        fieldgroups = field_counts_aggs.groupby(aggs[1:]).groups
        timegroups = timestamp_counts_aggs.groupby(aggs[1:]).groups
        ## Visualize
        for p, (plot_group, plot_index) in enumerate(fieldgroups.items()):
            fielddata = field_counts_aggs.loc[plot_index].set_index("date")["count"]
            denom = timestamp_counts_aggs.loc[timegroups[plot_group]].set_index("date")["count"] if group_normalize \
                    else date_counts
            fielddata = fielddata.reindex(dates).fillna(0)
            denom = denom.reindex(dates).fillna(0)
            if smoothing_window is None:
                plot_props_avg = fielddata / denom
            else:
                successes = fielddata.rolling(smoothing_window).sum().dropna()
                nobs = denom.rolling(smoothing_window).sum().dropna()
                plot_props_avg = successes / nobs
                plot_props_low, plot_props_high = bci(count=successes,
                                                    nobs=nobs + 1, ## Add 1 In Case of Missing Data
                                                    alpha=0.05,
                                                    method="normal")
            plot_props_avg.loc[denom<min_support] = np.nan
            plot_props_low.loc[denom<min_support] = np.nan
            plot_props_high.loc[denom<min_support] = np.nan
            if smoothing_window is not None:
                ax.fill_between(plot_props_low.index,
                                plot_props_low,
                                plot_props_high,
                                alpha=0.2,
                                color=f"C{p}")
            ax.plot(plot_props_avg.index,
                    plot_props_avg,
                    label = plot_group.title(),
                    alpha = 0.8,
                    linewidth=2,
                    color=f"C{p}")
        leg = ax.legend(loc="lower left", frameon=True, title="Group", framealpha=0.9)
    if not group_normalize:
        ax.set_ylabel("Proportion of All Posts", fontweight="bold")
    else:
        ax.set_ylabel("Proportion of Posts from Group", fontweight="bold")
    ax.set_xlabel("Date", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax

def visualize_keyword_change(field_counts,
                             timestamp_counts,
                             subset=None,
                             aggs=[],
                             field_filter_sets={"indorg":["ind"]},
                             timestamp_filter_sets={"indorg":["ind"]},
                             change_point=COVID_START, ## TODO: Year Over Year Change Option
                             group_normalize=True,
                             dropna=False,
                             min_freq_per_group=5,
                             k_top=15):
    """

    """
    ## Copy
    field_counts_df = field_counts.copy()
    timestamp_counts_df = timestamp_counts.copy()
    ## Data Filtering
    if field_filter_sets:
        ## Apply Filtering to Keyword Set
        field_index = list(field_counts_df.index.names)
        field_counts_df.reset_index(inplace=True)
        for field, values in field_filter_sets.items():
            field_counts_df = field_counts_df.loc[field_counts_df[field].isin(set(values))].copy()
        field_counts_df = field_counts_df.set_index(field_index)
    if timestamp_filter_sets:
        ## Apply Filtering To Timestamp Set
        for field, values in timestamp_filter_sets.items():
            timestamp_counts_df = timestamp_counts_df.loc[timestamp_counts_df[field].isin(set(values))].copy()
    ## Keyword Selection (If Desired)
    if subset is not None:
        field_counts_df = field_counts_df[subset]
    ## Cleaning (Unknown / Missing Aggregation Fields)
    field_index = list(field_counts_df.index.names)
    field_columns = list(field_counts_df.columns)
    if dropna:
        field_counts_df = field_counts_df.reset_index().dropna(subset=aggs,how="any")
        timestamp_counts_df = timestamp_counts_df.dropna(subset=aggs,how="any")
    else:
        field_counts_df = field_counts_df.reset_index()
        for a in aggs:
            field_counts_df[a] = field_counts_df[a].fillna("unk")
            timestamp_counts_df[a] = timestamp_counts_df[a].fillna("unk")
    ## Get Date Boundaries
    date_boundaries = {"pre":[],"post":[]}
    change_point_dt = pd.to_datetime(change_point)
    for d in timestamp_counts_df["date"].unique():
        if reverse_format_timestamp(d) < change_point_dt:
            date_boundaries["pre"].append(d)
        else:
            date_boundaries["post"].append(d)
    ## Aggregate Counts
    if aggs:
        field_counts_pre = field_counts_df.loc[field_counts_df["date"].isin(set(date_boundaries["pre"]))].\
                            groupby(aggs)[field_columns].sum().T
        field_counts_post = field_counts_df.loc[field_counts_df["date"].isin(set(date_boundaries["post"]))].\
                            groupby(aggs)[field_columns].sum().T
        timestamp_counts_pre = timestamp_counts_df.loc[timestamp_counts_df["date"].isin(set(date_boundaries["pre"]))].groupby(aggs).size()
        timestamp_counts_post = timestamp_counts_df.loc[timestamp_counts_df["date"].isin(set(date_boundaries["post"]))].groupby(aggs).size()
    else:
        field_counts_pre = field_counts_df.loc[field_counts_df["date"].isin(set(date_boundaries["pre"]))]\
                            [field_columns].sum(axis=0).to_frame("total")
        field_counts_post = field_counts_df.loc[field_counts_df["date"].isin(set(date_boundaries["post"]))]\
                            [field_columns].sum(axis=0).to_frame("total")
        timestamp_counts_pre = pd.Series(index=["total"],
                                         data=[len(timestamp_counts_df.loc[timestamp_counts_df["date"].isin(set(date_boundaries["pre"]))])])
        timestamp_counts_post = pd.Series(index=["total"],
                                          data=len(timestamp_counts_df.loc[timestamp_counts_df["date"].isin(set(date_boundaries["post"]))]))
    pre_total = timestamp_counts_pre.sum()
    post_total = timestamp_counts_post.sum()
    ## Frequency Filtering
    field_counts_pre = field_counts_pre.loc[(field_counts_pre >= min_freq_per_group).all(axis=1)]
    field_counts_post = field_counts_post.loc[(field_counts_post >= min_freq_per_group).all(axis=1)]
    terms = sorted(set(field_counts_pre.index) & set(field_counts_post.index))
    ## Cycle Through Groups, Plotting Dynamics
    groups = sorted(set(timestamp_counts_pre.index) | set(timestamp_counts_post.index))
    fig, ax = plt.subplots(1, len(groups), figsize=(10,5.8))
    for g, group in enumerate(groups):
        ## Isolate Group Data
        counts_pre = field_counts_pre.loc[terms, group]
        counts_post = field_counts_post.loc[terms, group]
        denom_pre = timestamp_counts_pre.loc[group] if group_normalize else pre_total
        denom_post = timestamp_counts_post.loc[group] if group_normalize else post_total
        ## Compute Ratio
        ratio = np.log((counts_post / denom_post) / (counts_pre / denom_pre)).sort_values()
        ratio = (ratio.nsmallest(k_top).append(ratio.nlargest(k_top))).drop_duplicates().sort_values()
        ## Compute Confidence Intervals
        pre_ci_low, pre_ci_high = bci(counts_pre, denom_pre, alpha=0.05)
        post_ci_low, post_ci_high = bci(counts_post, denom_post, alpha=0.05)
        ratio_high = np.log(post_ci_high / pre_ci_low).loc[ratio.index]
        ratio_low = np.log(post_ci_low / pre_ci_high).loc[ratio.index]
        if len(groups) == 1:
            pax = ax
        else:
            pax = ax[g]
        pax.barh(np.arange(0, len(ratio)),
                 left=ratio_low,
                 width=ratio_high-ratio_low,
                 color=f"C{g}",
                 alpha=0.2)
        pax.scatter(ratio,
                    np.arange(0, len(ratio)),
                    color=f"C{g}",
                    alpha=0.8,
                    marker="o",
                    s=10)
        pax.axvline(0, color="black", linestyle="--", alpha=0.8)
        pax.set_yticks(np.arange(0, len(ratio)))
        pax.set_yticklabels(ratio.index.tolist(), fontsize=8)
        pax.set_xlabel("Match Rate Log Ratio\n(Pre/Post {})".format(change_point), fontweight="bold")
        pax.set_title("{} Term Dynamics".format(group.title()), fontweight="bold", fontstyle="italic", loc="left")
        pax.set_ylim(-.5, len(ratio)-.5)
    fig.tight_layout()
    return fig, ax

def _initialize_dict_vectorizer(vocabulary):
    """
    Initialize a vectorizer that transforms a counter dictionary
    into a sparse vector of counts (with a uniform feature index)

    Args:
        vocabulary (iterable): Input vocabulary
    
    Returns:
        _count2vec (DictVectorizer): Transformer
    """
    ## Sort
    vocabulary = sorted(vocabulary)
    ## Initialize Vectorizer
    _count2vec = DictVectorizer(separator=":",
                                dtype=int,
                                sort=False)
    ## Update Attributes
    _count2vec.vocabulary_ = dict((x, i) for i, x in enumerate(vocabulary))
    _count2vec.feature_names_ = vocabulary
    return _count2vec

def _ignore_line(data,
                 filters):
    """

    """
    ignore = False
    for f, fval in filters.items():
        if f in ["gender","indorg"]:
            if data.get("demographics") is None:
                ignore = True
            if not data.get("demographics").get(f) in fval:
                ignore = True
        if f in ["country","state","city"]:
            if data.get("location") is None:
                ignore = True
            if not data.get("location").get(f) in fval:
                ignore = True
        if f == "is_united_states":
            if data.get("location") is None:
                ignore = True
            else:
                country = data.get("country")
                if country == "United States":
                    is_united_states = "U.S."
                else:
                    is_united_states = "Non-U.S."
                if not is_united_states in fval:
                    ignore = False
        if ignore:
            break
    return ignore

def get_topic_distribution(files,
                           condition,
                           filters={},
                           threshold=0.5,
                           n_topics=25,
                           min_freq=10,
                           max_percentile=99,
                           max_vocab_size=100000,
                           min_n_gram=1,
                           max_n_gram=1,
                           k_representatives=30,
                           max_iter=100,
                           sample_rate=1):
    """

    """
    ## Learn Vocabulary
    vocab = Counter()
    for file in tqdm(files, desc="Learning Vocabulary", file=sys.stdout):
        sampler = random.Random(RANDOM_STATE)
        text_samples = []
        with gzip.open(file,"r") as the_file:
            for line in the_file:
                if sampler.uniform(0,1) > sample_rate:
                    continue
                line_data = json.loads(line)
                if line_data.get(condition) < threshold:
                    continue
                if filters and _ignore_line(line_data, filters):
                    continue
                text = line_data.get("text")
                text_samples.append(text.lower())
        text_tokens = list(map(tokenizer.tokenize, text_samples))
        text_tokens = [list(filter(lambda t: not t.startswith("<") and not any(char in string.punctuation or char in SPECIAL for char in t),i)) for i in text_tokens]
        text_tokens = list(map(replace_emojis, text_tokens))
        text_tokens = list(map(lambda t: get_ngrams(t, min_n_gram, max_n_gram), text_tokens))
        text_counts = Counter(flatten(text_tokens))
        vocab += text_counts
    ## Filter Vocabulary (Frequency and Size)
    vocab = Counter(dict((i, v) for i, v in vocab.items() if v >= min_freq))
    vocab = vocab.most_common(max_vocab_size)
    vocab = pd.DataFrame(vocab)
    vocab = vocab.loc[vocab[1] < np.percentile(vocab[1], max_percentile)]
    vocab = vocab[0].tolist()
    ## Generate Document-Term Matrix
    dvec = _initialize_dict_vectorizer(vocab)
    X = []
    tau = []
    for file in tqdm(files, desc="Generating Document Term Matrix", file=sys.stdout):
        sampler = random.Random(RANDOM_STATE)
        text_samples = []
        with gzip.open(file,"r") as the_file:
            for line in the_file:
                if sampler.uniform(0,1) > sample_rate:
                    continue
                line_data = json.loads(line)
                if filters and _ignore_line(line_data, filters):
                    continue
                if line_data.get(condition) < threshold:
                    continue
                text = line_data.get("text")
                text_samples.append(text.lower())
                tau.append(line_data.get("date"))
        if len(text_samples) == 0:
            continue
        text_tokens = list(map(tokenizer.tokenize, text_samples))
        text_tokens = [list(filter(lambda t: not t.startswith("<") and not any(char in string.punctuation for char in t),i)) for i in text_tokens]
        text_tokens = list(map(replace_emojis, text_tokens))
        text_tokens = list(map(lambda t: get_ngrams(t, min_n_gram, max_n_gram), text_tokens))
        text_tokens = list(map(Counter, text_tokens))
        X.append(dvec.transform(text_tokens))
    X = vstack(X)
    tau = pd.to_datetime(tau)
    ## Fit Model
    LOGGER.info("Fitting LDA Model with {} Topics".format(n_topics))
    lda = LDA(n_components=n_topics, random_state=RANDOM_STATE, max_iter=max_iter, verbose=1)
    lda.fit(X)
    ## Compute Topic Distribution
    LOGGER.info("Computing Document-Topic Distribution")
    topic_dist = lda.transform(X)
    ## Get Representative Terms
    LOGGER.info("Isolating Representative Terms")
    representatives = {}
    for topic, topic_term_dist in enumerate(lda.components_):
        topic_representatives = [dvec.feature_names_[i] for i in np.argsort(topic_term_dist)[-k_representatives:][::-1]]
        representatives[topic+1] = topic_representatives
        LOGGER.info("{}) {}".format(topic+1, ", ".join([" ".join(i) for i in topic_representatives[:10]])))
    return dvec, lda, topic_dist, tau, representatives

def visualize_topic_trends(topic_dist,
                           tau,
                           representatives,
                           k_top=10,
                           weighted=False,
                           smoothing_window=1):
    """

    """
    ## Isolate Dates
    tau_dt = [i.date() for i in tau]
    ## Smoothed Distribution Over Time
    if weighted:
        topic_dist_agg = pd.pivot_table(pd.DataFrame(np.hstack([np.array(tau_dt).reshape(-1,1), topic_dist]),
                                                     columns=["date"] + list(range(topic_dist.shape[1]))),
                                        index="date",
                                        values=list(range(topic_dist.shape[1])),
                                        aggfunc=sum)
    else:
        topic_dist_agg = pd.pivot_table(pd.DataFrame(np.vstack([tau_dt, topic_dist.argmax(axis=1)]).T),
                                        columns=1,
                                        index=0,
                                        aggfunc=len).fillna(0).astype(int)
    topic_dist_agg_smooth = topic_dist_agg.rolling(SMOOTHING_WINDOW, axis=0).sum()
    topic_dist_agg_smooth = topic_dist_agg_smooth.apply(lambda x: x / sum(x), axis=1)
    topic_dist_agg_smooth.dropna(inplace=True)
    ## Change Counts
    topic_count_pre = topic_dist_agg.loc[topic_dist_agg.index < pd.to_datetime(COVID_START)].sum(axis=0)
    topic_count_post = topic_dist_agg.loc[topic_dist_agg.index >= pd.to_datetime(COVID_START)].sum(axis=0)
    topic_prev_pre = topic_count_pre / topic_count_pre.sum()
    topic_prev_post = topic_count_post / topic_count_post.sum()
    ## Compute Ratio
    ratio = np.log(topic_prev_post / topic_prev_pre).sort_values()
    ratio = (ratio.nsmallest(k_top).append(ratio.nlargest(k_top))).drop_duplicates().sort_values()
    ## Compute Confidence Intervals
    pre_ci_low, pre_ci_high = bci(topic_count_pre, topic_count_pre.sum(), alpha=0.05)
    post_ci_low, post_ci_high = bci(topic_count_post, topic_count_post.sum(), alpha=0.05)
    ratio_high = np.log(post_ci_high / pre_ci_low).loc[ratio.index]
    ratio_low = np.log(post_ci_low / pre_ci_high).loc[ratio.index]
    ## Plot Figure
    fig, ax = plt.subplots(1, 2, figsize=(10,5.8))
    lower = np.zeros(topic_dist_agg_smooth.shape[0])
    for topic in topic_dist_agg_smooth.values.T:
        ax[0].fill_between(topic_dist_agg_smooth.index,
                           lower,
                           lower + topic,
                           alpha = 0.5)
        lower = lower + topic
    ax[1].barh(np.arange(0, len(ratio)),
                left=ratio_low,
                width=ratio_high-ratio_low,
                color=f"C0",
                alpha=0.2)
    ax[1].scatter(ratio,
                  np.arange(0, len(ratio)),
                  color=f"C0",
                  alpha=0.8,
                  marker="o",
                  s=10)
    ax[0].set_xlabel("Date", fontweight="bold")
    ax[0].set_ylabel("Topic Proportion", fontweight="bold")
    ax[0].set_xlim(topic_dist_agg_smooth.index.min(), topic_dist_agg_smooth.index.max())
    ax[0].set_ylim(0,1)
    ax[0].axvline(COVID_START, color="black", linestyle="--")
    ax[1].axvline(0, color="black", linestyle="--", alpha=0.8)
    ax[1].set_yticks(np.arange(0, len(ratio)))
    ax[1].set_yticklabels(["{}) ".format(i+1) + ", ".join([" ".join(j) for j in representatives[i+1][:8]]) for i in ratio.index.tolist()],
                          fontsize=8)
    ax[1].yaxis.tick_right()
    ax[1].set_ylim(-0.5, len(ratio)-.5)
    ax[1].set_xlabel("Topic Prevalence Ratio\n(Pre/Post {})".format(COVID_START), fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax

def get_representative_tweets(files,
                              condition,
                              filters={},
                              min_n_gram=1,
                              max_n_gram=1,
                              min_freq=10,
                              pos_threshold=0.9,
                              neg_threshold=0.1,
                              sample_rate=1,
                              random_state=42):
    """

    """
    ## Representative Tweet Cache
    representative_tweets = {"negative":[],"positive":[]}
    tau = {"negative":[], "positive":[]}
    scores = {"negative":[], "positive":[]}
    ## Cycle Through Files
    for f in tqdm(files, "Representative Tweet Search"):
        sampler = random.Random(random_state)
        with gzip.open(f,"r") as the_file:
            for line in the_file:
                if sampler.uniform(0,1) > sample_rate:
                    continue
                line_data = json.loads(line)
                if filters and _ignore_line(line_data, filters):
                    continue
                if line_data.get(condition) > pos_threshold:
                    key = "positive"
                elif line_data.get(condition) < neg_threshold:
                    key = "negative"
                else:
                    continue
                representative_tweets[key].append(line_data.get("text"))
                tau[key].append(line_data.get("date"))
                scores[key].append(line_data.get(condition))
    ## Sizes
    n_neg = len(representative_tweets["negative"])
    n_pos = len(representative_tweets["positive"])
    LOGGER.info("Identified {:,d} negative samples and {:,d} positive samples".format(n_neg, n_pos))
    ## Format Dates
    for key, vals in tau.items():
        tau[key] = [i.date() for i in pd.to_datetime(vals)]
    ## Tokenize Examples
    LOGGER.info("Tokenizing Examples")
    tokens = list(map(lambda x: tokenizer.tokenize(x.lower()), representative_tweets.get("positive"))) + \
             list(map(lambda x: tokenizer.tokenize(x.lower()), representative_tweets.get("negative")))
    tokens = list(map(lambda i: list(filter(lambda t: not t.startswith("<") and t not in string.punctuation + SPECIAL, i)), tokens))    
    tokens = list(map(lambda t: get_ngrams(t, min_n_gram, max_n_gram), tokens))
    ## Vectorize
    LOGGER.info("Vectorizing Examples")
    vocab = Counter(flatten(tokens))
    dvec = _initialize_dict_vectorizer(list(vocab))
    X = dvec.transform(list(map(Counter, tokens)))
    y = np.array([1]*n_pos+[0]*n_neg)
    ## Sample Mask
    sample_mask = np.nonzero(X.sum(axis=1) != 0)[0]
    X = X[sample_mask]
    y = y[sample_mask]
    ## Feature Mask
    feature_mask = np.nonzero(X.sum(axis=0) >= min_freq)[1]
    X = X[:,feature_mask]
    ## Transform
    tfidf = TfidfTransformer()
    X = tfidf.fit_transform(X)
    ## Fit Secondary Classifier
    LOGGER.info("Fitting Representative Classifier")
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict_proba(X)[:,1]
    ## Shap Explanations
    LOGGER.info("Computing Shap Values")
    le = LinearExplainer(model=model, data=X)
    shap_vals = le.shap_values(X)
    features = [" ".join(dvec.feature_names_[i]) for i in feature_mask]
    ## Bootstrap Shap Contribution
    LOGGER.info("Computing Bootstrap Shap Value Range")
    shap_ci = bootstrap_sample(shap_vals, func=np.mean, axis=0, sample_percent=70, samples=50)
    shap_ci = pd.DataFrame(index=features, data=shap_ci.T, columns=["lower","median","upper"])
    shap_ci.index = replace_emojis(shap_ci.index)
    ## Filter Representative Tweets
    LOGGER.info("Flattening Data")
    tau_flat = tau.get("positive") + tau.get("negative"); tau_flat = [tau_flat[i] for i in sample_mask]
    text_flat = representative_tweets.get("positive") + representative_tweets.get("negative"); text_flat = [text_flat[i] for i in sample_mask]
    scores_flat = scores.get("positive") + scores.get("negative"); scores_flat = [scores_flat[i] for i in sample_mask]
    ## Select Representative Tweets
    LOGGER.info("Isolating Representative Tweets")
    tau_flat_month = [(i.year, i.month) for i in tau_flat]
    months = sorted(set(tau_flat_month))
    monthly_reps = {month:[] for month in months}
    summary_str = []
    for month in tqdm(months, file=sys.stdout):
        month_mask = [i for i, m in enumerate(tau_flat_month) if m == month]
        month_preds = [scores_flat[i] for i in month_mask]
        month_rank = np.argsort(month_preds)[::-1]
        month_mask = [month_mask[i] for i in month_rank[:10]]
        month_tweets = [text_flat[i] for i in month_mask]
        month_preds = [scores_flat[i] for i in month_mask]
        month_shap = shap_vals[month_mask]
        summary_str.append("#"*100 + "\n### Month of {}\n".format(month) + "#"*100)
        count = 1
        for pred, tweet, shap in zip(month_preds, month_tweets, month_shap):
            monthly_reps[month].append((pred,
                                        tweet,
                                        sorted(zip(features, shap), key=lambda x: x[1])[-10:][::-1]))
            pstr = "{}) [{:.3f}] {}".format(count, pred, tweet)
            pstr = "\n\t\t".join(textwrap.wrap(pstr, 80))
            summary_str.append(pstr)
            summary_str.append("   Influence:")
            for term, val in monthly_reps[month][-1][-1]:
                summary_str.append("\t* {} ({:.4f})".format(term ,val))
            count+=1
    summary_str = "\n".join(summary_str)
    ## Shap Visualization
    top_shap = shap_ci["median"].nlargest(15).append(shap_ci["median"].nsmallest(15)).sort_values().drop_duplicates().index
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.barh(list(range(len(top_shap))),
            left=shap_ci.loc[top_shap]["lower"],
            width=shap_ci.loc[top_shap]["upper"]-shap_ci.loc[top_shap]["lower"],
            alpha=0.5,
            color="C0")
    ax.scatter(shap_ci.loc[top_shap]["median"],
               list(range(len(top_shap))),
               color="C0",
               s=10)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_yticks(list(range(len(top_shap))))
    ax.set_yticklabels(top_shap)
    ax.set_xlabel("Shap Value (Post-hoc)", fontweight="bold")
    ax.set_ylabel("N-gram", fontweight="bold")
    ax.set_ylim(-.5, len(top_shap)-0.5)
    fig.tight_layout()
    return (fig, ax), summary_str, shap_ci

def main():
    """

    """
    ## Identify Data Files
    files = sorted(glob(f"{DATA_DIR}*.gz"))
    ## Plot Directory
    _ = create_plot_dirs(PLOT_DIR)
    ## Keyword Analysis
    if KEYWORD_COUNTS or KEYWORD_COUNTS:
        ## Run Counter
        LOGGER.info("Counting Keywords")
        keyword_counts, timestamp_counts = count_fields(files,
                                                        frequency="day",
                                                        fields=["keywords"],
                                                        keys=["date","demographics","location"])
        ## Count Analysis Over Time
        if KEYWORD_COUNTS:
            ## Isolate Keyword Counts
            LOGGER.info("Visualizing Keyword Matches Over Time (Aggregations)")
            for aggset, field_filter, time_filter, gnorm in AGGS:
                fig, ax = visualize_trends(keyword_counts.loc["keywords"],
                                           timestamp_counts,
                                           subset=None,
                                           aggs=aggset,
                                           visualize=True,
                                           field_filter_sets=field_filter,
                                           timestamp_filter_sets=time_filter,
                                           group_normalize=gnorm,
                                           dropna=True,
                                           smoothing_window=SMOOTHING_WINDOW)
                fig.savefig("{}/keywords/{}_keyword_proportion.png".format(PLOT_DIR, "-".join(aggset)), dpi=300)
                plt.close(fig)
            ## Individual Keyword Breakdown (Require Average of k Per Window)
            keyword_plot_thresh = len(timestamp_counts["date"].unique()) * K_MATCHES_PER_WINDOW / SMOOTHING_WINDOW
            keywords_to_plot = keyword_counts.loc["keywords"].sum(axis=0).loc[keyword_counts.loc["keywords"].sum(axis=0) > keyword_plot_thresh].index.tolist()
            LOGGER.info("Visualizing Keyword Matches Over Time (Individual Breakdowns)")
            for keyword in tqdm(keywords_to_plot, desc="Keyword", file=sys.stdout):
                fig, ax = visualize_trends(keyword_counts.loc["keywords"],
                                           timestamp_counts,
                                           subset=[keyword],
                                           aggs=["date"],
                                           visualize=True,
                                           field_filter_sets=MH_FILTER_SET,
                                           timestamp_filter_sets=MH_FILTER_SET,
                                           group_normalize=True,
                                           dropna=True,
                                           smoothing_window=SMOOTHING_WINDOW)
                keyword_clean = keyword.replace(" ","-").replace(".","-").replace("/","-")
                fig.savefig("{}/keywords/timeseries/{}_keyword_proportion.png".format(PLOT_DIR, keyword_clean), dpi=300)
                plt.close(fig)
        ## Keyword Dynamics
        if KEYWORD_DYNAMICS:
            LOGGER.info("Visualizing Keyword Dynamics")
            ## Isolate Keyword Counts
            for aggset, field_filter, time_filter, gnorm in AGGS:
                try:
                    fig, ax = visualize_keyword_change(keyword_counts.loc["keywords"],
                                                       timestamp_counts,
                                                       subset=None,
                                                       aggs=aggset[1:],
                                                       field_filter_sets=field_filter,
                                                       timestamp_filter_sets=time_filter,
                                                       change_point=COVID_START,
                                                       group_normalize=gnorm,
                                                       dropna=True,
                                                       min_freq_per_group=5,
                                                       k_top=15)
                    fig.savefig("{}/keywords/{}_keyword_dynamics.png".format(PLOT_DIR, "-".join(aggset)), dpi=300)
                    plt.close(fig)
                except:
                    LOGGER.warning("Encountered Issue Visualizing Keyword Change for Aggset: {}".format(aggset))
                    plt.close("all")
                    continue
    ## Classifier Predictions
    if MENTAL_HEALTH_COUNTS:
        ## Run Counter
        LOGGER.info("Counting Classifier Predictions")
        mental_health_counts, timestamp_counts = count_fields(files,
                                                              frequency="day",
                                                              fields=MH_FIELDS,
                                                              bins=MH_BINS,
                                                              keys=["date","demographics","location"])
        ## Isolate Positive Bins
        _, bins = _establish_bins(MH_BINS)
        positive_bins = []
        i = 0
        while i < MH_BINS:
            if bins[i] < MH_POSITIVE_THRESHOLD:
                i += 1
                continue
            positive_bins.append((round(bins[i],BIN_PREC), round(bins[i+1],BIN_PREC)))
            i += 1
        ## Cycle Through Conditions
        for condition in MH_FIELDS:
            ## Get Counts
            LOGGER.info(f"Visualizing Trends: {condition}")
            condition_counts = mental_health_counts.loc[condition].copy()
            for aggset, field_filter, time_filter, gnorm in AGGS:
                fig, ax = visualize_trends(condition_counts,
                                           timestamp_counts,
                                           subset=positive_bins,
                                           aggs=aggset,
                                           visualize=True,
                                           field_filter_sets=field_filter,
                                           timestamp_filter_sets=time_filter,
                                           group_normalize=gnorm,
                                           dropna=True,
                                           smoothing_window=SMOOTHING_WINDOW)
                fig.savefig("{}/models/{}_{}_proportion.png".format(PLOT_DIR, condition, "-".join(aggset)), dpi=300)
                plt.close(fig)
    ## Topic Modeling
    if MENTAL_HEALTH_TOPICS:
        ## Cycle Through Conditions
        for condition in MH_FIELDS:
            LOGGER.info("Learning Topic Distribution for Condition: {}".format(condition))
            ## Compute Topic Distribution
            dvec, lda, topic_dist, tau, representatives = get_topic_distribution(files,
                                                                                 condition,
                                                                                 filters=MH_FILTER_SET,
                                                                                 threshold=MH_POSITIVE_THRESHOLD,
                                                                                 n_topics=MH_NUM_TOPICS,
                                                                                 min_freq=10,
                                                                                 max_percentile=99,
                                                                                 max_vocab_size=100000,
                                                                                 min_n_gram=MH_TOPIC_NGRAM[0],
                                                                                 max_n_gram=MH_TOPIC_NGRAM[1],
                                                                                 k_representatives=30,
                                                                                 max_iter=100,
                                                                                 sample_rate=MH_TOPIC_SAMPLE_RATE)
            ## Visualize Topic Dynamics Over Time
            fig, ax = visualize_topic_trends(topic_dist,
                                             tau,
                                             representatives,
                                             k_top=10,
                                             weighted=False,
                                             smoothing_window=SMOOTHING_WINDOW)
            fig.savefig(f"{PLOT_DIR}models/topics/{condition}_topic_dynamics.png", dpi=300)
            plt.close(fig)
    ## Representative Examples
    if MENTAL_HEALTH_REPRESENTATIVES:
        for condition in MH_FILTER_SET:
            LOGGER.info(f"Identifying Representative Posts for Condition: {condition}")
            ## Generate Summary Data
            (fig, ax), summary_str, shap_ci = get_representative_tweets(files,
                                                                        condition,
                                                                        filters=MH_FILTER_SET,
                                                                        min_n_gram=MH_REP_NGRAM[0],
                                                                        max_n_gram=MH_REP_NGRAM[1],
                                                                        min_freq=10,
                                                                        pos_threshold=0.9,
                                                                        neg_threshold=0.1,
                                                                        sample_rate=MH_REP_SAMPLE_RATE,
                                                                        random_state=RANDOM_STATE)
            ## Save Figure
            fig.savefig(f"{PLOT_DIR}models/representative/{condition}_post_hoc_shap_values.png", dpi=300)
            plt.close(fig)
            ## Save Summary
            with open(f"{PLOT_DIR}models/representative/{condition}_representative_posts.txt","w") as the_file:
                the_file.write(summary_str)
            ## Save Shap Value Ranges
            shap_ci.to_csv(f"{PLOT_DIR}models/representative/{condition}_shap_values.csv")

#########################
### Execute
#########################

if __name__ == "__main__":
    _ = main()