
##############################
### Configuration
##############################

## Paths
DATA_DIR = "./data/results/twitter/2018-2020/falconet-full/"
PLOT_DIR = "./plots/twitter/2018-2020/falconet-full/"

## Analysis Flags
KEYWORD_COUNTS = True
KEYWORD_DYNAMICS = True
MENTAL_HEALTH_COUNTS = True

## Analysis Parameters
MH_BINS=40
MH_FIELDS=["anxiety","depression"]
MH_POSITIVE_THRESHOLD=0.8
SMOOTHING_WINDOW=30
COVID_START="2020-03-01"
AGGS = [(
            ["date"], ## Groups
            {"indorg":["ind"]}, ## Field Filter Set
            {"indorg":["ind"]}, ## Timestamp Filter Set
            False ## Group-level Normalization
        ),
        (
            ["date","gender"],
            {"indorg":["ind"],"gender":["man","woman"]},
            {"indorg":["ind"],"gender":["man","woman"]},
            True
        ),
        (
            ["date","is_united_states"],
            {"indorg":["ind"],"is_united_states":["U.S.","Non-U.S."]},
            {"indorg":["ind"],"is_united_states":["U.S.","Non-U.S."]},
            True
        ),
        (
            ["date","state"],
            {"indorg":["ind"],"country":["United States"], "state":["California","New York","Florida","Maryland"]},
            {"indorg":["ind"],"country":["United States"]},
            False
)]

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
from glob import glob
from datetime import datetime
from multiprocessing import Pool
from collections import Counter
from functools import partial

## External Library
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.stats.proportion import proportion_confint as bci

## Local
from mhlib.util.logging import initialize_logger

##############################
### Globals
##############################

## Initialize Logger
LOGGER = initialize_logger()

## Precision of Bin Rounding
BIN_PREC = 4

## Register Timestamp Converters for Plotting
_ = register_matplotlib_converters()

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

##############################
### Functions
##############################

def create_plot_dirs(plot_dir):
    """

    """
    subdirs = ["keywords","models","keywords/timeseries/","keywords/context/"]
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
                            groupby(aggs)[field_columns].sum(axis=0).T
        field_counts_post = field_counts_df.loc[field_counts_df["date"].isin(set(date_boundaries["post"]))].\
                            groupby(aggs)[field_columns].sum(axis=0).T
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
            ## Individual Keyword Breakdown (Require Average of k Per Week)
            avg_per_day = 2 / 7
            keyword_plot_thresh = len(timestamp_counts["date"].unique()) * avg_per_day
            keywords_to_plot = keyword_counts.loc["keywords"].sum(axis=0).loc[keyword_counts.loc["keywords"].sum(axis=0) > keyword_plot_thresh].index.tolist()
            LOGGER.info("Visualizing Keyword Matches Over Time (Individual Breakdowns)")
            for keyword in tqdm(keywords_to_plot, desc="Keyword", file=sys.stdout):
                fig, ax = visualize_trends(keyword_counts.loc["keywords"],
                                           timestamp_counts,
                                           subset=[keyword],
                                           aggs=["date"],
                                           visualize=True,
                                           field_filter_sets={"indorg":["ind"]},
                                           timestamp_filter_sets={"indorg":["ind"]},
                                           group_normalize=True,
                                           dropna=True,
                                           smoothing_window=SMOOTHING_WINDOW)
                keyword_clean = keyword.replace(" ","-").replace(".","-").replace("/","-")
                fig.savefig("{}/keywords/timeseries/{}_keyword_proportion.png".format(PLOT_DIR, keyword_clean), dpi=300)
                plt.close(fig)
        ## Keyword Dynamics ## TODO
        if KEYWORD_DYNAMICS:
            LOGGER.info("Visualizing Keyword Dynamics")
            ## Isolate Keyword Counts
            for aggset, field_filter, time_filter, gnorm in AGGS:
                if "state" in aggset:
                    continue
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

#########################
### Execute
#########################

if __name__ == "__main__":
    _ = main()