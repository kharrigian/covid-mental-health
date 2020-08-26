
############################
### Configuration
############################

## Metadata
PLATFORM = "reddit"
CONDITION = "depression"

## Analysis Paramters
K_EXTREME = 10
DISPLAY_TOP = 5
RANK_BY_ABLATION = True

## Filtering Parameters
MIN_POSTS_PER_WINDOW = 10
MIN_TOKENS_PER_WINDOW = 25

## Groups (Target Probabilities to Showcase)
GROUPS = {
    "High (Pr(y=1)~1)":1,
    "Moderate (Pr(y=1)~0.7)":0.7,
    "Low (Pr(y=1)~0.3)":0.3,
    "No Evidence (Pr(y=1)~0)":0
}

############################
### Imports
############################

## Standard Library
import os
import sys
import gzip
import json
from glob import glob
from datetime import datetime
import textwrap

## External Libraries
import joblib
import numpy as np
import pandas as pd

## Mental Health
from mhlib.util.logging import initialize_logger

############################
### Globals
############################

## Initialize Logger
LOGGER = initialize_logger()

## Result Directories
RESULTS_DIRS = {
    "reddit":"./data/results/reddit/2017-2020/inference/monthly-weekly_step/",
    "twitter":"./data/results/twitter/2018-2020/inference/monthly-weekly_step/"
}
RESULTS_DIR = RESULTS_DIRS.get(PLATFORM)

## Models
MODEL_PATHS = {
    ("reddit","anxiety"):"../mental-health/models/falconet_v2/20200824135147-SMHD-Anxiety/model.joblib",
    ("reddit","depression"):"../mental-health/models/falconet_v2/20200824135305-SMHD-Depression/model.joblib",
    ("twitter","anxiety"):"../mental-health/models/falconet_v2/20200824135305-SMHD-Depression/model.joblib",
    ("twitter","depression"):"../mental-health/models/falconet_v2/20200824135027-Multitask-Anxiety/model.joblib"
}
LOGGER.info("Loading Model")
MODEL = joblib.load(MODEL_PATHS.get((PLATFORM,CONDITION)))

############################
### Helpers
############################

def load_processed_data(file,
                        min_date=None,
                        max_date=None):
    """

    """
    ## Get Raw Data (With Appropriate Tokens Used by Model)
    data = MODEL.vocab._loader.load_user_data(filename=file,
                                              min_date=min_date,
                                              max_date=max_date)
    ## Get Feature Representation
    tokens = [i["text_tokenized"] for i in data]
    token_counts = MODEL.vocab._count_tokens(tokens)
    X = MODEL._count2vec.transform(token_counts).toarray()[0]
    return data, X

def predict_posts(posts,
                  rank_by_ablation=False):
    """

    """
    token_counts = list(map(MODEL.vocab._count_tokens, [[i["text_tokenized"]] for i in posts]))
    X = MODEL._count2vec.transform(token_counts).toarray()
    X_T = MODEL.preprocessor.transform(X)
    y = MODEL.model.predict_proba(X_T)[:,1]
    if not rank_by_ablation:
        return y, None
    X_base = X.sum(axis=0,keepdims=True)
    X_T_base = MODEL.preprocessor.transform(X_base)
    X_T_ablation = MODEL.preprocessor.transform(X_base - X)
    y_base = MODEL.model.predict_proba(X_T_base)[:,1]
    y_ablation = MODEL.model.predict_proba(X_T_ablation)[:,1]
    return y, y_base - y_ablation

def asciihist(it,
              numbins=10,
              minmax=None,
              eps=0,
              str_tag='',
              scale_output=30):
    """
    Create an ASCII histogram from an interable of numbers.
    Source: https://gist.github.com/bgbg/608d9ef4fd75032731651257fe67fc81
    """
    bins = range(numbins)
    freq = {}.fromkeys(bins,0)
    itlist=list(it)
    #sort the list before binning it
    itlist.sort()
    if minmax:
        #discard values that are outside minmax range
        itmin=minmax[0]
        itmax=minmax[1]
        while itlist[0]<itmin: itlist.pop(0)
        while itlist[-1]>=itmax+eps: itlist.pop()
    else:
        #bin all values
        itmin=itlist[0]
        itmax=itlist[-1]
        eps=1
    cutoffs = [itmin]
    bin_increment = (itmax-itmin)/numbins
    #fill all but the last bin
    for bin in bins[:-1]:
        cutoff = itmin+(bin+1)*bin_increment
        cutoffs.append(cutoff)
        while itlist and itlist[0]<cutoff:
            freq[bin]=freq[bin]+1
            #discard the binned item
            itlist.pop(0)
    #the rest go in the last bin
    freq[bins[-1]]=len(itlist)
    if str_tag:
        str_tag = '%s '%str_tag
    if scale_output is not None:
        max_freq = max(freq.values())
        scaling_factor = float(scale_output)/float(max_freq)
        scaled_freq = {}
        for b in freq.keys():
            scaled_freq[b] = int(freq[b] * scaling_factor)
    else:
        scaled_freq = freq
    for bin in bins:
        LOGGER.info("%s%8.2f |%-5d | %s"%(str_tag,
                                 cutoffs[bin],
                                 freq[bin],
                                 "*"*scaled_freq[bin]))

def get_extremes(predictions,
                 date,
                 date_ranges,
                 groups,
                 k_extreme=10,
                 display_top=5,
                 rank_by_ablation=False):
    """

    """
    ## Alert User
    LOGGER.info("\n"+ "#"*100 + "\n### Analysis for Date Range: {} to {}\n".format(date_ranges[date][0], date_ranges[date][1]) + "#"*100 + "\n")
    ## Filter Predictions
    predictions_nn = predictions[date].dropna().copy()
    ## Identify Targets
    extremes = {}
    for gname, gtarget in groups.items():
        extremes[gname] = (predictions_nn - gtarget).map(abs).nsmallest(k_extreme).index.tolist()
    ## Load Targets
    extreme_data = {gname:{} for gname in groups.keys()}
    extreme_X ={gname:{} for gname in groups.keys()}
    for ext, ext_dict in extremes.items():
        for filename in ext_dict:
            file_data, file_X = load_processed_data(filename,
                                                    pd.to_datetime(date_ranges[date][0]),
                                                    pd.to_datetime(date_ranges[date][1]))
            extreme_data[ext][filename] = file_data
            extreme_X[ext][filename] = file_X
    ## Get Feature Representations and Predictions
    extreme_features = {}
    extreme_preds = {}
    pred_dict = {}
    for group, group_files in extremes.items():
        extreme_features[group] = MODEL.preprocessor.transform(np.vstack([extreme_X[group][f] for f in group_files]))
        extreme_preds[group] = MODEL.model.predict_proba(extreme_features[group])[:,1]
        pred_dict[group] = dict(zip(group_files, extreme_preds[group]))
    ## Compare Means (Validate Extremes)
    LOGGER.info("\n"+ "#"*50 + "\n### Overall Distribution\n" + "#"*50 + "\n")
    for group, group_preds in extreme_preds.items():
        LOGGER.info("Mean {}: {:.3f} (sig={:.3f})".format(group, group_preds.mean(), group_preds.std()))
    ## Make Post-Level Predictions
    extreme_post_predictions = {group:{} for group in groups}
    extreme_post_influences = {group:{} for group in groups}
    for ext, ext_dict in extreme_data.items():
        for file, posts in ext_dict.items():
            y, y_ablation = predict_posts(posts,
                                          rank_by_ablation=rank_by_ablation)
            extreme_post_predictions[ext][file] = y
            extreme_post_influences[ext][file] = y_ablation
    ## Plot Post-Level Distributions
    for g, (group, group_dict) in enumerate(extreme_post_predictions.items()):
        y_ext = np.hstack(list(group_dict.values()))
        LOGGER.info("\n"+ "#"*50 + "\n### Post Distribution: {}\n".format(group.title()) + "#"*50 + "\n")
        _ = asciihist(y_ext, numbins=10)
    ## See Top Ranked Posts
    sorter = extreme_post_predictions if not rank_by_ablation else extreme_post_influences
    for e, (group, group_dict) in enumerate(extreme_data.items()):
        LOGGER.info("\n"+ "#"*50 + "\n### Examples: {}\n".format(group.title()) + "#"*50)
        for file in group_dict.keys():
            LOGGER.info("\n~~~~~~~~~~~~ User: {} [Pr(y=1) = {:.4f}] ~~~~~~~~~~~~\n".format(os.path.basename(file).rstrip(".json.gz"),pred_dict[group][file]))
            if PLATFORM == "reddit":
                file_post_text = [("r/" + i["subreddit"], i["text"]) for i in extreme_data[group][file]]
            else:
                file_post_text = [("tweet", i["text"]) for i in extreme_data[group][file]]
            file_post_preds = sorter[group][file]
            file_examples = sorted(zip(file_post_text, file_post_preds), key=lambda x: x[1], reverse=True)
            for f, ex in enumerate(file_examples[:display_top]):
                pstr = "({}) [{:.5f}] -- {} -- {}".format(f+1, ex[1], ex[0][0], ex[0][1])
                LOGGER.info("\n\t\t".join(textwrap.wrap(pstr, 80)))

############################
### Load Predictions
############################

## Identify Prediction Files
pred_files = sorted(glob(f"{RESULTS_DIR}*/{CONDITION}.predictions.csv"))
pred_files = list(filter(lambda f: ".results/" not in f, pred_files))

## Load Predictions
LOGGER.info("Loading Predictions")
predictions = {}
support = {}
tokens = {}
unique_tokens = {}
date_ranges = {}
for pred_file in sorted(pred_files):
    start, stop = pred_file.split("/")[-2].split("_")
    if stop in predictions:
        continue
    date_ranges[stop] = (start, stop)
    pred_file_df = pd.read_csv(pred_file, index_col=0)
    predictions[stop] = pred_file_df["y_pred"].to_dict()
    support[stop] = pred_file_df["support"].to_dict()
    tokens[stop] = pred_file_df["matched_tokens"].to_dict()
    unique_tokens[stop] = pred_file_df["unique_matched_tokens"].to_dict()

## Format
LOGGER.info("Formatting Predictions")
predictions = pd.DataFrame(predictions)
support = pd.DataFrame(support)
tokens = pd.DataFrame(tokens)
unique_tokens = pd.DataFrame(unique_tokens)

## Date Filtering
LOGGER.info("Filtering Out Abnormal Dates")
dates = pd.to_datetime(predictions.columns)
date_diffs = [(y-x).days for x, y in zip(dates[:-1],dates[1:])]
dates_drop = [d.date().isoformat() for d, dd in zip(dates[:-1], date_diffs) if dd != np.median(date_diffs)]
for df in [predictions, support, tokens, unique_tokens]:
    df.drop(dates_drop, axis=1, inplace=True)
dates = pd.to_datetime(predictions.columns)

## Filter Data
LOGGER.info("Applying Activity Thresholds for Filtering")
predictions_filtered = predictions.copy()
support_filtered = support.copy()
tokens_filtered = tokens.copy()
unique_tokens_filtered = unique_tokens.copy()
for thresh, df in zip([MIN_POSTS_PER_WINDOW, MIN_TOKENS_PER_WINDOW],
                      [support, tokens]):
    mask = df.copy()
    for col in mask.columns:
        mask.loc[mask[col] < MIN_POSTS_PER_WINDOW, col] = np.nan
        mask.loc[mask[col] >= MIN_POSTS_PER_WINDOW, col] = 1
    predictions_filtered = predictions_filtered * mask
    support_filtered = support_filtered * mask
    tokens_filtered = tokens_filtered * mask
    unique_tokens_filtered = unique_tokens_filtered * mask

############################
### Identify Extreme Candidates
############################

## Choose Dates
analysis_dates = ["2019-01-28",'2019-04-15','2019-12-30','2020-04-20','2020-06-15']

## Get Extremes
for ad in analysis_dates:
    _ = get_extremes(predictions=predictions_filtered,
                     date=ad,
                     date_ranges=date_ranges,
                     groups=GROUPS,
                     k_extreme=K_EXTREME,
                     display_top=DISPLAY_TOP,
                     rank_by_ablation=RANK_BY_ABLATION)