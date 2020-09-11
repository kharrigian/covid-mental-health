
#######################
### Imports
#######################

## Standard Library
import os
import sys
import json
import gzip
import argparse
from glob import glob
from datetime import datetime
from multiprocessing import Pool

## External Library
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import vstack, csr_matrix

## Local
from mhlib.util.helpers import chunks
from mhlib.util.logging import initialize_logger
from mhlib.preprocess.tokenizer import STOPWORDS

#######################
### Globals
#######################

## Logging
LOGGER = initialize_logger()

## Load COVID Resources
COVID_TERM_FILE = "./data/resources/covid_terms.json"
COVID_SUBREDDIT_FILE = "./data/resources/covid_subreddits.json"
with open(COVID_TERM_FILE,"r") as the_file:
    COVID_TERMS = json.load(the_file)
with open(COVID_SUBREDDIT_FILE,"r") as the_file:
    COVID_SUBREDDITS = json.load(the_file)

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
    parser = argparse.ArgumentParser(description="Infer mental health status within preprocessed files")
    ## Generic Arguments
    parser.add_argument("model",
                        type=str,
                        help="Path to cached model file (.joblib)")
    parser.add_argument("--input",
                        type=str,
                        default=None,
                        help="Path to input folder of processed *.gz files or a single processed *.gz file")
    parser.add_argument("--output_folder",
                        type=str,
                        default=None,
                        help="Name of output folder for placing predictions")
    parser.add_argument("--min_date",
                        type=str,
                        default="2000-01-01",
                        help="Lower date boundary (isoformat str) if desired")
    parser.add_argument("--max_date",
                        type=str,
                        default=None,
                        help="upper date boundary (isoformat str) if desired")
    parser.add_argument("--n_samples",
                        type=int,
                        default=None,
                        help="Number of post samples to isolate for modeling if desired")
    parser.add_argument("--randomized",
                        action="store_true",
                        default=False,
                        help="If included along with samples, will use randomized selection instead of recent")
    parser.add_argument("--analyze_features",
                        action="store_true",
                        default=False,
                        help="If included, analyze feature effects")
    parser.add_argument("--bootstrap_samples",
                        type=int,
                        default=100,
                        help="Number of samples to use for estimating feature support range")
    parser.add_argument("--bootstrap_sample_percent",
                        type=float,
                        default=70,
                        help="Sample Percentage (0,100] to use for estimating feature support range")
    parser.add_argument("--keep_missing",
                        action="store_true",
                        default=False,
                        help="If included, will make predictions for users without any features")
    parser.add_argument("--keep_covid",
                        action="store_true",
                        default=False,
                        help="If included, will not filter out COVID-related posts or subreddits")
    parser.add_argument("--chunksize",
                        default=None,
                        type=int,
                        help="Number of files to make inferences for at a time.")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Could not find model file {args.model}")
    if args.input is None:
        raise ValueError("Must provide --input folder or .gz file")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Could not find input filepath {args.input}")
    if args.output_folder is None:
        raise ValueError("Must provide an --output_folder argument")
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    return args

def get_file_list(args):
    """

    """
    if os.path.isfile(args.input):
        return [args.input]
    elif os.path.isdir(args.input):
        return glob(f"{args.input}*.gz")
    else:
        raise ValueError("Did not recognize command line --input")

def get_date_bounds(args):
    """

    """
    min_date = pd.to_datetime(args.min_date)
    if args.max_date is None:
        max_date = pd.to_datetime(datetime.now())
    else:
        max_date = pd.to_datetime(args.max_date)
    return min_date, max_date

def update_filter_set(model,
                      filter_subreddits=None,
                      filter_terms=None):
    """

    """
    ## Add to Filter Terms
    if filter_terms is not None:
        model.vocab._loader._ignore_terms.update(filter_terms)
    ## Add to Filter Subreddits
    if filter_subreddits is not None:
        model.vocab._loader._ignore_subreddits.update(filter_subreddits)
    return model

def predict_and_interpret(filenames,
                          model,
                          min_date=None,
                          max_date=None,
                          n_samples=None,
                          randomized=False,
                          interpret=False,
                          bootstrap_samples=100,
                          bootstrap_sample_percent=30,
                          ignore_missing=True,
                          chunksize=None):
    """

    """
    ## Date Boundaries
    if min_date is not None and isinstance(min_date,str):
        min_date=pd.to_datetime(min_date)
    if max_date is not None and isinstance(max_date,str):
        max_date=pd.to_datetime(max_date)
    ## Get Chunks
    if chunksize is None:
        chunksize = len(filenames)
    filechunks = list(chunks(filenames, chunksize))
    ## Initialize Cache
    X_test = []
    support = []
    y_pred = {}
    n = {}
    tn = {}
    tn_binary = {}
    filtered_filenames = []
    ## Cycle Through Chunks
    for j, file_chunk in enumerate(filechunks):
        LOGGER.info("[Beginning to Process File Chunk {}/{}]".format(j+1, len(filechunks)))
        ## Vectorize the data
        LOGGER.info("Vectorizing Test Files")
        chunk_files, X_chunk, _, n_ = model._load_vectors(file_chunk,
                                                          None,
                                                          min_date=min_date,
                                                          max_date=max_date,
                                                          n_samples=n_samples, 
                                                          randomized=randomized,
                                                          return_post_counts=True)
        ## Ignore Users without any features
        if ignore_missing:
            LOGGER.info("Filtering Out Users Without Any Recognized Terms")
            missing_mask = np.nonzero(X_chunk.sum(axis=1)>0)[0]
            chunk_files = [chunk_files[m] for m in missing_mask]
            X_chunk = X_chunk[missing_mask]
            n_ = n_[missing_mask]
        n_ = dict((zip(chunk_files, n_)))
        ## Count Tokens
        tn_ = dict((filename, count) for filename, count in zip(chunk_files, X_chunk.sum(axis=1)))
        tn_binary_ = dict((filename, count) for filename, count in zip(chunk_files, (X_chunk>0).sum(axis=1)))
        ## Apply Any Additional Preprocessing
        LOGGER.info("Generating Feature Set")
        X_chunk = model.preprocessor.transform(X_chunk)    
        ## Feed Forward
        LOGGER.info("Computing Logits")
        support_ = np.multiply(X_chunk, model.model.coef_)
        logits = support_.sum(axis=1) + model.model.intercept_
        ## Get Predictions
        LOGGER.info("Computing Probabilities")
        p = dict(zip(chunk_files, 1 / (1 + np.exp(-logits))))
        ## Cache Results
        if interpret:
            X_test.append(X_chunk)
            support.append(support_)
        y_pred.update(p)
        n.update(n_)
        tn.update(tn_)
        tn_binary.update(tn_binary_)
        filtered_filenames.extend(chunk_files)
    ## Format Cache
    n = np.array([n[filename] for filename in filtered_filenames])
    tn = np.array([tn[filename] for filename in filtered_filenames])
    tn_binary = np.array([tn_binary[filename] for filename in filtered_filenames])
    ## Interpretation
    feature_range = None
    if interpret:
        ## Concatenate Features and Support
        if isinstance(X_test[0], csr_matrix):
            X_test = vstack(X_test).toarray()
        else:
            X_test = np.vstack(X_test)
        if isinstance(support[0], csr_matrix):
            support = vstack(support)
        else:
            support = np.vstack(support)
        ## Get Features
        feature_names = model.get_feature_names()
        ## Get Feature Range (Bootstrap used for Confidence Intervals)
        sample_size = int(X_test.shape[0] * bootstrap_sample_percent / 100)
        feature_range = []
        for _ in tqdm(list(range(bootstrap_samples)), desc="Bootstrap Feature Samples", file=sys.stdout):
            sind = np.random.choice(X_test.shape[0], size=sample_size, replace=True)
            feature_range.append(support[sind].mean(axis=0))
        feature_range = np.percentile(np.vstack(feature_range), [2.5, 50, 97.5], axis=0)
        feature_range = pd.DataFrame(feature_range.T,
                                     index=feature_names,
                                     columns=["lower","median","upper"])
    return y_pred, feature_range, n, tn, tn_binary

def plot_feature_range(feature_range,
                       condition,
                       k_top=20,
                       language_only=True):
    """

    """
    ## Sort Features
    feature_range = feature_range.sort_values("median", ascending=False)
    ## Isolate Language
    if language_only:
        feature_range = feature_range.loc[feature_range.index.map(lambda i: isinstance(i, tuple))]
    ## Ignore Stopwords
    feature_range = feature_range.loc[feature_range.index.map(lambda i: i[0].lower() not in STOPWORDS if isinstance(i, tuple) else False)]
    ## Identify Feature Extremes
    top = feature_range.head(k_top).append(feature_range.tail(k_top)).iloc[::-1]
    ## Feature Format
    feature_formatter = lambda i: "_".join(i) if isinstance(i, tuple) else i
    top.index = top.index.map(feature_formatter)
    ## Create Figure
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.barh(list(range(len(top))),
            top["upper"]-top["lower"],
            left=top["lower"],
            alpha=0.5,
            color=["darkred"]*k_top+["navy"]*k_top,
    )
    ax.scatter(top["median"],
               list(range(len(top))),
               color=["darkred"]*k_top+["navy"]*k_top,
               alpha=.9,
               zorder=10)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5, zorder=-1)
    ax.set_yticks(list(range(len(top))))
    ax.set_yticklabels(top.index.tolist(), fontsize=10)
    ax.set_ylim(-.5,len(top)-.5)
    ax.set_xlabel("Mean Feature Responsibility\n(Impact on Positive Prediction)", fontweight="bold", fontsize=14)
    ax.set_title(f"Condition: {condition.title()}", fontweight="bold", fontsize=16, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig, ax

def main():
    """

    """
    ## Parse Command-line Arguments
    args = parse_arguments()
    ## Identify Filenames
    filenames = get_file_list(args)
    LOGGER.info("Found {} Files".format(len(filenames)))
    ## Load Model
    LOGGER.info(f"Loading Model: {args.model}")
    model = joblib.load(args.model)
    ## Append to Filter Set
    if args.keep_covid:
        filter_terms = None
        filter_subreddits = None
    else:
        filter_terms = COVID_TERMS["covid"]
        filter_subreddits = COVID_SUBREDDITS["covid"]
    model = update_filter_set(model,
                              filter_terms=filter_terms,
                              filter_subreddits=filter_subreddits)
    ## Get Date Boundaries
    LOGGER.info(f"Parsing Date Boundaries")
    min_date, max_date = get_date_bounds(args)
    if max_date < min_date:
        raise ValueError("Maximum Date in arguments occurs before minimum date!")
    ## Make Predictions
    y_pred, feature_range, n, tn, tn_binary = predict_and_interpret(filenames,
                                                                    model,
                                                                    min_date=min_date,
                                                                    max_date=max_date,
                                                                    n_samples=args.n_samples,
                                                                    randomized=args.randomized,
                                                                    interpret=args.analyze_features,
                                                                    bootstrap_samples=args.bootstrap_samples,
                                                                    bootstrap_sample_percent=args.bootstrap_sample_percent,
                                                                    ignore_missing=not args.keep_missing,
                                                                    chunksize=args.chunksize)
    ## Combine Predictions and Support
    y_pred = pd.DataFrame(pd.Series(y_pred),columns=["y_pred"])
    y_pred["support"] = n
    y_pred["matched_tokens"] = tn
    y_pred["unique_matched_tokens"] = tn_binary
    y_pred = y_pred.reset_index().rename(columns={"index":"filename"})
    ## Cache Predictions
    pred_file = f"{args.output_folder}{model._target_disorder}.predictions.csv"
    support_file = f"{args.output_folder}{model._target_disorder}.feature_responsibility.csv"
    LOGGER.info(f"Caching Predictions at: {pred_file}")
    y_pred.to_csv(pred_file, index=False) 
    ## Cache Feature Range
    if feature_range is not None:
        LOGGER.info(f"Caching Feature Reponsibilities at : {support_file}")
        feature_range.to_csv(support_file)
        ## Plot Feature Range (Top Values)
        LOGGER.info("Visualizing Feature Reponsibilities")
        fig, ax = plot_feature_range(feature_range,
                                    condition=model._target_disorder,
                                    k_top=20)
        fig.savefig(f"{args.output_folder}{model._target_disorder}.feature_responsibility.png", dpi=300)
        plt.close(fig)
    LOGGER.info("Script Complete!")


#######################
### Execute
#######################

if __name__ == "__main__":
    _ = main()