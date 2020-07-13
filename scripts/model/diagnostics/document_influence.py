

########################
### Imports
########################

## Standard Library
import os
import sys
import json
import gzip
import argparse
from datetime import datetime

## External Libraries
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mhlib.util.logging import initialize_logger
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer

########################
### Globals
########################

## Create Logger
LOGGER = initialize_logger()

## Load COVID Resources
COVID_TERM_FILE = "./data/resources/covid_terms.json"
COVID_SUBREDDIT_FILE = "./data/resources/covid_subreddits.json"
with open(COVID_TERM_FILE,"r") as the_file:
    COVID_TERMS = json.load(the_file)
with open(COVID_SUBREDDIT_FILE,"r") as the_file:
    COVID_SUBREDDITS = json.load(the_file)

########################
### Functions
########################

def parse_arguments():
    """
    Parse command-line to identify configuration filepath.

    Args:
        None
    
    Returns:
        args (argparse Object): Command-line argument holder.
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Estimate document-level influence on user-prediction")
    ## Generic Arguments
    parser.add_argument("model",
                        type=str,
                        help="Path to cached model .joblib file")
    parser.add_argument("input_file",
                        type=str,
                        help="Path to processed file")
    parser.add_argument("output_dir",
                        type=str,
                        help="Path to output directory")
    parser.add_argument("--min_date",
                        type=str,
                        default=None,
                        help="Lower date boundary")
    parser.add_argument("--max_date",
                        type=str,
                        default=None,
                        help="Upper date boundary")
    parser.add_argument("--keep_covid",
                        default=False,
                        action="store_true")
    parser.add_argument("--sample_rate",
                        type=float,
                        default=1,
                        help="Document Sample Frequency")
    parser.add_argument("--random_state",
                        type=int,
                        default=42)
    ## Parse Arguments
    args = parser.parse_args()
    return args

def load_documents(input_file,
                   min_date=None,
                   max_date=None,
                   sample_rate=1):
    """

    """
    ## Load Data
    data = []
    with gzip.open(input_file,"r") as the_file:
        try:
            data = json.load(the_file)
        except:
            for line in the_file:
                data.append(json.loads(line))
    ## Sort By Time (Oldest to Newest)
    data = sorted(data, key=lambda d: d["created_utc"])
    ## Date Filter
    if min_date is not None:
        min_date = pd.to_datetime(min_date).timestamp()
    elif min_date is None:
        min_date = data[0]["created_utc"]
    if max_date is not None:
        max_date = pd.to_datetime(max_date).timestamp()
    elif max_date is None:
        max_date = data[-1]["created_utc"]
    ## Time Filter
    data = list(filter(lambda d: d["created_utc"]>=min_date and d["created_utc"]<=max_date, data))
    ## Sample
    if sample_rate < 1:
        data = [d for d in data if np.random.uniform() < sample_rate]
    return data

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

def load_bow(model,
             documents):
    """

    """
    ## Filtering
    documents = model.vocab._loader.filter_user_data(documents)
    ## Counts
    token_counts = []
    for d, doc in enumerate(documents):
        token_counts.append(model.vocab._count_tokens([doc["text_tokenized"]]))
    ## Vectorize
    vec = model._count2vec.transform(token_counts).toarray()
    ## Metadata
    meta = []
    vocabulary = model._count2vec.feature_names_
    for d, v in zip(documents, vec):
        d_meta = {}
        for col in ["created_utc","text","subreddit"]:
            if col in d:
                d_meta[col] = d[col]
        d_toks = [vocabulary[i] for i in np.nonzero(v)[0]]
        d_meta["tokens"] = d_toks
        meta.append(d_meta)
    ## Drop Null
    nn_mask = np.nonzero((vec!=0).any(axis=1))[0]
    vec = vec[nn_mask]
    meta = [meta[n] for n in nn_mask]
    return meta, vec

def create_leave_one_out_X(X):
    """

    """
    X_loo = []
    X_sum = X.sum(axis=0)
    for r, row in enumerate(X):
        X_loo.append(X_sum - row)
    X_loo = np.vstack(X_loo)
    return X_loo

def main():
    """

    """
    ## Parse Arguments
    LOGGER.info("Parsing Command-line Arguments")
    # args = parse_arguments()
    args = Args()
    ## Get Subject
    subject = os.path.basename(args.input_file).split(".")[0]
    ## Output Directory
    if not os.path.exists(args.output_dir):
        _ = os.makedirs(args.output_dir)
    ## Load Model
    LOGGER.info("Loading Mental Health Classifier")
    model = joblib.load(args.model)
    ## Update Model Filtering
    if not args.keep_covid:
        LOGGER.info("Updating Subreddit + Term Filter Set")
        model = update_filter_set(model, set(COVID_SUBREDDITS["covid"]), set(COVID_TERMS["covid"]))
    ## Set Seed
    np.random.seed(args.random_state)
    ## Load Documents
    LOGGER.info("Loading Documents")
    documents = load_documents(args.input_file,
                               min_date=args.min_date,
                               max_date=args.max_date,
                               sample_rate=args.sample_rate)
    ## Load Bag of Words Representations (Filtered)
    LOGGER.info("Creating Bag-of-Words Representation")
    documents, X = load_bow(model, documents)
    ## Length Check
    if len(documents) == 0:
        LOGGER.info("No matched vocab! Exiting early.")
    ## Create LOO Bag of Words Representation
    LOGGER.info("Creating Leave-one-out Matrix")
    X_loo = create_leave_one_out_X(X)
    X_reg = X.sum(axis=0).reshape(1,-1)
    ## Transform
    LOGGER.info("Transforming Bag-of-Words into Full Feature Set")
    X_loo_T = model.preprocessor.transform(X_loo)
    X_reg_T = model.preprocessor.transform(X_reg)
    ## Make Probability Prediction
    LOGGER.info("Computing Disorder Probabilities")
    y_pred_loo = model.model.predict_proba(X_loo_T)[:,1]
    y_pred_reg = model.model.predict_proba(X_reg_T)[:,1]
    ## Histogram
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.hist(y_pred_loo, color="C0", alpha=0.5, bins=40, edgecolor="navy")
    ax.set_xlabel("Predicted Probability", fontweight="bold")
    ax.set_ylabel("Leave One Out Samples", fontweight="bold")
    ax.axvline(y_pred_reg[0], color = "darkred", linestyle="--", alpha=0.9, linewidth=3,
               label="All Documents: Pr(Disorder) = {:.4f}".format(y_pred_reg[0]))
    ax.set_xlim(y_pred_loo.min()-0.025, y_pred_loo.max()+0.025)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    plt.savefig(f"{args.output_dir}{subject}.loo_probability_histogram.png",dpi=300)
    plt.close(fig)
    ## Compute Document Influence (Positive indicates presence of document increases disorder probability)
    LOGGER.info("Computing Document Influence (Baseline Pr(Disorder) = {:.4f})".format(y_pred_reg[0]))
    doc_influence = y_pred_reg - y_pred_loo
    for doc, doc_infl, doc_score in zip(documents, doc_influence, y_pred_loo):
        doc["document_influence"] = doc_infl
    ## Compute Feature Differences
    LOGGER.info("Computing Feature Differences")
    ## Regress Feature Differences on Document Influence
    LOGGER.info("Regressing Document Influence on Document Terms")
    X_tfidf = TfidfTransformer().fit_transform(X)
    doc_influence_scaled = (doc_influence - doc_influence.mean()) / doc_influence.std()
    if np.any(np.isnan(doc_influence_scaled)):
        LOGGER.info("No variation in document influence. Exiting")
        exit()
    infl_model = Ridge()
    infl_model = infl_model.fit(X_tfidf, doc_influence_scaled)
    ## Extract Feature Influence
    LOGGER.info("Extracting Feature Influence")
    full_feature_idx = dict((f, i) for i, f in enumerate(model.get_feature_names()))
    feature_influence = pd.Series(index=model._count2vec.feature_names_,
                                  data=infl_model.coef_).sort_values()
    ## Visualize Feature Influence
    LOGGER.info("Visualizing Feature Influence")
    fig, ax = plt.subplots(figsize=(10,5.8))
    plot_vals = feature_influence.head(20).append(feature_influence.tail(20)).drop_duplicates()
    plot_vals = plot_vals.loc[plot_vals != 0]
    plot_val_coef = [model.model.coef_[0][full_feature_idx[i]] for i in plot_vals.index]
    yind = list(range(len(plot_vals)))
    ax.barh(yind,
            plot_vals.values,
            color = list(map(lambda i: "darkred" if i <= 0 else "navy", plot_val_coef)),
            alpha = 0.5)
    ax.set_yticks(yind)
    ax.set_yticklabels(plot_vals.index.tolist())
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Fitted Effect on Document Influence", fontweight="bold")
    fig.tight_layout()
    plt.savefig(f"{args.output_dir}{subject}.document_influence.png",dpi=300)
    plt.close(fig)
    ## Sort Documents by Influence
    LOGGER.info("Sorting Documents by Influence")
    documents = sorted(documents, key = lambda x: x["document_influence"], reverse=True)
    ## Output Documents With Highest Influence
    LOGGER.info("~"*50 + "\nTop 5 Most Influential Documents (Positive)\n" + "~"*50)
    for d, doc in enumerate(documents[:5]):
        LOGGER.info("{}) [{:e}] {}".format(d+1, doc.get("document_influence"), doc.get("text")))
    LOGGER.info("~"*50 + "\nTop 5 Most Influential Documents (Negative)\n" + "~"*50)
    for d, doc in enumerate(documents[-5:][::-1]):
        LOGGER.info("{}) [{:e}] {}".format(d+1, doc.get("document_influence"), doc.get("text")))


#####################
### Execute
#####################

if __name__ == "__main__":
    _ = main()