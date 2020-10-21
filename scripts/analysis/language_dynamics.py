
#####################
### Configuration
#####################

## Location of Data
# PLATFORM = "reddit"
# DATA_DIR = "./data/processed/reddit/2017-2020/histories/"
# CACHE_DIR = "./data/processed/reddit/2017-2020/language_dynamics/"
# PLOT_DIR = "./plots/reddit/2017-2020/language_dynamics/"

PLATFORM = "twitter"
DATA_DIR = "./data/processed/twitter/2018-2020/timelines/"
CACHE_DIR = "./data/processed/twitter/2018-2020/language_dynamics/"
PLOT_DIR = "./plots/twitter/2018-2020/language_dynamics/"

## Analysis Parameters
NGRAMS = (1, 1)
RERUN_VOCAB = False
RERUN_COUNT = False
# REFERENCE_MODELS = {
#             "depression":"../mental-health/models/20201020193024-Falconet-SMHD-Depression/model.joblib",
#             "anxiety":"../mental-health/models/20201020193251-Falconet-SMHD-Anxiety/model.joblib",
# }
REFERENCE_MODELS = {
            "depression":"../mental-health/models/20201015121355-Falconet-Multitask-Depression/model.joblib",
            "anxiety":"../mental-health/models/20201015115714-Falconet-Multitask-Anxiety/model.joblib",
}


## Date Boundaries
DATE_START = "2019-01-01"
DATE_END = "2020-06-15"
CACHE_FREQ = "W-Mon"
COVID_START = "2020-03-19"

## Extra Parameters
NUM_PROCESSES = 8
RANDOM_STATE = 42
CHUNKSIZE = 30
FILTER_US = True
FILTER_IND = True

#####################
### Imports
#####################

## Standard Library
import os
import sys
import json
import gzip
from glob import glob
from datetime import datetime
from collections import Counter
from multiprocessing import Pool

## External Libraries
import joblib
import demoji
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, vstack
from adjustText import adjust_text
from sklearn.feature_extraction import DictVectorizer

## Mental Health Library
from mhlib.util.helpers import chunks
from mhlib.model.vocab import Vocabulary
from mhlib.model.feature_extractors import LIWCTransformer
from mhlib.util.logging import initialize_logger

#####################
### Globals
#####################

## Logger
LOGGER = initialize_logger()

## Demographic Files
DEMO_FILES = {
    "./data/processed/twitter/2013-2014/timelines/":"./data/processed/twitter/2013-2014/demographics.csv",
    "./data/processed/twitter/2016/timelines/":"./data/processed/twitter/2016/demographics.csv",
    "./data/processed/twitter/2018-2020/timelines/":"./data/processed/twitter/2018-2020/demographics.csv",
    "./data/processed/reddit/2017-2020/histories/":"./data/processed/reddit/2017-2020/geolocation.csv"
}

## Cache/Plot Directories
for d in [PLOT_DIR, CACHE_DIR]:
    if not os.path.exists(d):
        _ = os.makedirs(d)

#####################
### Helpers
#####################

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

def get_quadrant(x, y):
    """

    """
    if x > 0 and y > 0:
        return 1
    if x > 0 and y < 0:
        return 2
    if x < 0 and y < 0:
        return 3
    if x < 0 and y > 0:
        return 4
    return None

#####################
### Learn Vocabulary
#####################

## Processed Data Files
processed_files = sorted(glob(f"{DATA_DIR}*.json.gz"))

## Filter Files by Demographics
if PLATFORM == "twitter":
    demos = pd.read_csv(DEMO_FILES.get(DATA_DIR),index_col=0)
    if FILTER_US:
        demos = demos.loc[demos["country"] == "United States"]
    if FILTER_IND:
        demos = demos.loc[demos["indorg"] == "ind"]
    demo_ind = set(demos.index.map(os.path.abspath).str.replace("/raw/","/processed/"))
    processed_files = [f for f in processed_files if os.path.abspath(f) in demo_ind]
elif PLATFORM == "reddit":
    demos = pd.read_csv(DEMO_FILES.get(DATA_DIR),
                        usecols=list(range(7)),
                        index_col=0)
    if FILTER_US:
        demos = demos.loc[demos["country_argmax"]=="US"]
    demo_ind = set(demos.index.map(os.path.abspath).str.replace("/raw/","/processed/"))
    processed_files = [f for f in processed_files if os.path.abspath(f) in demo_ind]

## Learn/Load Vocabulary
vocab_cache_file = "{}vocab_ngram{}-{}_{}_{}.joblib".format(CACHE_DIR, NGRAMS[0], NGRAMS[1], DATE_START, DATE_END)

if not os.path.exists(vocab_cache_file) or RERUN_VOCAB:
    LOGGER.info("Learning Vocabulary")
    ## Learn Vocabulary
    vocab = Vocabulary(filter_negate=True,
                       filter_upper=True,
                       filter_numeric=True,
                       filter_punctuation=True,
                       filter_user_mentions=True,
                       filter_url=True,
                       filter_retweet=True,
                       filter_stopwords=False,
                       keep_pronouns=True,
                       preserve_case=False,
                       filter_hashtag=False,
                       strip_hashtag=False,
                       max_vocab_size=250000,
                       min_token_freq=10,
                       max_token_freq=None,
                       ngrams=NGRAMS,
                       binarize_counter=True,
                       filter_mh_subreddits=None,
                       filter_mh_terms=None,
                       keep_retweets=False,
                       external_vocab=[],
                       external_only=False)
    vocab = vocab.fit(processed_files,
                      chunksize=CHUNKSIZE,
                      jobs=8,
                      min_date=DATE_START,
                      max_date=DATE_END,
                      prune=True,
                      prune_freq=50)
    ## Cache
    _ = joblib.dump(vocab, vocab_cache_file)
else:
    LOGGER.info("Loading Vocabulary")
    vocab = joblib.load(vocab_cache_file)

## Initialize Vectorizer
LOGGER.info("Initializing Dict Vectorizer")
ngrams = vocab.get_ordered_vocabulary()
vocab.dvec = _initialize_dict_vectorizer(ngrams)

#####################
### Load Counts over Time
#####################

## Date Range 
LOGGER.info("Establishing Date Range")
date_range = list(pd.date_range(DATE_START, DATE_END, freq=CACHE_FREQ))
if pd.to_datetime(DATE_START) < date_range[0]:
    date_range = [pd.to_datetime(DATE_START)] + date_range
if pd.to_datetime(DATE_END) > date_range[-1]:
    date_range = date_range + [pd.to_datetime(DATE_END)]
date_range = [d.date() for d in date_range]

## Assignment Function
def assign_bin_index(created_utc):
    """

    """
    created_utc_dt = datetime.fromtimestamp(created_utc).date()
    if created_utc_dt < date_range[0]:
        return None
    if created_utc_dt >= date_range[-1]:
        return None
    for b, (bstart, bend) in enumerate(zip(date_range[:-1], date_range[1:])):
        if created_utc_dt >= bstart and created_utc_dt < bend:
            return b

## Counting Function
def count_language_usage(filename):
    """

    """
    file_data = vocab._loader.load_user_data(filename)
    counts = [Counter() for _ in range(len(date_range)-1)]
    time_bins = list(map(lambda f: assign_bin_index(f["created_utc"]), file_data))
    for tb, fd in zip(time_bins, file_data):
        if tb is None:
            continue
        counts[tb] += Counter(vocab.get_ngrams(fd["text_tokenized"], NGRAMS[0], NGRAMS[1]))
    counts_vec = vocab.dvec.transform(counts).toarray()
    return counts_vec

## Count Cache File
X_cache_file = "{}X_ngram{}-{}_{}-{}_{}.joblib".format(CACHE_DIR, NGRAMS[0], NGRAMS[1], DATE_START, DATE_END, CACHE_FREQ)

## Count/Load Language Usage
if not os.path.exists(X_cache_file) or RERUN_COUNT:
    ## Count Usage Over Time (In Chunks)
    LOGGER.info("Count Language Usage")
    X = np.zeros((len(date_range)-1, len(ngrams)))
    processed_file_chunks = list(chunks(processed_files, CHUNKSIZE))
    mp = Pool(NUM_PROCESSES)
    for fchunk in tqdm(processed_file_chunks, desc="File Chunk", file=sys.stdout, total=len(processed_file_chunks)):
        fchunk_x = list(mp.imap_unordered(count_language_usage, fchunk))
        fchunk_x = np.stack(fchunk_x).sum(axis=0)
        X += fchunk_x
    _ = mp.close()
    ## Cache
    _ = joblib.dump(X, X_cache_file)
else:
    ## Load From Cache
    LOGGER.info("Loading Cached Language Usage")
    X = joblib.load(X_cache_file)

#####################
### Log-ratio analysis
#####################

LOGGER.info("Starting Log-ratio Analysis")

## Smoothing
SMOOTHING = 0.01

## Indices
d_before = [i for i, d in enumerate(date_range[:-1]) if d < pd.to_datetime(COVID_START)]
d_after = [i for i, d in enumerate(date_range[:-1]) if d >= pd.to_datetime(COVID_START)]

## Separate and Aggregate
X_before = X[d_before].sum(axis=0)
X_after = X[d_after].sum(axis=0)

## Normalize Usage
X_before_norm = (X_before + SMOOTHING) / (X_before + SMOOTHING).sum()
X_after_norm = (X_after + SMOOTHING) / (X_after + SMOOTHING).sum()

## Compute Ratios
log_ratio = pd.Series(np.log10(X_after_norm / X_before_norm), index=ngrams)
log_ratio.index = log_ratio.index.map(lambda i: "_".join(i))
log_ratio.index = replace_emojis(log_ratio.index)

## Plot Ratios
ktop = 40
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
log_ratio.nlargest(ktop).sort_values(ascending=True).plot.barh(ax=ax[0])
log_ratio.nsmallest(ktop).sort_values(ascending=False).plot.barh(ax=ax[1])
for a in ax:
    a.set_xlabel("Log Ratio (Pre. vs. Post Pandemic)", fontweight="bold")
ax[0].set_title(f"{ktop} Largest Increases", fontweight="bold", fontstyle="italic", loc="left")
ax[1].set_title(f"{ktop} Largest Decreases", fontweight="bold", fontstyle="italic", loc="left")
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}top_ngram_log_ratio.png", dpi=300)
plt.close(fig)

#####################
### Model-aware Analysis
#####################

LOGGER.info("Starting Model-aware Analysis")

## Load Reference Models
LOGGER.info("Loading Reference Models")
REFERENCE_MODEL_OBJ = dict((x, joblib.load(y)) for x, y in REFERENCE_MODELS.items())

## Combine Reference Model Coefficients with Ratios
log_ratio_df = log_ratio.to_frame("ratio")
for condition, model in REFERENCE_MODEL_OBJ.items():
    model_features = list(map(lambda i: "_".join(i) if isinstance(i, tuple) else i, model.get_feature_names()))
    model_coefs = pd.Series(model.model.coef_[0], index=model_features)
    model_coefs.index = replace_emojis(model_coefs.index)
    model_coefs = model_coefs.loc[model_coefs.index.map(lambda i: not (i.startswith("GloVe_") or i.startswith("LIWC=") or i.startswith("LDA_TOPIC")))]
    log_ratio_df[condition] = model_coefs

## Coefficient/Ratio Map Plot Function
def plot_coefficient_ratio_map(log_ratio_df,
                               cond,
                               ratio_beta=0.5,
                               ktop=30):
    """

    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.6))
    pdf = log_ratio_df[[cond, "ratio"]].copy().dropna()
    pdf["outlier_score"] = pdf.applymap(abs).rank(axis=0).apply(lambda row: ratio_beta * row[cond] + (1-ratio_beta) * row["ratio"], axis=1)
    pdf["quadrant"] = pdf.apply(lambda row: get_quadrant(row[cond], row["ratio"]), axis=1)
    pdf.dropna(inplace=True)
    m = ax[0].scatter(pdf[cond], pdf["ratio"], c=pdf["outlier_score"], cmap=plt.cm.coolwarm, alpha=0.5, s=5)
    texts = []
    xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
    for quadrant, mult in zip(range(1, 5), [(1,1),(1,-1),(-1,-1),(-1,1)]):
        pdf_q = pdf.loc[pdf["quadrant"]==quadrant]
        pdf_q_largest = pdf_q.loc[pdf_q["outlier_score"].nlargest(ktop).index].rank(axis=0)
        for tok, row in pdf_q_largest.iterrows():
            x = row[cond]*mult[0]
            y = row["ratio"]*mult[1]
            ax[1].scatter(x, y, color="C0", alpha=0.05)
            texts.append(ax[1].text(x, y, tok, fontsize=6, ha="center", va="center"))
            if x < xmin:
                xmin = x - 10
            if x > xmax:
                xmax = x + 10
            if y < ymin:
                ymin = y - 10
            if y > ymax:
                ymax = y + 10
    adjust_text(texts, lim=10, arrowprops=dict(arrowstyle="-", color='C0', lw=0.5))
    ax[1].set_xlim(xmin, xmax)
    ax[1].set_ylim(ymin, ymax)
    for a in ax:
        a.axvline(0, color="black", linestyle="--", alpha=0.5, zorder=-1)
        a.axhline(0, color="black", linestyle="--", alpha=0.5, zorder=-1)
        a.set_xticks([])
        a.set_yticks([])
    ax[0].set_ylabel("Log Ratio (Pre vs. Post Pandemic)", fontweight="bold", labelpad=10)
    for a in ax:
        a.set_xlabel("Model Coefficient ({})".format(cond.title()), fontweight="bold", labelpad=10)
    ax[0].set_title("N-gram vs. Model Coefficient", loc="left", fontweight="bold", fontstyle="italic")
    ax[1].set_title("Largest Outliers", loc="left", fontweight="bold", fontstyle="italic")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    return fig, ax

## Create Ratio/Coefficient Maps
for cond in REFERENCE_MODEL_OBJ.keys():
    fig, ax = plot_coefficient_ratio_map(log_ratio_df, cond, 0.5, 30)
    fig.savefig(f"{PLOT_DIR}ngram_coefficient_map_{cond}.png", dpi=300)
    plt.close(fig)

#####################
### LIWC Analysis
#####################

LOGGER.info("Starting LIWC Analysis")

## Initialize Transformer
liwc = LIWCTransformer(vocab=vocab, norm="matched")

## LIWC Categories of Interest (Subset of Full)
mh_cats =  {'posemo': 'Positive Emotion',
            'negemo': 'Negative Emotion',
            'anx': 'Anxiety',
            'anger': 'Anger',
            'sad': 'Sadness',
            'family': 'Family',
            'friend': 'Friendship',
            'work': 'Work',
            'health': 'Health',
            'achieve': 'Achievement',
            'leisure': 'Leisure',
            'home': 'Home',
            'money': 'Money',
            'death': 'Death'}
pronoun_cats = {"ppron":"Personal Pronouns",
                "ipron":"Impersonal Pronouns",
                "i":"1st Person Singular",
                "we":"1st Person Plural",
                "you":"2nd Person",
                "shehe":"3rd Person Singular",
                "they":"3rd Person Plural"}

## Compute LIWC Matrices
X_liwc = liwc.fit_transform(X)
X_discrete_liwc = liwc.fit_transform(np.vstack([X_before, X_after]))

