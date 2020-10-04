
"""
Robustness Metrics:
- Number of Matches (Overall)
- Match Rate (Overall)
- Daily Match Rate
    * Mean
    * Median
    * Min
    * Max
    * 25th Percentile
    * 75th Percentile
- Percentage of Matches with COVID Terms
- Neighbor N-Grams (Pre-COVID)
- Neighbor N-Grams (Post-COVID)
"""

#######################
### Configuration
#######################

## Path to Falconet Output
DATA_DIR = "./data/results/twitter/2018-2020/falconet-2020/"

## Path to Cache Directory/Plot Directory
CACHE_DIR = "./data/results/twitter/2018-2020/keywords/robustness/"
PLOT_DIR = "./plots/twitter/2018-2020/keywords/robustness/"

## Context Analysis Parameters
CONTEXT_WINDOW = None
NGRAMS=(1,1)
MIN_FREQ=30
MIN_CONTEXT_FREQ=10
SMOOTHING = 0.01
PRE_COVID_WINDOW = ["2019-03-19","2019-08-01"]
POST_COVID_WINDOW = ["2020-03-19","2020-08-01"]

## Meta Parameters
NUM_JOBS = 8

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
from glob import glob
from datetime import datetime
from dateutil.parser import parse
from collections import Counter
from multiprocessing import Pool
from functools import partial
import textwrap

## External Libraries
import demoji
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

## Tokenizer
TOKENIZER = Tokenizer(keep_case=False,
                      negate_handling=False,
                      negate_token=False,
                      keep_punctuation=False,
                      keep_numbers=False,
                      expand_contractions=True,
                      keep_user_mentions=False,
                      keep_pronouns=True,
                      keep_url=False,
                      keep_hashtags=True,
                      keep_retweets=True,
                      emoji_handling=None,
                      strip_hashtag=False)

## Regex Rules
SPECIAL = "“”…‘’´"
INCLUDE_HASHTAGS = True
INCLUDE_MENTIONS = False

## Coronavirus Keywords
coronavirus_keyword_file = "./data/resources/falconet/corona_virus.keywords"
coronavirus_keywords = list(map(lambda i: i.strip(), open(coronavirus_keyword_file,"r").readlines()))
coronavirus_keywords = sorted(set(flatten([[i, i.lower()] for i in coronavirus_keywords])))

## Mental Health Keywords
mental_health_keywords = {}
for mhlist, mhfile in [("Crisis (Level 1)", "crisis_level1.keywords"),
                       ("Crisis (Level 2)", "crisis_level2.keywords"),
                       ("Crisis (Level 3)", "crisis_level3.keywords"),
                       ("SMHD", "smhd.keywords"),
                       ("CLSP", "pmi.keywords")]:
    mhkeys = list(map(lambda i: i.strip(), open(f"./data/resources/falconet/{mhfile}","r").readlines()))
    mhkeys = sorted(set(flatten([[i, i.lower()] for i in mhkeys])))
    mental_health_keywords[mhlist] = mhkeys


## Reverse Mental Health Keyword List
mental_health_keywords_reverse = dict()
for mhlist, terms in mental_health_keywords.items():
    for t in terms:
        if t not in mental_health_keywords_reverse:
            mental_health_keywords_reverse[t] = []
        mental_health_keywords_reverse[t].append(mhlist)


#######################
### Helpers
#######################

def load_file(filename):
    """

    """
    data = []
    with gzip.open(filename,"r") as the_file:
        for line in the_file:
            data.append(json.loads(line))
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

def count_keywords_in_file(filename):
    """

    """
    ## Storage
    n_posts = Counter()
    keywords_by_date = dict()
    ## Load File
    f_data = load_file(filename)
    ## Count Keywords in Posts
    for post in f_data:
        ## Count Date
        post_date = parse(post.get("date")).date()
        n_posts[post_date] += 1
        ## Count Keywords
        post_keywords = post.get("keywords")
        if post_keywords:
            post_keyword_counts = Counter(post_keywords)
            if post_date not in keywords_by_date:
                keywords_by_date[post_date] = []
            keywords_by_date[post_date].append(post_keyword_counts)
    ## Sum Keywords
    for date, keyword_list in keywords_by_date.items():
        keywords_by_date[date] = sum(keyword_list, Counter())
    return n_posts, keywords_by_date

def keyword_comorbidity(filename):
    """

    """
    ## Storage
    comorbidity = dict()
    morbidity = Counter()
    ## Load File
    f_data = load_file(filename)
    ## Filter No Keywords
    f_data = list(filter(lambda p: p.get("keywords"), f_data))
    ## Get Comorbidity
    for post in f_data:
        post_keywords = post.get("keywords")
        for i, p in enumerate(post_keywords):
            morbidity[p] += 1
            if p not in comorbidity:
                comorbidity[p] = Counter()
            for j, pj in enumerate(post_keywords):
                if p != pj:
                    comorbidity[p][pj] += 1
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

def get_context(filename,
                ngrams=(1,1),
                window=None,
                include_mentions=False,
                min_date=None,
                max_date=None):
    """

    """
    ## Load File Data
    data = load_file(filename)
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
                    max_date=None):
    """

    """
    ## Load File Data
    data = load_file(filename)
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

## Get Processed Files
filenames = sorted(glob(f"{DATA_DIR}*_minimal.json.gz"))

## Get Keyword Counts Over Time (Parallel Processing)
mp = Pool(NUM_JOBS)
mp_results = list(tqdm(mp.imap_unordered(count_keywords_in_file, filenames),
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

## Rolling Statistics
window_size = 14
rolling_keyword_counts = keyword_counts.rolling(window_size,axis=0).sum().iloc[window_size:]
rolling_n_posts = n_posts["count"].rolling(window_size).sum().iloc[window_size:]
rolling_keyword_counts_normed = rolling_keyword_counts.apply(lambda x: x / rolling_n_posts,axis=0)

## Compute Summary Statistics
overall_num_matches = keyword_counts.sum(axis=0)
overall_match_rate = overall_num_matches / n_posts["count"].sum()
max_match_rate = rolling_keyword_counts_normed.max(axis=0)
min_match_rate = rolling_keyword_counts_normed.min(axis=0)
median_match_rate = rolling_keyword_counts_normed.median(axis=0)
lower_match_rate = rolling_keyword_counts_normed.apply(lambda x: np.nanpercentile(x, 25), axis=0)
upper_match_rate = rolling_keyword_counts_normed.apply(lambda x: np.nanpercentile(x, 75), axis=0)
cv_match_rate = rolling_keyword_counts_normed.std(axis=0) / rolling_keyword_counts_normed.mean(axis=0)

#######################
### Representative Posts
#######################

def get_posts(filename,
              keywords):
    """

    """
    data = load_file(filename)
    data = list(filter(lambda i: i.get("keywords"), data))
    all_matches = []
    for k in keywords:
        k_data = list(filter(lambda i: k in set(i.get("keywords")), data))
        k_data = [{"keyword":k, "tweet_id":d.get("tweet_id"),"user_id":d.get("user_id"),"date":d.get("date"),"text":d.get("text")} for d in k_data]
        all_matches.extend(k_data)
    return all_matches

def find_keyword_examples(filenames,
                          keywords,
                          n=10):
    """

    """
    ## Find Matches
    mp = Pool(NUM_JOBS)
    helper = partial(get_posts, keywords=keywords)
    mp_matches = list(tqdm(mp.imap_unordered(helper, filenames),
                        total=len(filenames),
                        leave=False,
                        file=sys.stdout))
    mp.close()
    ## Sample
    mp_matches = pd.DataFrame(flatten(mp_matches))
    mp_matches["date"] = mp_matches["date"].map(lambda x: parse(x).date())
    mp_matches.sort_values("date", inplace=True)
    sample = []
    for keyword in keywords:
        mp_keyword_matches = mp_matches.loc[mp_matches["keyword"]==keyword]
        mp_keyword_matches = mp_keyword_matches.drop_duplicates("text")
        mp_keyword_matches = mp_keyword_matches.sample(min(n, len(mp_keyword_matches)), random_state=42, replace=False).sort_values("date")
        sample.append(mp_keyword_matches)
    sample = pd.concat(sample).reset_index(drop=True)
    return sample

## Find Representative Posts
representative_examples = []
keyword_chunks = list(chunks(keyword_counts.columns.tolist(), 20))
for keyword_chunk in tqdm(keyword_chunks,position=0,desc="Keyword Chunk",file=sys.stdout):
    keyword_examples = find_keyword_examples(filenames, keyword_chunk, n=20)
    representative_examples.append(keyword_examples)
representative_examples = pd.concat(representative_examples).reset_index(drop=True)

#######################
### Keyword Co-morbidity
#######################

## Get Keyword Comorbidity
mp = Pool(NUM_JOBS)
mp_results = list(tqdm(mp.imap_unordered(keyword_comorbidity, filenames),
                       total=len(filenames),
                       file=sys.stdout))
mp.close()

## Format into Matrix
comorbidity = np.zeros((keyword_counts.shape[1], keyword_counts.shape[1]))
keyword2ind = dict(zip(keyword_counts.columns.tolist(), range(keyword_counts.shape[1])))
for kc, kc_co in mp_results:
    for _k, _c in kc.items():
        comorbidity[keyword2ind[_k], keyword2ind[_k]] += _c
    for _k, _k_co in kc_co.items():
        for _kc, _co in _k_co.items():
            comorbidity[keyword2ind[_k], keyword2ind[_kc]] += _co
comorbidity = comorbidity.astype(int)

## Normalize
comorbidity_normalized = comorbidity / comorbidity.diagonal().reshape(-1,1)

## Format into DataFrame
comorbidity = pd.DataFrame(comorbidity, columns=keyword_counts.columns, index=keyword_counts.columns)
comorbidity_normalized = pd.DataFrame(comorbidity_normalized, columns=keyword_counts.columns, index=keyword_counts.columns)

## Get Coronavirus Overlap
coronavirus_cols = [c for c in coronavirus_keywords if c in comorbidity.columns]
if len(coronavirus_cols) == 0:
    coronavirus_overlap = pd.Series(index=keyword_counts.columns, data=np.zeros_like(keyword_counts.columns))
else:
    coronavirus_overlap = comorbidity[coronavirus_cols].sum(axis=1)

#######################
### Keyword Context (Neighbors)
#######################

## Initialize Regex for Keywords (To Get Spans)
KEYWORD_REGEX = create_regex_dict(keyword_counts.columns.tolist(), include_hashtag=INCLUDE_HASHTAGS)

## Get Context
context = dict()
for window_lbl, window in zip(["pre","post"],[PRE_COVID_WINDOW, POST_COVID_WINDOW]):
    ## Use Multiprocessing to Get Context
    mp = Pool(NUM_JOBS)
    con_helper = partial(get_context,
                         ngrams=NGRAMS,
                         window=CONTEXT_WINDOW,
                         include_mentions=INCLUDE_MENTIONS,
                         min_date=pd.to_datetime(window[0]),
                         max_date=pd.to_datetime(window[1]))
    win_context = list(tqdm(mp.imap_unordered(con_helper, filenames),
                            desc="{}-COVID 19 Context Calculator".format(window_lbl.title()),
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
    context_vocab = context_vocab.index.tolist()
    ## Use Multiprocessing to Get General Vocab Usage
    mp = Pool(NUM_JOBS)
    voc_helper = partial(get_vocab_usage,
                         vocab=context_vocab,
                         ngrams=NGRAMS,
                         min_date=pd.to_datetime(window[0]),
                         max_date=pd.to_datetime(window[1]))
    win_vocab = list(tqdm(mp.imap_unordered(voc_helper, filenames),
                          desc="{}-COVID 19 Vocab Counter".format(window_lbl.title()),
                          file=sys.stdout,
                          total=len(filenames)))
    mp.close()
    ## Concatenate Counts
    win_vocab = list(filter(lambda i: sum(i.values()) > 0, win_vocab))
    win_vocab_counts = sum(win_vocab, Counter())
    ## Cache 
    context[window_lbl] = {
                            "context_counts":win_context_concat,
                            "keyword_counts":win_keyword_counts,
                            "vocab_counts":win_vocab_counts
                          }

## Compute PMI Values (Need p(x | keyword), p(x))
pmi = {}
for window_lbl, window_counts in context.items():
    ## Get Relative Keyword Frequencies
    p_keyword = pd.Series(window_counts.get("keyword_counts"))
    p_keyword = (p_keyword + SMOOTHING) / (p_keyword + SMOOTHING).sum()
    ## Get Relative N-Gram Frequencies
    p_ngram = pd.DataFrame(window_counts.get("vocab_counts").most_common(), columns=["ngram","freq"])
    p_ngram["nlen"] = p_ngram["ngram"].map(len)
    p_ngram["p_x"] = np.nan
    for n in range(min(NGRAMS), max(NGRAMS)+1):
        p_ngram.loc[p_ngram["nlen"]==n,"p_x"] = (p_ngram.loc[p_ngram["nlen"]==n,"freq"] + SMOOTHING) / \
                                                (p_ngram.loc[p_ngram["nlen"]==n,"freq"] + SMOOTHING).sum()
    ## Concatenate Frequencies/Probabilities
    context_df = []
    for keyword, keyword_context in window_counts.get("context_counts").items():
        keyword_context_df = pd.DataFrame(keyword_context.most_common(), columns=["ngram","context_freq"])
        keyword_context_df["freq"] = keyword_context_df["ngram"].map(lambda i: window_counts.get("vocab_counts").get(i, None))
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
    ## Cache
    pmi[window_lbl] = context_df

## Comparison
pmi_comparison = []
for window_lbl, pmi_df in pmi.items():
    pmi_df_pivot = pd.pivot_table(pmi_df.loc[(pmi_df["context_freq"]>=MIN_CONTEXT_FREQ])&
                                             (pmi_df["freq"]>=MIN_FREQ)],
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
    neighbors["period"] = window_lbl
    pmi_comparison.append(neighbors)

## Format Comparison
neighbor_cols = [f"neighbors_n_{n}" for n in range(min(NGRAMS), max(NGRAMS)+1)]
pmi_comparison = pd.concat(pmi_comparison)
pmi_comparison = pd.pivot_table(pmi_comparison,
                                index="keyword",
                                columns="period",
                                values=neighbor_cols,
                                aggfunc=lambda x:x.iloc[0])
pmi_comparison = pmi_comparison.dropna().copy()
for nc in neighbor_cols:
    pmi_comparison[nc, "overlap"] = pmi_comparison.apply(lambda row: jaccard(row[nc, "pre"],row[nc, "post"]),axis=1)
    for period in ["pre","post"]:
        pmi_comparison[nc, period] = pmi_comparison[nc, period].map(lambda i: ", ".join("_".join(k) for k in i[:25]))

#######################
### Combine Statistics
#######################

## Combine Match Rates
summary = pd.concat([
    overall_num_matches.to_frame("num_matches"),
    overall_match_rate.to_frame("match_rate"),
    coronavirus_overlap.to_frame("coronavirus_overlap"),
    max_match_rate.to_frame("max_match_rate"),
    min_match_rate.to_frame("min_match_rate"),
    median_match_rate.to_frame("median_match_rate"),
    lower_match_rate.to_frame("lower_match_rate"),
    upper_match_rate.to_frame("upper_match_rate"),
    cv_match_rate.to_frame("cv_match_rate"),
], axis=1)

## Append Neighbors
for nc in neighbor_cols:
    summary = pd.concat([summary,
                         pmi_comparison[nc].rename(columns={"post":f"post_{nc}",
                                                            "pre":f"pre_{nc}",
                                                            "overlap":f"overlap_{nc}"})
                        ],
                        axis=1)

## Format
summary = summary.sort_values("num_matches",ascending=False)

## Cache
summary.to_csv(f"{CACHE_DIR}summary.csv")

#######################
### Visualize
#######################

def visualize_keyword_summary(keyword,
                              window=14,
                              min_context_freq=MIN_CONTEXT_FREQ,
                              min_freq=MIN_FREQ,
                              n_examples=5,
                              alpha=0.05):
    """

    """
    ## Get Timeseries (Smoothed and Confidence Intervals)
    posts = n_posts["count"].rolling(window).sum()
    timeseries = keyword_counts[keyword].rolling(window).sum()
    timeseries = timeseries.reindex(posts.index)
    ci_median = timeseries / posts
    ci_lower, ci_upper = proportion_confint(timeseries, posts, alpha=alpha, method="normal")
    ## Get PMI Before
    pmi_pre = pmi.get("pre")
    keyword_pmi_pre = pmi_pre.loc[pmi_pre["keyword"]==keyword]
    keyword_pmi_pre = keyword_pmi_pre.loc[keyword_pmi_pre["context_freq"]>=min_context_freq]
    keyword_pmi_pre = keyword_pmi_pre.loc[keyword_pmi_pre["freq"]>=min_freq]
    keyword_pmi_pre = keyword_pmi_pre.set_index("ngram")["pmi"].copy()
    keyword_pmi_pre = keyword_pmi_pre.nlargest(15)
    keyword_pmi_pre.index = keyword_pmi_pre.index.map(lambda i: " ".join(i))
    keyword_pmi_pre.index = replace_emojis(keyword_pmi_pre.index)
    ## Get PMI After
    pmi_pre = pmi.get("post")
    keyword_pmi_post = pmi_pre.loc[pmi_pre["keyword"]==keyword]
    keyword_pmi_post = keyword_pmi_post.loc[keyword_pmi_post["context_freq"]>=min_context_freq]
    keyword_pmi_post = keyword_pmi_post.loc[keyword_pmi_post["freq"]>=min_freq]
    keyword_pmi_post = keyword_pmi_post.set_index("ngram")["pmi"].copy()
    keyword_pmi_post = keyword_pmi_post.nlargest(15)
    keyword_pmi_post.index = keyword_pmi_post.index.map(lambda i: " ".join(i))
    keyword_pmi_post.index = replace_emojis(keyword_pmi_post.index)
    ## Get Representative Examples
    keyword_reps = representative_examples.loc[representative_examples["keyword"]==keyword]
    keyword_reg = create_regex_dict([keyword],False)
    ## Generate Figure
    fig, ax = plt.subplots(2, 2, figsize=(13,6))
    ax[0,0].fill_between(ci_lower.index, ci_lower, ci_upper, color="C0",alpha=0.4)
    ax[0,0].plot(ci_median.index, ci_median.values, color="C0", alpha=0.8)
    ax[0,0].axvline(pd.to_datetime(POST_COVID_WINDOW[0]), color="black",linestyle="--", alpha=0.8, label="COVID-19 Lockdown")
    ax[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax[0,0].set_ylabel("Proportion\nof Tweets", fontweight="bold")
    ax[0,0].tick_params(axis="x", rotation=45)
    ax[0,0].legend(loc="upper left")
    for tick in ax[0,0].xaxis.get_major_ticks():
         tick.label1.set_horizontalalignment('right')
    keyword_pmi_pre.iloc[::-1].plot.barh(ax=ax[1,0], color="C0", alpha=0.7)
    keyword_pmi_post.iloc[::-1].plot.barh(ax=ax[1,1], color="C0", alpha=0.7)
    for a in ax[1]:
        a.set_ylabel("")
        a.set_xlabel("Context Strength", fontweight="bold")
    start = 1
    right = -1
    bolden = lambda x: " ".join(["$\\bf{"+k+"}$" for k in x.split()])
    for p, post in enumerate(keyword_reps["text"].sample(min(n_examples, len(keyword_reps)), random_state=42).values):
        formatted_post = "\n".join(textwrap.wrap(post.replace("$",""), 90))
        for r, (reg,_) in keyword_reg.items():
            formatted_post= reg.sub("<REPLACE_HERE>", formatted_post)
        formatted_post = formatted_post.replace("<REPLACE_HERE>",bolden(keyword))       
        formatted_post = replace_emojis([formatted_post])[0]         
        txt = ax[0,1].text(0.01, start, formatted_post, fontsize=6, ha="left", va="bottom",
                           bbox=dict(boxstyle="round", facecolor="white", alpha=0, pad=0.1))
        transf = ax[0,1].transAxes.inverted()
        bb = txt.get_window_extent(renderer = fig.canvas.get_renderer())
        bb_datacoords = bb.transformed(transf)
        start = bb_datacoords.y1 + 0.01
        if bb_datacoords.x1 > right:
            right = bb_datacoords.x1
    ax[0,1].set_ylim(1, bb_datacoords.y1)
    ax[0,1].set_xlim(0, right + 0.01)
    ax[0,1].axis("off")
    ax[0,1].set_title("Representative Examples", fontweight="bold", loc="left", fontstyle="italic")
    ax[0,0].set_title("Match Rate", fontweight="bold", loc="left", fontstyle="italic")
    ax[1,0].set_title("Pre COVID-19 Context", fontweight="bold", loc="left", fontstyle="italic")
    ax[1,1].set_title("Post COVID-19 Context", fontweight="bold", loc="left", fontstyle="italic")
    for a in ax:
        for b in a:
            b.spines["top"].set_visible(False)
            b.spines["right"].set_visible(False)
            b.tick_params(labelsize=8)
    fig.tight_layout()
    term_lists = mental_health_keywords_reverse.get(keyword, None)
    if term_lists:
        fig.suptitle("Keyword: {} ({})".format(keyword, ", ".join(term_lists)), fontweight="bold", y=.975)
    else:
        fig.suptitle(f"Keyword: {keyword}", fontweight="bold", y=.975)
    fig.subplots_adjust(top=.875)
    return fig, ax

## Generate Figures
summary_dir = f"{PLOT_DIR}summaries/"
if not os.path.exists(summary_dir):
    _ = os.makedirs(summary_dir)
errors = []
to_plot = keyword_counts.sum(axis=0)
to_plot = to_plot.loc[to_plot >= 250].index.tolist()
for keyword in tqdm(to_plot):
    keyword_clean = keyword.replace(" ","_").replace("/","-").replace(":","-")
    try:
        fig, ax = visualize_keyword_summary(keyword,
                                            window=14,
                                            n_examples=5,
                                            min_context_freq=MIN_CONTEXT_FREQ,
                                            min_freq=MIN_FREQ)
        fig.savefig(f"{summary_dir}{keyword_clean}.png", dpi=300)
        plt.close(fig)
    except IndexError:
        errors.append(keyword)
        plt.close()
        continue
    except KeyboardInterrupt:
        break
