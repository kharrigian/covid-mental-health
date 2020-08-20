
########################
## Configuration
########################

## Designate Models
MODEL_A = {
        "path":"./data/results/twitter/2018-2020/word2vec/word2vec_2019-02_2019-06/word2vec.model",
        "name":"2019"
}
MODEL_B = {
        "path":"./data/results/twitter/2018-2020/word2vec/word2vec_2020-02_2020-06/word2vec.model",
        "name":"2020"
}

## Output Directory
OUTPUT_DIR = "./data/results/twitter/2018-2020/word2vec/"

## Parameters
K_NEIGHBORS = 300
PRIMARY_MIN_FREQUENCY = 100
SECONDARY_MIN_FREQUENCY = 250
SHOW_TOP = 50 ## Number of Top Examples
DISPLAY_TOP = 30 ## Number of Words Per Example
K_CORRELATION_ANALYSIS = False

########################
## Imports
########################

## Standard Library
import os
import sys
import json
from glob import glob

## External
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise
from gensim.models import Word2Vec

########################
## Keywords
########################

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

## Create Match Dictionary (Subreddit Lists + Term Regular Expressions)
MATCH_DICT = {
    "mental_health":{
        "terms":MH_TERMS["terms"]["smhd"],
        "name":"SMHD"
    },
    "crisis":{
        "terms":CRISIS_KEYWORDS,
        "name":"JHU Crisis"
    },
    "mental_health_keywords":{
        "terms":MH_KEYWORDS,
        "name":"JHU CLSP"
    },
    "covid":{
        "terms":COVID_TERMS["covid"],
        "name":"COVID-19"
    }
}

########################
## Helpers
########################

def intersection_at_k(a, b):
    """

    """
    if not isinstance(a, set):
        a = set(a)
    if not isinstance(b, set):
        b = set(b)
    assert len(a) == len(b)
    overlap = set(a) & set(b)
    return len(overlap)

########################
## Prepare Models/Data
########################

## Load Models
word2vec_a = Word2Vec.load(MODEL_A.get("path"))
word2vec_b = Word2Vec.load(MODEL_B.get("path"))

## Get Shared Vocabulary
vocab_a = word2vec_a.wv.index2entity; V_a = len(vocab_a)
vocab_b = word2vec_b.wv.index2entity; V_b = len(vocab_b)
vocab_shared = set(vocab_a) & set(vocab_b); V_shared = len(vocab_shared)

## Get Frequencies
freq_a = np.array([word2vec_a.vocabulary.raw_vocab[i] for i in vocab_a])
freq_b = np.array([word2vec_b.vocabulary.raw_vocab[i] for i in vocab_b])

########################
## Compute Similarity
########################

print("Computing Similarity")

## Frequency Mask
mask_a = np.nonzero(freq_a >= PRIMARY_MIN_FREQUENCY)[0]
mask_b = np.nonzero(freq_b >= PRIMARY_MIN_FREQUENCY)[0]

## Mask the Vocabularies
vocab_a_masked_term2ind = dict((vocab_a[m],m) for m in mask_a)
vocab_b_masked_term2ind = dict((vocab_b[m],m) for m in mask_b)
vocab_shared_masked = sorted(set(vocab_a_masked_term2ind) & set(vocab_b_masked_term2ind))
vocab_shared_term2ind = dict((v, i) for i, v in enumerate(vocab_shared_masked))

## Get Shared Masks For Each Domain
mask_shared_a = [vocab_a_masked_term2ind[v] for v in vocab_shared_masked]
mask_shared_b = [vocab_b_masked_term2ind[v] for v in vocab_shared_masked]

## Compute Similarity Matrices
sim_a = pairwise.cosine_similarity(word2vec_a.wv.vectors[mask_shared_a])
sim_b = pairwise.cosine_similarity(word2vec_b.wv.vectors[mask_shared_b])

## Get Sorted Values
top_k_a = np.argsort(sim_a,axis=1)[:,::-1][:,1:]
top_k_b = np.argsort(sim_b,axis=1)[:,::-1][:,1:]

########################
## Compute Intersection
########################

## Optionally Run Analysis of Metric ~ K
if K_CORRELATION_ANALYSIS:

    print("Running Correlation Analysis")

    ## Compute Intersection at K (over many K)
    kthresh = np.arange(100, len(vocab_shared_masked)-1, 250).astype(int)
    intersection = np.zeros((len(vocab_shared_masked),len(kthresh)))
    for t, threshold in tqdm(enumerate(kthresh), total=len(kthresh), desc="Computing Intersection at K"):
        overlap_at_k = list(map(lambda a, b: intersection_at_k(a[:threshold], b[:threshold]), top_k_a, top_k_b))
        intersection[:,t] = overlap_at_k

    ## Correlation
    corr = pd.DataFrame(intersection, columns=kthresh).corr(method="spearman").iloc[::-1]

    ## Plot Correlation
    fig, ax = plt.subplots(figsize=(10,5.8))
    m = ax.imshow(corr.values, aspect="auto", cmap=plt.cm.coolwarm)
    cbar = fig.colorbar(m)
    cbar.set_label("Spearman Correlation")
    ax.set_xlabel("Threshold (1)", fontweight="bold")
    ax.set_ylabel("Threshold (2)", fontweight="bold")
    ax.set_xticks(list(range(corr.shape[1])))
    ax.set_xticklabels(list(map(lambda i, v: "{:,d}".format(v) if i % 5 == 0 else "", range(corr.shape[1]), corr.columns)))
    ax.set_yticks(list(range(corr.shape[1])))
    ax.set_yticklabels(list(map(lambda i, v: "{:,d}".format(v) if i % 5 == 0 else "", range(corr.shape[1]), corr.columns))[::-1])
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}overlap_at_k_correlation.png", dpi=300)
    plt.close(fig)

## Compute Intersection
print(f"Computing Intersection at {K_NEIGHBORS}")
nearest_neighbors = pd.Series(list(map(lambda a, b: intersection_at_k(a[:K_NEIGHBORS], b[:K_NEIGHBORS]), top_k_a, top_k_b)),
                              index=vocab_shared_masked).to_frame("overlap")
nearest_neighbors["freq_a"] = nearest_neighbors.index.map(word2vec_a.vocabulary.raw_vocab.get)
nearest_neighbors["freq_b"] = nearest_neighbors.index.map(word2vec_b.vocabulary.raw_vocab.get)

## Apply Secondary Filtering and Sort
nearest_neighbors_filtered = nearest_neighbors.loc[(nearest_neighbors[["freq_a","freq_b"]] >= SECONDARY_MIN_FREQUENCY).all(axis=1)].sort_values("overlap")

########################
## Show Biggest Changes
########################

print("########### TERMS WITH THE BIGGEST CHANGES ###########")

## Display Examples
for t, term in enumerate(nearest_neighbors_filtered.index[:SHOW_TOP]):
    ## Get Top Values
    top_a = [vocab_shared_masked[v] for v in top_k_a[vocab_shared_term2ind[term]]][:DISPLAY_TOP]
    top_b = [vocab_shared_masked[v] for v in top_k_b[vocab_shared_term2ind[term]]][:DISPLAY_TOP]
    ## Show Top Values
    print("#"*5 + f" {t+1}) {term} " + "#"*5)
    print("\t{}: ".format(MODEL_A.get("name")) + ", ".join(top_a)) 
    print("\t{}: ".format(MODEL_B.get("name")) + ", ".join(top_b)+"\n") 

########################
## Show Keyword Changes
########################

print("########### Keyword Analysis ###########")

## Show Changes Specifically Relevant To Keywords
changes = {}
change_scores = []
for term_group, term_values in MATCH_DICT.items():
    group_name = term_values.get("name")
    changes[group_name] = {}
    print("#"*50 + f"\n### Keyword Group: {group_name}\n" + "#" *50 + "\n")
    for term in term_values.get("terms"):
        if term.lower() not in vocab_shared_term2ind:
            continue
        ## Get Top Values
        top_a = [vocab_shared_masked[v] for v in top_k_a[vocab_shared_term2ind[term.lower()]]][:DISPLAY_TOP]
        top_b = [vocab_shared_masked[v] for v in top_k_b[vocab_shared_term2ind[term.lower()]]][:DISPLAY_TOP]
        ## Get Change Score
        change_score = nearest_neighbors.loc[term.lower()]["overlap"] / K_NEIGHBORS
        ## Cache Top Values
        changes[group_name][term] = {MODEL_A.get("name"):top_a, MODEL_B.get("name"):top_b}
        change_scores.append((group_name, term, change_score))
        ## Show Top Values
        print("#"*5 + f" {term} " + "#"*5)
        print("\t{}: ".format(MODEL_A.get("name")) + ", ".join(top_a)) 
        print("\t{}: ".format(MODEL_B.get("name")) + ", ".join(top_b)+"\n") 

## Format Change Scores
change_scores = pd.DataFrame(change_scores,columns=["keyword_group","term","overlap_score"])
change_scores = change_scores.drop_duplicates(["keyword_group","term"])
change_scores = change_scores.groupby(["term"]).agg({"keyword_group":lambda x: ", ".join(sorted(x)), "overlap_score":max})
change_scores.sort_values("overlap_score", inplace=True)

## Format Change Lists
changes_df = []
for group_name in changes.keys():
    group_changes_df = pd.DataFrame(changes[group_name]).T
    group_changes_df["group"] = group_name
    group_changes_df["overlap_score"] = group_changes_df.index.map(lambda i: change_scores.loc[i, "overlap_score"])
    group_changes_df["keyword_group"] = group_changes_df.index.map(lambda i: change_scores.loc[i, "keyword_group"])
    changes_df.append(group_changes_df)
changes_df = pd.concat(changes_df)
changes_df = changes_df[~changes_df.index.duplicated(keep='first')].sort_values("overlap_score")
for col in [MODEL_A.get("name"), MODEL_B.get("name")]:
    changes_df[col] = changes_df[col].map(lambda i: ", ".join(i))

## Dump Changes
changes_df.to_csv(f"{OUTPUT_DIR}keyword_changes_K-{K_NEIGHBORS}_Min1-{PRIMARY_MIN_FREQUENCY}_Min2-{SECONDARY_MIN_FREQUENCY}.csv")
