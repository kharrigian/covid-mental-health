
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
import shap
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

## Local
from mhlib.util.logging import initialize_logger

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
                        help="Path to cached model directory")
    parser.add_argument("input",
                        type=str,
                        help="Path to processed test data")
    parser.add_argument("output_folder",
                        type=str,
                        help="Path to output folder")
    parser.add_argument("--sample_percent",
                        type=float,
                        default=100,
                        help="(0,100] value for sampling files from each set")
    parser.add_argument("--max_sample_size",
                        type=int,
                        default=None,
                        help="If included, represents the max number of files considered in a sample.")
    parser.add_argument("--min_date_test",
                        type=str,
                        default="2020-01-01",
                        help="Lower date bound for test data")
    parser.add_argument("--max_date_test",
                        type=str,
                        default="2020-06-01",
                        help="Upper date bound for test data")
    parser.add_argument("--n_samples",
                        type=int,
                        default=None,
                        help="Number of post samples to use in test data")
    parser.add_argument("--randomized",
                        action="store_true",
                        default=False,
                        help="If sampling posts, whether it should be done randomly.")
    parser.add_argument("--filter_null",
                        action="store_true",
                        default=False,
                        help="If included, removes samples without any matched vocabulary terms")
    parser.add_argument("--keep_covid",
                        action="store_true",
                        default=False,
                        help="If included, keeps samples that match COVID keyword list in test set")
    parser.add_argument("--model_shift",
                        action="store_true",
                        default=False,
                        help="If included, fits logistic regression model between sample splits")
    parser.add_argument("--random_state",
                        type=int,
                        default=42,
                        help="Decides random state to use for any sampling procedures")
    ## Parse Arguments
    args = parser.parse_args()
    return args

def bootstrap_sample(X,
                     Y=None,
                     func=np.mean,
                     axis=0,
                     sample_percent=70,
                     samples=100):
    """

    """
    sample_size = int(sample_percent / 100 * X.shape[0])
    if Y is not None:
        sample_size_Y = int(sample_percent / 100 * Y.shape[0])
    estimates = []
    for sample in range(samples):
        sample_ind = np.random.choice(X.shape[0], size=sample_size, replace=True)
        X_sample = X[sample_ind]
        if Y is not None:
            samply_ind_y = np.random.choice(Y.shape[0], size=sample_size_Y, replace=True)
            Y_sample = Y[samply_ind_y]
            sample_est = func(X_sample, Y_sample)
        else:
            sample_est = func(X_sample, axis=axis)
        estimates.append(sample_est)
    estimates = np.vstack(estimates)
    ci = np.percentile(estimates, [2.5, 50, 97.5], axis=axis)
    return ci

def load_trained_model(args):
    """

    """
    ## Load Model
    model = joblib.load(args.model+"model.joblib")
    ## Load Training Labels
    splits = json.load(open(args.model+"splits.json","r"))
    train_labels = {}
    for fold, fold_data in splits["train"].items():
        train_labels.update(fold_data["dev"])
    test_labels = splits["test"]["1"]["test"]
    return model, train_labels, test_labels

def vectorize_files(model,
                    filenames,
                    min_date=None,
                    max_date=None,
                    n_samples=None,
                    randomized=False):
    """

    """
    ## Bag of Words Representation
    filenames, X, _ = model._load_vectors(filenames,
                                          None,
                                          min_date=min_date,
                                          max_date=max_date,
                                          n_samples=n_samples, 
                                          randomized=randomized)
    ## Identify Users With Features
    nonnull_mask = np.nonzero(X.sum(axis=1)>0)[0]
    return filenames, X, nonnull_mask

def get_file_list(args):
    """

    """
    if os.path.isfile(args.input):
        return [args.input]
    elif os.path.isdir(args.input):
        return glob(f"{args.input}*.gz")
    else:
        raise ValueError("Did not recognize command line --input")

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

def plot_shap(X,
              SV,
              feature_names,
              top_k=50):
    """

    """
    _ = shap.summary_plot(shap_values=SV,
                          features=X,
                          feature_names=feature_names,
                          max_display=top_k)
    fig = plt.gcf()
    fig.tight_layout()
    return fig

def plot_shap_comparison(SV_train,
                         SV_test,
                         feature_names,
                         xlabel="Training",
                         ylabel="Test",
                         include_error=False):
    """

    """
    ## Percentile Influence Range
    shap_train_q = bootstrap_sample(SV_train,
                                    axis=0,
                                    sample_percent=70,
                                    samples=500)
    shap_test_q = bootstrap_sample(SV_test,
                                   axis=0,
                                   sample_percent=70,
                                   samples=500)
    ## Plot Comparison
    train_var = np.vstack([(shap_train_q[1]-shap_train_q[0]), (shap_train_q[2]-shap_train_q[1])])
    test_var = np.vstack([(shap_test_q[1]-shap_test_q[0]), (shap_test_q[2]-shap_test_q[1])])
    fig, ax = plt.subplots()
    ax.errorbar(shap_train_q[1],
                shap_test_q[1],
                xerr=train_var if include_error else None,
                yerr=test_var if include_error else None,
                fmt="x",
                ms=10,
                alpha=0.01)
    ax.set_xlabel(f"Average Shap Value ({xlabel})", fontweight="bold")
    ax.set_ylabel(f"Average Shap Value ({ylabel})", fontweight="bold")
    ax.axvline(0, color="black", alpha=0.2, linestyle="--")
    ax.axhline(0, color="black", alpha=0.2, linestyle="--")
    xrange = np.percentile(shap_train_q[1], [5,95])
    yrange = np.percentile(shap_test_q[1], [5,95])
    ax.set_xlim(xrange[0],xrange[1])
    ax.set_ylim(yrange[0],yrange[1])
    ax.set_xticks([0])
    ax.set_yticks([0])
    fig.tight_layout()
    ## Format Percentile Range
    shap_q_df = pd.DataFrame([shap_train_q[1], shap_test_q[1]],
                             columns=feature_names,
                             index=["train","test"]).T
    return (fig, ax), shap_q_df

def plot_probability_distributions(model,
                                   features,
                                   names):
    """

    """
    ## Create Figure
    fig, ax = plt.subplots(len(features), 1, figsize=(10,5))
    bins = np.linspace(0, 1, 21)
    for i, (x, g) in enumerate(zip(features,names)):
        p = model.model.predict_proba(x)[:,1]
        q = bootstrap_sample(p).T[0]
        ax[i].hist(p, bins=bins, alpha=.7)
        for j, l in zip(q, ["--","-","--"]):
            ax[i].axvline(j, linestyle=l, linewidth=2, color="C0", alpha=.7)
            if l == "-":
                ax[i].text(j-0.01,
                           ax[i].get_ylim()[1] / 2,
                           "{:.2f}".format(j),
                           rotation=90,
                           ha="center",
                           va="center",
                           fontweight="bold")
        ax[i].set_title(f"{g} Data", loc="left", fontweight="bold")
        ax[i].set_ylabel("Sample\nSize",fontweight="bold")
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
    ax[-1].set_xlabel("Pr(Condition)", fontweight="bold")
    fig.tight_layout()
    return fig, ax

def model_dataset_shift(X1,
                        X2,
                        y1=None,
                        y2=None,
                        feature_names=None,
                        balance=True,
                        n1=None,
                        n2=None):
    """

    """
    LOGGER.info("~"*100 + f"\nModeling Dataset Shift ({n1} vs. {n2})\n" + "~"*100)
    features = []
    for y_gt, gt_mask in zip([[0], [1], [0, 1]], ["control","target","mixed"]):
        ## Get Sample Points and Concatenate Data Sets
        if y1 is not None:
            mask1 = np.nonzero([i in y_gt for i in y1])[0]
        else:
            mask1 = np.arange(X1.shape[0])
        if y2 is not None:
            mask2 = np.nonzero([i in y_gt for i in y2])[0]
        else:
            mask2 = np.arange(X2.shape[0])
        X_m = np.vstack([X1[mask1], X2[mask2]])
        y_m = np.array([0]*len(mask1) + [1] * len(mask2))
        ## Balance
        if balance:
            train_idx = np.nonzero(y_m==0)[0]
            test_idx = np.nonzero(y_m==1)[0]
            balance_size = min(len(train_idx), len(test_idx))
            train_idx = sorted(np.random.choice(train_idx, balance_size, replace=False))
            test_idx = sorted(np.random.choice(test_idx, balance_size, replace=False))
            idx = np.array([train_idx + test_idx])[0]
            X_m = X_m[idx]
            y_m = y_m[idx]
        ## Sample
        X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_m, y_m, test_size=0.2, shuffle=True)
        ## Fit Model
        model = LogisticRegression(C=1000, max_iter=1000, solver="lbfgs")
        model.fit(X_m_train, y_m_train)
        ## Predict
        y_pred_train = model.predict(X_m_train)
        y_pred_test = model.predict(X_m_test)
        ## Score
        score_train = metrics.classification_report(y_m_train, y_pred_train, output_dict=False)
        score_test = metrics.classification_report(y_m_test, y_pred_test, output_dict=False)
        LOGGER.info("~"*50 + "\nTrain Group Label: {}\n".format(gt_mask.title()) + "~"*50)
        LOGGER.info("Train:")
        LOGGER.info(score_train)
        LOGGER.info("Test:")
        LOGGER.info(score_test)
        ## Features
        if feature_names is None:
            feature_names = list(range(X_m_train.shape[1]))
        feature_coefs = pd.Series(model.coef_[0], index=feature_names)
        feature_coefs.name = gt_mask
        features.append(feature_coefs)
    ## Concatenate Features
    features = pd.concat(features, axis=1, sort=True)
    return features

def filter_nn(mask,
              X,
              files):
    """

    """
    files = [files[i] for i in mask]
    X = X[mask]
    return X, files

def summarize_bow(X,
                  model):
    """

    """
    ## Token Distribution (per user)
    n_tokens_per_user = X.sum(axis=1).astype(int)
    n_unique_tokens_per_user = (X>0).sum(axis=1).astype(int)
    ## Token Distribution (across users)
    token_dist = X.sum(axis=0).astype(int)
    binary_token_dist = (X>0).sum(axis=0).astype(int)
    binary_token_dist_norm = binary_token_dist / X.shape[0]
    ## LIWC Hit Rate
    liwc = model.preprocessor._transformers.get("liwc")
    if liwc is not None:
        liwc_vocab = (liwc._dim_map>0).any(axis=1).reshape(-1,1).astype(int)
        liwc_matches = np.matmul(X, liwc_vocab).T[0].astype(int)
        liwc_hit_rate = np.divide(liwc_matches,
                                  n_tokens_per_user,
                                  where=n_tokens_per_user>0)
        liwc_hit_rate[n_tokens_per_user==0] = np.nan
        liwc_hit_rate = np.array([g for g in liwc_hit_rate if not np.isnan(g)])

    else:
        liwc_hit_rate = None
    ## GloVe Hit Rate
    glove = model.preprocessor._transformers.get("glove")
    if glove is not None:
        glove_vocab = np.any(glove.embedding_matrix!=0,axis=1).reshape(-1,1).astype(int)
        glove_matches = np.matmul(X, glove_vocab).T[0].astype(int)
        glove_hit_rate = np.divide(glove_matches,
                                  n_tokens_per_user,
                                  where=n_tokens_per_user>0)
        glove_hit_rate[n_tokens_per_user==0] = np.nan
        glove_hit_rate = np.array([g for g in glove_hit_rate if not np.isnan(g)])
    else:
        glove_hit_rate = None
    ## Group Information
    output = {"n_tokens_per_user":n_tokens_per_user,
              "n_unique_tokens_per_user":n_unique_tokens_per_user,
              "prob_token":binary_token_dist_norm,
              "liwc_hit_rate":liwc_hit_rate,
              "glove_hit_rate":glove_hit_rate}
    return output

def plot_bow_summary(summary_dicts,
                     names):
    """
    2 Figures:
        - Distribution (histogram) comparison
        - Token Distribution Correlation
    """
    ## Distributional Figure
    fig1, axes1 = plt.subplots(2, 3, figsize=(10,5))
    for i, (sd, nm) in enumerate(zip(summary_dicts, names)):
        ## Tokens Per User
        c, b = np.histogram(sd["n_tokens_per_user"], bins=20)
        c = c / c.sum()
        b = (b[:-1] + b[1:]) / 2
        axes1[0,0].plot(b,
                        c,
                        color=f"C{i}",
                        label=nm,
                        marker="o",
                        alpha=0.5)
        axes1[0,0].set_xscale("symlog")
        ## Unique Tokens Per User
        c, b = np.histogram(sd["n_unique_tokens_per_user"], bins=20)
        c = c / c.sum()
        b = (b[:-1] + b[1:]) / 2
        axes1[0,1].plot(b,
                        c,
                        color=f"C{i}",
                        label=nm,
                        marker="o",
                        alpha=0.5)
        axes1[0,1].set_xscale("symlog")
        ## Average Token Usage
        avg_token_usage = sd["n_tokens_per_user"] / sd["n_unique_tokens_per_user"]
        avg_token_usage = np.array([i for i in avg_token_usage if not np.isnan(i)])
        c, b = np.histogram(avg_token_usage, bins=20)
        c = c / c.sum()
        b = (b[:-1] + b[1:]) / 2
        axes1[0,2].plot(b,
                        c,
                        color=f"C{i}",
                        label=nm,
                        marker="o",
                        alpha=0.5)
        ## LIWC Hit Rate
        if sd["liwc_hit_rate"] is not None:
            c, b = np.histogram(sd["liwc_hit_rate"], bins=20)
            c = c / c.sum()
            b = (b[:-1] + b[1:]) / 2
            axes1[1,0].plot(b,
                            c,
                            color=f"C{i}",
                            label=nm,
                            marker="o",
                            alpha=0.5)
        else:
            axes1[1,0].axis("off")
        ## GloVe Hit Rate
        if sd["glove_hit_rate"] is not None:
            c, b = np.histogram(sd["glove_hit_rate"], bins=20)
            c = c / c.sum()
            b = (b[:-1] + b[1:]) / 2
            axes1[1,1].plot(b,
                            c,
                            color=f"C{i}",
                            label=nm,
                            marker="o",
                            alpha=0.5)
        else:
            axes1[1,1].axis("off")
        ## Legend
        axes1[1,2].plot([],
                        [],
                        color=f"C{i}",
                        label=nm,
                        marker="o")
        axes1[1,2].axis("off")
    ## Formatting
    axes1[0,0].set_title("Tokens Per User", loc="left", fontsize=10)
    axes1[0,1].set_title("Unique Tokens Per User", loc="left", fontsize=10)
    axes1[0,2].set_title("Average Token Usage", loc="left", fontsize=10)
    axes1[1,2].legend(loc="center", title="Data Sample")
    if sd["liwc_hit_rate"] is not None:
        axes1[1,0].set_title("LIWC Hit Rate", loc="left", fontsize=10)
    if sd["glove_hit_rate"] is not None:
        axes1[1,1].set_title("GloVe Hit Rate", loc="left", fontsize=10)
    fig1.tight_layout()
    ## Token Distribution Correlation
    fig2, axes2 = plt.subplots(len(names), len(names), figsize=(10,8))
    for i, (sd, nm) in enumerate(zip(summary_dicts, names)):
        for j, (sdj, nmj) in enumerate(zip(summary_dicts, names)):
            if i == j:
                 axes2[i,j].hist(sd["prob_token"],
                                 bins=100,
                                 color="C0",
                                 alpha=0.8)
                 axes2[i,j].set_yscale("symlog")
            else:
                axes2[i,j].scatter(sdj["prob_token"],
                                   sd["prob_token"],
                                   s=5,
                                   alpha=0.05)
            if j == 0:
                axes2[i,j].set_ylabel(nm , fontweight="bold")
            if i == len(names)-1:
                axes2[i,j].set_xlabel(nmj, fontweight="bold")
    fig2.tight_layout()
    fig2.suptitle("Unigram Probability Distributions", fontweight="bold", y=.98)
    fig2.subplots_adjust(top=.95)
    return fig1, fig2

def plot_matrices(x=[],
                  names=[],
                  sample_size=100,
                  transform=lambda y: y,
                  cbar_label=None,
                  scale_bounds=None):
    """

    """
    ## Stack and Note Boundaries
    X = []
    name_ranges = {}
    cur_ind = 0
    for _x, _n in zip(x, names):
        _x_N = _x.shape[0]
        _x_N_sample = min(sample_size, _x_N)
        _x_sample = np.random.choice(_x_N, _x_N_sample, replace=False)
        X.append(_x[_x_sample])
        name_ranges[_n] = [cur_ind, cur_ind + _x_N_sample]
        cur_ind += _x_N_sample
    X = np.vstack(X)   
    ## Transformations
    X = transform(X)
    ## Bounds
    if scale_bounds is not None:
        bounds = np.percentile(X, scale_bounds)
        X = np.where(X > bounds[1], bounds[1], X)
        X = np.where(X < bounds[0], bounds[0], X)
    ## Plot
    fig, ax = plt.subplots(figsize=(10,5))
    f = ax.imshow(X,
                  aspect="auto",
                  cmap=plt.cm.coolwarm)
    ax.set_xlabel("Feature", fontweight="bold")
    cbar = fig.colorbar(f)
    if cbar_label is not None:
        cbar.set_label(cbar_label, fontweight="bold")
    yticks = []
    yticklabels = []
    for name, rang in name_ranges.items():
        yticks.append(rang[0]); yticklabels.append("")
        yticks.append((rang[0] + rang[1])/2); yticklabels.append(name)
        yticks.append(rang[1]); yticklabels.append("")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation=90, ha="center", va="center")
    ax.tick_params(axis='y', which='major', pad=15)
    ax.set_ylim(X.shape[0]-0.5, -0.5)
    fig.tight_layout()
    return fig, ax

def symfunc(x,
            func=np.log10):
    """

    """
    x_t = np.where(x > 0, func(x), -func(abs(x)))
    return x_t

def feature_index_map(feature_names):
    """

    """
    ## Identify Feature Indices
    glove = set([i for i, f in enumerate(feature_names) if f.startswith("GloVe_Dim_")])
    liwc = set([i for i, f in enumerate(feature_names) if f.startswith("LIWC=")])
    lda = set([i for i, f in enumerate(feature_names) if f.startswith("LDA_TOPIC_")])
    tok = set([i for i, f in enumerate(feature_names) if i not in glove | liwc | lda])
    ## Define
    feature_groups = {}
    if len(glove) > 0:
        feature_groups["glove"] = sorted(glove)
    if len(liwc) > 0:
        feature_groups["liwc"] = sorted(liwc)
    if len(lda) > 0:
        feature_groups["lda"] = sorted(lda)
    if len(tok) > 0:
        feature_groups["tokens"] = sorted(tok)
    return feature_groups

def plot_matrix_distributions(x=[],
                              names=[],
                              bins=20,
                              func=np.mean,
                              stat_name="Mean"):
    """

    """
    ## Plot 1: Histogram
    fig, ax = plt.subplots(2, 1, figsize=(10,5))
    for i, (_x, _n) in enumerate(zip(x, names)):
        ## Calculate Statistics
        stat_col = func(_x, axis=0)
        stat_row = func(_x, axis=1)
        ## Column Distribution
        c, b = np.histogram(stat_col, bins=bins)
        c = c / c.sum()
        b = (b[:-1] + b[1:]) / 2
        ax[0].plot(b,
                   c,
                   color=f"C{i}",
                   label=_n,
                   marker="o",
                   alpha=0.5)
        ## Row Distribution
        c, b = np.histogram(stat_row, bins=bins)
        c = c / c.sum()
        b = (b[:-1] + b[1:]) / 2
        ax[1].plot(b,
                   c,
                   color=f"C{i}",
                   label=_n,
                   marker="o",
                   alpha=0.5)
    for i in range(2):
        ax[i].set_ylabel("Proportion", fontweight="bold")
        ax[i].legend(loc="upper right", frameon=True)
    ax[0].set_xlabel(f"Column {stat_name} Value Distribution", fontweight="bold")
    ax[1].set_xlabel(f"Row {stat_name} Value Distribution", fontweight="bold")
    fig.tight_layout()
    ## Plot 2: Pair Plot of Means
    ci = {}
    ci_pearson = {}
    ci_spearman = {}
    for _x, _n in zip(x, names):
        ci[_n] = bootstrap_sample(_x,
                                  func=np.mean,
                                  axis=0,
                                  sample_percent=70,
                                  samples=100)
        for __x, __n in zip(x, names):
            if _n == __n:
                continue
            ci_pearson[(_n, __n)] = bootstrap_sample(X=_x,
                                                     Y=__x,
                                                     func=lambda x, y: np.array([pearsonr(x.mean(axis=0), y.mean(axis=0))[0]]),
                                                     sample_percent=70,
                                                     samples=100)
            ci_spearman[(_n, __n)] = bootstrap_sample(X=_x,
                                                      Y=__x,
                                                      func=lambda x, y: np.array([spearmanr(x.mean(axis=0), y.mean(axis=0))[0]]),
                                                      sample_percent=70,
                                                      samples=100)
    fig2, axes = plt.subplots(len(x), len(x), figsize=(10,8))
    for i, i_n in enumerate(names):
        i_x_ci = ci[i_n]
        for j, j_n in enumerate(names):
            ## Case 1: Histogram
            if i == j:
                axes[i,j].hist(i_x_ci[1],
                               bins=min(100, int(len(i_x_ci[1]) / 10)),
                               color="C0",
                               alpha=0.5)
            ## Case 2: Correlation
            else:
                j_x_ci = ci[j_n]
                j_var = np.vstack([(j_x_ci[1]-j_x_ci[0]), (j_x_ci[2]-j_x_ci[1])])
                i_var = np.vstack([(i_x_ci[1]-i_x_ci[0]), (i_x_ci[2]-i_x_ci[1])])
                axes[i,j].errorbar(j_x_ci[1],
                                   i_x_ci[1],
                                   xerr=j_var,
                                   yerr=i_var,
                                   fmt="o",
                                   color="C0",
                                   alpha=0.4)
                p_corr = ci_pearson[(j_n,i_n)].T[0]
                s_corr = ci_spearman[(j_n,i_n)].T[0]
                axes[i,j].set_title("Pearson: {:.2f} ({:.2f},{:.2f}); Spearman: {:.2f} ({:.2f},{:.2f})".format(
                                    p_corr[1], p_corr[0], p_corr[2], s_corr[1], s_corr[0], s_corr[2]
                ), fontsize=6, loc="left")
            if j == 0:
                axes[i,j].set_ylabel(i_n, fontweight="bold")
            if i == len(names)-1:
                axes[i,j].set_xlabel(j_n, fontweight="bold")
    fig2.tight_layout()
    fig2.suptitle(stat_name, fontweight="bold", y=0.98)
    fig2.subplots_adjust(top=.92)
    return (fig, ax), (fig2, axes)

def plot_token_probability_correlation(model,
                                       summary_dicts=[],
                                       X_vals=[],
                                       names=[]):
    """

    """
    fig, ax = plt.subplots(len(names), 1, figsize=(10,8))
    for i, (summary_dict, X, name) in enumerate(zip(summary_dicts, X_vals, names)):
        preds = model.model.predict_proba(X)[:,1]
        ax[i].scatter(summary_dict["n_tokens_per_user"], preds, color="C0", alpha=.6, label="Total")
        ax[i].scatter(summary_dict["n_unique_tokens_per_user"], preds, color = "C1", alpha=.6, label="Unique")
        ax[i].set_xlabel("# Tokens")
        ax[i].set_ylabel("Predicted Probability")
        ax[i].legend(loc = "lower right", frameon=True, fontsize=8)
        ax[i].set_xscale("log")
    fig.tight_layout()
    return fig, ax

def main():
    """

    """
    ## Parse Arguments
    LOGGER.info("Parsing Command-line")
    args = parse_arguments()
    ## Set Seed
    LOGGER.info("Setting Random Seed")
    np.random.seed(args.random_state)
    ## Output Directory
    LOGGER.info("Initializing Output Directory")
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    ## Load Model and Training Labels
    LOGGER.info("Loading Trained Model")
    model, train_labels, test_labels_in = load_trained_model(args)
    ## Get Files
    train_files = sorted(train_labels.keys())
    test_files_in = sorted(test_labels_in.keys())
    test_files_ood = get_file_list(args)
    ## Sample Files
    if args.max_sample_size is None:
        max_sample_size = 100000
    else:
        max_sample_size = args.max_sample_size
    train_files = np.random.choice(train_files, min(max_sample_size, int(len(train_files) * args.sample_percent / 100)), replace=False)
    test_files_in = np.random.choice(test_files_in, min(max_sample_size, int(len(test_files_in) * args.sample_percent / 100)), replace=False)
    test_files_ood = np.random.choice(test_files_ood, min(int(len(test_files_ood) * args.sample_percent / 100), max_sample_size), replace=False)
    ## Vectorize Training Data
    LOGGER.info(f"Vectorizing Training Data ({len(train_files)} files)")
    train_files, X_train, nnmask_train = vectorize_files(model,
                                                         train_files,
                                                         min_date=model._min_date,
                                                         max_date=model._max_date,
                                                         n_samples=model._n_samples,
                                                         randomized=model._randomized)
    LOGGER.info(f"Vectorizing Test Data (In-distribution) ({len(test_files_in)} files)")
    test_files_in, X_test_in, nnmask_test_in = vectorize_files(model,
                                                         test_files_in,
                                                         min_date=model._min_date,
                                                         max_date=model._max_date,
                                                         n_samples=model._n_samples,
                                                         randomized=model._randomized)
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
    ## Load  Test Data
    LOGGER.info("Vectorizing Test Data (OOD) ({} files)".format(len(test_files_ood)))
    test_files_ood, X_test_ood, nnmask_test_ood = vectorize_files(model,
                                                      sorted(test_files_ood),
                                                      min_date=args.min_date_test,
                                                      max_date=args.max_date_test,
                                                      n_samples=args.n_samples,
                                                      randomized=args.randomized)
    ## Filter Out Nulls
    if args.filter_null:
        LOGGER.info("Filtering Out Null Rows")
        X_train, train_files = filter_nn(nnmask_train, X_train, train_files)
        X_test_in, test_files_in = filter_nn(nnmask_test_in, X_test_in, test_files_in)
        X_test_ood, test_files_ood = filter_nn(nnmask_test_ood, X_test_ood, test_files_ood)
    ## Plot Word Counts
    fig, ax = plot_matrices(x=[X_train, X_test_in, X_test_ood],
                            names=["Training", "Test (Within)", "Test (OOD)"],
                            sample_size=100,
                            transform=lambda i: np.log10(i + 1e-10),
                            cbar_label="Word Count (log-10)")
    fig.savefig(f"{args.output_folder}matrix_bow.png", dpi=300)
    plt.close(fig)
    ## Training Labels
    LOGGER.info("Compiling Ground Truth Labels")
    y_train = (np.array([train_labels[f] for f in train_files]) != "control").astype(int)
    y_test_in =  (np.array([test_labels_in[f] for f in test_files_in]) != "control").astype(int)
    ## Compute BOW Summary Stats
    LOGGER.info("Computing BOW Summary Stats")
    summary_X_train = summarize_bow(X_train, model)
    summary_X_test_in = summarize_bow(X_test_in, model)
    summary_X_test_ood = summarize_bow(X_test_ood, model)
    ## Plot BOW Summary Stats
    LOGGER.info("Plotting BOW Summary Stats")
    fig1, fig2 = plot_bow_summary([summary_X_train, summary_X_test_in, summary_X_test_ood],
                                  ["Training", "Test (Within)", "Test (OOD)"])
    fig1.savefig(f"{args.output_folder}bow_summary.png",dpi=300)
    fig2.savefig(f"{args.output_folder}bow_probability_dist_correlation.png",dpi=300)
    plt.close(fig1); plt.close(fig2)    
    ## Transform BOW Representations
    LOGGER.info("Transforming BOW Representations")
    X_train = model.preprocessor.transform(X_train)
    X_test_in = model.preprocessor.transform(X_test_in)
    X_test_ood = model.preprocessor.transform(X_test_ood)
    ## Feature Names
    LOGGER.info("Getting Feature Names")
    feature_names = model.get_feature_names()
    feature_names = list(map(lambda f: f if not isinstance(f, tuple) else "_".join(f), feature_names))
    ## Predicted Probability Distributions
    LOGGER.info("Plotting Probability Distributions")
    fig, ax = plot_probability_distributions(model,
                                             [X_train, X_test_in, X_test_ood],
                                             ["Training","Test (Within)", "Test (OOD)"])
    fig.savefig(f"{args.output_folder}predicted_probability_distributions.png", dpi=300)
    plt.close(fig)
    ## Plot Predicted Probability Correlation vs. Tokens
    LOGGER.info("Plotting Token Count vs. Probability")
    fig, ax = plot_token_probability_correlation(model,
                                                 [summary_X_train, summary_X_test_in, summary_X_test_ood],
                                                 [X_train, X_test_in, X_test_ood],
                                                 ["Training", "Test (Within)", "Test (OOD)"])
    fig.savefig(f"{args.output_folder}token_probability_correlation.png", dpi=300)
    plt.close(fig)
    ## Plot Feature Sets
    LOGGER.info("Creating Feature Matrix Comparison Visuals")
    fig, ax = plot_matrices(x=[X_train, X_test_in, X_test_ood],
                            names=["Training", "Test (Within)", "Test (OOD)"],
                            sample_size=100,
                            transform=lambda i: i,
                            cbar_label="Feature Value",
                            scale_bounds=[1,99])
    fig.savefig(f"{args.output_folder}matrix_transformed_features.png", dpi=300)
    plt.close(fig)
    for f_type, f_ind in feature_index_map(feature_names).items():
        ## Plot Raw Matrix
        fig, ax = plot_matrices(x=[X_train[:,f_ind],X_test_in[:,f_ind],X_test_ood[:,f_ind]],
                                names=["Training", "Test (Within)", "Test (OOD)"],
                                sample_size=100,
                                transform=lambda i: i,
                                cbar_label=f"Feature Value ({f_type.title()})",
                                scale_bounds=[1,99])
        fig.savefig(f"{args.output_folder}matrix_transformed_features_{f_type}.png")
        plt.close(fig)
        ## Plot Distribution
        (fig, ax), (fig2, ax2) = plot_matrix_distributions(x=[X_train[:,f_ind],X_test_in[:,f_ind],X_test_ood[:,f_ind]],
                                            names=["Training", "Test (Within)", "Test (OOD)"],
                                            bins=50,
                                            func=np.mean,
                                            stat_name=f"Mean {f_type.title()} Feature")
        fig.savefig(f"{args.output_folder}matrix_distribution_transformed_features_{f_type}.png", dpi=300)
        fig2.savefig(f"{args.output_folder}feature_correlation_{f_type}.png", dpi=300)
        plt.close(fig)
        plt.close(fig2)
    ## Shap Explainer
    LOGGER.info("Computing Shap Values")
    explainer = shap.LinearExplainer(model.model,
                                     X_train)
    ## Compute Shap Values
    shap_train = explainer.shap_values(X_train)
    shap_test_in = explainer.shap_values(X_test_in)
    shap_test_ood = explainer.shap_values(X_test_ood)
    ## Shap Summary Plots
    LOGGER.info("Plotting Shap Values")
    for x, sv, lbl in zip([X_train, X_test_in, X_test_ood],
                          [shap_train, shap_test_in, shap_test_ood],
                          ["train","test_in","test_ood"]):
        shap_fig = plot_shap(x, sv, feature_names, 50)
        shap_fig.savefig(f"{args.output_folder}shap_summary_{lbl}.png", dpi=300)
        plt.close(shap_fig)
    ## Shap Value Comparison Over Features and Dataset Shift
    LOGGER.info("Starting Dataset Comparsion by Shap and H-Distance (optionally)")
    for (s1, x1, y1), (s2, x2, y2), sname, slbl in zip([(shap_train, X_train, y_train), (shap_train, X_train, y_train), (shap_test_in, X_test_in, y_test_in)],
                                         [(shap_test_in, X_test_in, y_test_in), (shap_test_ood, X_test_ood, None), (shap_test_ood,X_test_ood, None)],
                                         ["train-test_in","train-test_ood","test_in-test_ood"],
                                         [("Training","Test (Within)"), ("Training","Test (OOD)"), ("Test (Within)","Test (OOD)")]):
        ## Shap Value Comparison
        (shap_fig, shap_ax), shap_q_df = plot_shap_comparison(s1,
                                                              s2,
                                                              feature_names,
                                                              xlabel=slbl[0],
                                                              ylabel=slbl[1],
                                                              include_error=False
                                                              )
        shap_fig.savefig(f"{args.output_folder}shap_comparison_{sname}.png", dpi=300)
        plt.close(shap_fig)
        shap_q_df.to_csv(f"{args.output_folder}shap_comparison_{sname}.csv")
        ## Model Dataset Shift
        if args.model_shift:
            covariate_shift_features = model_dataset_shift(x1,
                                                        x2,
                                                        y1,
                                                        y2,
                                                        feature_names,
                                                        balance=True,
                                                        n1=slbl[0],
                                                        n2=slbl[1])
            covariate_shift_features["task"] = model.model.coef_[0]
            covariate_shift_features.to_csv(f"{args.output_folder}covariate_shift_{sname}.csv")
    LOGGER.info("Script Complete!")

#####################
### Exectute
#####################

if __name__ == "__main__":
    _ = main()