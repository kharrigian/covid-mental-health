
####################
### Configuration
####################

## Result Directory
RESULTS_DIR = "./data/results/reddit/2017-2020/inference/monthly-weekly_step/"
# RESULTS_DIR = "./data/results/twitter/2018-2020/inference/monthly-weekly_step/"

## Plot Directory
PLOT_DIR = "./plots/reddit/2017-2020/inference/monthly-weekly_step/"
# PLOT_DIR = "./plots/twitter/2018-2020/inference/monthly-weekly_step/"

## Metadata
FREQUENCY = "weekly"
PLATFORM = "reddit"
CONDITION = "anxiety"

## Parameters
POS_THRESHOLD = 0.9
MIN_POSTS_PER_WINDOW = 10
MIN_TOKENS_PER_WINDOW = 25
COVID_START = "2020-02-01"
CACHE_CONCATENATION=True

####################
### Imports
####################

## Standard Library
import os
import sys
import gzip
import json
from glob import glob
from datetime import datetime

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from fbprophet import Prophet
from statsmodels.api import tsa
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

## Mental Health
from mhlib.util.logging import initialize_logger

####################
### Globals
####################

## Initialize Logger
LOGGER = initialize_logger()

## Date Converters for Plotting
_ = register_matplotlib_converters()

## Establish Plot Directory
if not os.path.exists(PLOT_DIR):
    _ = os.makedirs(PLOT_DIR)

####################
### Helpers
####################

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

def cache_results(predictions,
                  support,
                  tokens,
                  unique_tokens,
                  date_ranges):
    """

    """
    ## Update Indices
    predictions.index = predictions.index.map(os.path.basename).str.rstrip(".json.gz")
    support.index = support.index.map(os.path.basename).str.rstrip(".json.gz")
    tokens.index = tokens.index.map(os.path.basename).str.rstrip(".json.gz")
    unique_tokens.index = unique_tokens.index.map(os.path.basename).str.rstrip(".json.gz")
    ## Date Range Format
    date_ranges = pd.DataFrame(date_ranges, index=["start","stop"]).T
    date_ranges.index.name = "date_label"
    ## Write Data
    outdir = f"{RESULTS_DIR}{PLATFORM}.{CONDITION}.results/"
    if not os.path.exists(outdir):
        _ = os.makedirs(outdir)
    for df, df_name in zip([predictions,support,tokens,unique_tokens,date_ranges],
                           ["predictions","num_posts","num_tokens","num_unique_tokens","date_ranges"]):
        df.to_csv(f"{outdir}{CONDITION}.{df_name}.csv")

####################
### Load Predictions
####################

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

## Data Caching
if CACHE_CONCATENATION:
    _ = cache_results(predictions,
                      support,
                      tokens,
                      unique_tokens,
                      date_ranges)

## Apply Activity Thresholds
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

## Binarize Cleaned Predictions
LOGGER.info("Binarizing Predictions")
predictions_binary = predictions_filtered.applymap(lambda x: x > POS_THRESHOLD if not pd.isnull(x) else np.nan)

####################
### Visualize Population Level
####################

## Population-level Bootstrap
LOGGER.info("Running Population-level Bootstrap (Average)")
pred_CI = bootstrap_sample(predictions_filtered.values,
                           func=np.nanmean,
                           axis=0,
                           sample_percent=30,
                           samples=1000)
pred_CI = pd.DataFrame(pred_CI.T,
                       index=dates,
                       columns=["lower","median","upper"]).astype(float)
pred_CI["n"] = (~predictions_filtered.isnull()).sum(axis=0)

## Population-Level Binary
LOGGER.info("Running Population-level Bootstrap (Binary)")
pred_CI_binary = bootstrap_sample(predictions_binary.values,
                                  func=lambda x, axis: np.nansum(x, axis) / np.sum(~pd.isnull(x), axis),
                                  axis=0,
                                  sample_percent=30,
                                  samples=250)
pred_CI_binary = pd.DataFrame(pred_CI_binary.T,
                              index=dates,
                              columns=["lower","median","upper"]).astype(float)
pred_CI_binary["n"] = (~predictions_binary.isnull()).sum(axis=0)

## Visualize Population-Level Predictions
LOGGER.info("Visualizing Bootstrap Results")
for CI, CI_name, ylbl in zip([pred_CI, pred_CI_binary],
                             ["population_level","population_level_binary"],
                             [f"Mean Pr({CONDITION.title()})", f"Percent Pr({CONDITION.title()}) > {POS_THRESHOLD}"]):
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.axvline(pd.to_datetime("2020-03-01"),
               linestyle="--",
               color="black",
               linewidth=3,
               alpha=.9,
               label="COVID-19 Spike in US (March 1, 2020)")
    ax.fill_between(CI.index,
                    CI["lower"].astype(float).values,
                    CI["upper"].astype(float).values,
                    color="C0",
                    alpha=0.3)
    ax.plot(CI.index,
            CI["median"].astype(float),
            marker="o",
            markersize=5,
            color="C0",
            linewidth=2,
            linestyle="-",
            alpha=.8,
            label="Population Average")
    ax2 = ax.twinx()
    ax2.plot(CI.index,
             CI["n"].values.astype(float),
             marker="o",
             color="C1",
             linewidth=2,
             linestyle="--",
             alpha=.5,
             label="# Users Modeled")
    ax2.set_ylabel("# Users Modeled",
                   fontweight="bold")
    ax.set_xlabel("Date Start", fontweight="bold")
    ax.set_ylabel(ylbl, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}inferences_{CI_name}_{CONDITION}.png", dpi=300)
    plt.close()

####################
### Threshold Visualization
####################

LOGGER.info("Analyzing Results ~ Binarization Threshold")

## Compute Summary Statistics ~ Threshold
binary_threshold_CI = []
thresholds = np.linspace(0, 1, 11)[1:-1]
for t in tqdm(thresholds, "Binarization Threshold"):
    t_df = (predictions_filtered.applymap(lambda x: int(x > t) if not pd.isnull(x) else np.nan))
    t_df_ci = bootstrap_sample(t_df.values,
                               func=lambda x, axis: np.nansum(x, axis) / np.sum(~pd.isnull(x), axis),
                               axis=0,
                               sample_percent=30,
                               samples=250)
    t_df_ci = pd.DataFrame(t_df_ci.T, index=dates, columns=["lower","median","upper"]).astype(float)
    t_df_ci["threshold"] = t
    binary_threshold_CI.append(t_df_ci.reset_index().rename(columns={"index":"date"}))
binary_threshold_CI = pd.concat(binary_threshold_CI)

## Visualize Distributions (Line-plot)
fig, ax = plt.subplots(figsize=(10,5.8))
for t, threshold in enumerate(thresholds):
    plot_df = binary_threshold_CI.loc[binary_threshold_CI["threshold"]==threshold]
    ax.fill_between(plot_df["date"],plot_df["lower"],plot_df["upper"],color="C0",alpha=threshold)
    ax.plot(plot_df["date"],plot_df["median"],color="C0",alpha=threshold, linestyle="--", marker="o")
ax.set_xlabel("Date",
              fontweight="bold")
ax.set_ylabel("Proportion of Population Predicted w/ {}".format(CONDITION.title()),
              fontweight="bold")
ax.set_ylim(bottom=0)
ax.axvline(pd.to_datetime(COVID_START), color="black", linestyle="--", alpha=.8, label="COVID-19 Start")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}inferences_binary_threshold_timeseries_{CONDITION}.png", dpi=300)
plt.close()

## Visualize Distributions (Heat Map)
binary_threshold_median = binary_threshold_CI.pivot(index="threshold",columns="date",values=["median"])["median"].iloc[::-1]
binary_threshold_dev = binary_threshold_median.apply(lambda row: row - row.mean(), axis=1)
fig, ax = plt.subplots(figsize=(10,5.8))
vmin = binary_threshold_dev.min().min()
vmax = binary_threshold_dev.max().max()
norm = colors.TwoSlopeNorm(vmin=vmin,
                            vcenter=0,
                            vmax=vmax)
m = ax.imshow(binary_threshold_dev.values,
              interpolation="nearest",
              cmap=plt.cm.coolwarm,
              alpha=0.5,
              aspect="auto",
              norm=norm)
ax.set_yticks(range(binary_threshold_median.shape[0]))
ax.set_yticklabels(["{:.2f}".format(i) for i in binary_threshold_median.index])
ax.set_xticks(list(range(binary_threshold_median.shape[1])))
ax.set_xticklabels(list(map(lambda i, val: val.date() if i % 4 == 0 else "",
                   range(binary_threshold_median.shape[1]),
                   binary_threshold_median.columns)),
                   rotation=45, ha="right")
ax.axvline(np.nonzero(binary_threshold_median.columns >= pd.to_datetime(COVID_START))[0][0]-.5,
           color="black")
cbar = fig.colorbar(m)
cbar.set_label("Deviation From Mean", fontweight="bold")
ax.set_xlabel("Date",
              fontweight="bold")
ax.set_ylabel("Binarization Threshold",
              fontweight="bold")
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}inferences_binary_threshold_heatmap_{CONDITION}.png", dpi=300)
plt.close()

####################
### Support/Prediction Correlation
####################

LOGGER.info("Examing Prediction vs. Support Correlation")

## Unravel Values
p = predictions_filtered.values.ravel()
s = support_filtered.values.ravel()
t = tokens_filtered.values.ravel()
ut = unique_tokens_filtered.values.ravel()

## Format Values, Scaling Support Values
d = pd.DataFrame(np.vstack([s,t,ut,p]).T,
                 columns=["Posts","Tokens","Unique Tokens","Predicted Probability"]).dropna()
for col in ["Posts","Tokens","Unique Tokens"]: d[col] = d[col].map(np.log)

## Create Plot
f = sns.pairplot(d.sample(frac=0.1, replace=False, random_state=42))
f.fig.tight_layout()
f.fig.savefig(f"{PLOT_DIR}inferences_probability_support_correlation_{CONDITION}.png", dpi=300)
plt.close(f.fig)

####################
### Timeseries Modeling (Prophet)
####################

## Parameters
n_models = 20
sample_percent = 0.7
replace = True

## Update User
LOGGER.info("Starting Prophet Forecast")

## Bootstrap Fit Procedure
forecasts = []
for n in range(n_models):
    ## Update User on Progress
    LOGGER.info("~"*50 + f"\nStarting Forecast {n+1}/{n_models}\n" + "~"*50)
    ## Sample Data (Bootstrap Formulation)
    pred_sample = predictions_binary.sample(frac=sample_percent, replace=replace)
    pred_sample = pred_sample.apply(lambda x: np.nansum(x) / sum(~pd.isnull(x)), axis=0)
    pred_sample = pred_sample.to_frame("y").reset_index().rename(columns={"index":"ds"})
    pred_sample["ds"] = pd.to_datetime(pred_sample["ds"])
    pred_sample_train = pred_sample.loc[pred_sample["ds"] < pd.to_datetime(COVID_START)]
    ## Build Model
    p = Prophet(growth="linear",
                changepoints=None,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode="additive",
                interval_width=.95)
    # p.add_seasonality(
    #     name='monthly', 
    #     period=30.5, 
    #     fourier_order=5
    # )
    ## Fit Model
    p.fit(pred_sample_train)
    ## Forecast and Cache
    sample_forecast = p.predict(pred_sample[["ds"]])
    sample_forecast["sample_n"] = n
    sample_forecast = pd.merge(sample_forecast, pred_sample, on =["ds"])
    forecasts.append(sample_forecast)
## Concatenate Forecasts
forecasts = pd.concat(forecasts)

## Raw Values
LOGGER.info("Computing Bootstrap Intervals (Raw Data)")
pred_raw = bootstrap_sample(predictions_binary.values,
                            func=lambda x, axis: np.nansum(x, axis) / np.sum(~pd.isnull(x), axis),
                            axis=0,
                            sample_percent=sample_percent*100,
                            samples=1000)
pred_raw = pd.DataFrame(pred_raw.T, columns=["lower","median","upper"]).astype(float)
pred_raw.index = pd.to_datetime(predictions_binary.columns)

## Plot
LOGGER.info("Visualizing Forecast")
fig, ax = plt.subplots(figsize=(10,8))
for sample in range(n_models):
    sample_f = forecasts.loc[forecasts["sample_n"] == sample]
    ax.fill_between(sample_f["ds"], sample_f["yhat_lower"], sample_f["yhat_upper"], color="C0", alpha=min(1 / n_models, 0.5))
    ax.plot(sample_f["ds"], sample_f["yhat"], color="C0", alpha=0.8, linestyle=":", label="Prophet Forecast" if sample == 0 else "")
ax.fill_between(pred_raw.index, pred_raw["lower"], pred_raw["upper"], color="C1", alpha=0.4)
ax.plot(pred_raw.index, pred_raw["median"], color="C1", linewidth=2, alpha=0.8, linestyle="-", label="Measurement")
ax.axvline(pd.to_datetime(COVID_START), color="black", linestyle="--", alpha=1, linewidth=3, label="Training Boundary ({})".format(COVID_START))
ax.set_ylabel(f"Percent Pr({CONDITION.title()}) > {POS_THRESHOLD}", fontweight="bold", fontsize=16)
ax.set_xlabel("Date", fontweight="bold", fontsize=16)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Predicted Level of {} on {} ({})".format(CONDITION.title(), PLATFORM.title(), FREQUENCY.title()),
             loc="left",
             fontweight="bold",
             fontstyle="italic",
             fontsize=18)
ax.tick_params(labelsize=14)
leg = ax.legend(loc="upper left", frameon=True, fontsize=14)
fig.autofmt_xdate()
ax.set_xlim(pred_raw.index.min(), pred_raw.index.max())
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}inferences_population_level_prophet_{CONDITION}.png", dpi=300)
plt.close()

####################
### Visualize Individual Level
####################

# ## Bin Predictions
# bin_function = lambda x: int(np.floor(x * 10)) if not pd.isnull(x) else np.nan
# predictions_binned = predictions.applymap(bin_function)

# ## Compute Transition Matrices
# tm = {}
# for start, end in zip(predictions_binned.columns[:-1], predictions_binned.columns[1:]):
#     x = predictions_binned[[start, end]].rename(columns={start:"start",end:"end"})
#     x = x.dropna().astype(int)
#     tm_ = metrics.confusion_matrix(x["start"], x["end"], labels=list(range(10)))
#     tm[(start, end)] = tm_

# ## Plot Transition Matrix
# fig, ax = plt.subplots(1, len(tm))
# for i, ((start, end), matrix) in enumerate(tm.items()):
#     ax[i].imshow(np.log(matrix),
#                  cmap=plt.cm.Purples,
#                  aspect="auto")
#     ax[i].set_ylabel(start, fontsize=6)
#     ax[i].set_xlabel(end, fontsize=6)
# fig.tight_layout()
# plt.savefig("test.png", dpi=300)
# plt.close