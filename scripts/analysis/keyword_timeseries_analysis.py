
#####################
### Configuration
#####################

## Location of Data
DATA_DIR = "./plots/reddit/2008-2020/keywords_subreddits/"
DATA_DIR = "./plots/twitter/2018-2020/keywords/"

## Date Boundaries
DATA_START = "2019-01-01"
COVID_START = "2020-03-01"

## Threshold
MIN_MATCHES = 150
PLOT_SUBREDDIT = False

#####################
### Imports
#####################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pandas.plotting import register_matplotlib_converters
_ = register_matplotlib_converters()

#####################
### Data Loading
#####################

## Load Posts Per Day
posts_per_day = pd.read_csv(f"{DATA_DIR}posts_per_day.csv", index_col=0)["num_posts"]
posts_per_day.index = pd.to_datetime(posts_per_day.index)
posts_per_day = posts_per_day.loc[posts_per_day.index >= pd.to_datetime(DATA_START)]

## Aggregate by Week and Month
posts_per_week = posts_per_day.resample("W-Mon").sum()
posts_per_month = posts_per_day.resample("MS").sum()

## Keywords/Subreddits Files
keyword_files = [("CLSP Mental Health Terms", "matches_per_day_CLSP_Mental_Health_Terms.csv", False),
                 ("COVID-19 Terms", "matches_per_day_COVID-19_Terms.csv", False),
                 ("SMHD Mental Health Terms", "matches_per_day_SMHD_Mental_Health_Terms.csv", False),
                 ("JHU Crisis Terms", "matches_per_day_JHU_Crisis_Terms.csv", False)]       
subreddit_files = [("Mental Health Subreddits", "matches_per_day_Mental_Health_Subreddits.csv", True),
                   ("COVID-19 Subreddits", "matches_per_day_COVID-19_Subreddits.csv", True)]
if PLOT_SUBREDDIT:
    keyword_files.extend(subreddit_files)

## Cycle Through Keywords
for k, kf, ksub in keyword_files:
    ## Load File (Daily)
    kf_df = pd.read_csv(f"{DATA_DIR}{kf}", index_col=0)
    kf_df.index = pd.to_datetime(kf_df.index)
    ## Isolate by Start Date
    kf_df = kf_df.loc[kf_df.index >= pd.to_datetime(DATA_START)]
    ## Create Aggregation by Week
    kf_df_weekly = kf_df.resample('W-Mon').sum()
    ## Create Aggregation by Month
    kf_df_monthly = kf_df.resample("MS").sum()
    ## Posts by Period
    pre_covid_matched_posts = kf_df.loc[kf_df.index < pd.to_datetime(COVID_START)].sum(axis=0)
    pre_covid_posts = posts_per_day.loc[posts_per_day.index < pd.to_datetime(COVID_START)].sum()
    post_covid_matched_posts = kf_df.loc[kf_df.index >= pd.to_datetime(COVID_START)].sum(axis=0)
    posts_covid_posts = posts_per_day.loc[posts_per_day.index >= pd.to_datetime(COVID_START)].sum()
    ## Isolate By Threshold
    good_cols = kf_df.sum(axis=0).loc[kf_df.sum(axis=0) > MIN_MATCHES].index.tolist()
    kf_df = kf_df[good_cols].copy()
    ## Relative Posts by Period
    pre_covid_prop_posts = pre_covid_matched_posts / pre_covid_posts
    post_covid_prop_posts = post_covid_matched_posts / posts_covid_posts
    period_prop_change = post_covid_prop_posts - pre_covid_prop_posts
    period_pct_change = (period_prop_change / pre_covid_prop_posts).dropna().sort_values() * 100
    period_prop_change = period_prop_change.loc[good_cols]
    period_pct_change = period_pct_change.loc[good_cols]
    period_pct_change = period_pct_change.loc[period_pct_change!=np.inf]
    ## Create Summary Plot
    fig, ax = plt.subplots(2, 2, figsize=(12,8), sharex=False, sharey=False)
    ## Matches Over Time
    ax[0][0].plot(pd.to_datetime(kf_df_weekly.index),
                  kf_df_weekly.sum(axis=1) / posts_per_week,
                  linewidth=2,
                  color="C0",
                  alpha=.7,
                  marker="o")
    ax[0][0].axvline(pd.to_datetime(COVID_START),
                     linestyle="--",
                     linewidth=2,
                     color="black",
                     alpha=0.5,
                     label="COVID-19 Start ({})".format(COVID_START))
    xticks = [i for i in kf_df_monthly.index if (i.month - 1) % 4 == 0]
    ax[0][0].set_xticks(xticks)
    ax[0][0].set_xticklabels([i.date() for i in xticks], rotation=25, ha="right")
    ax[0][0].legend(loc="upper left", frameon=True, framealpha=1)
    ax[0][0].set_ylabel("Matches Per Post (Weekly)", fontweight="bold")
    ax[0][0].set_xlabel("Week", fontweight="bold")
    ## Overall Match Rate Per Week
    ax[0][1].hist(kf_df_weekly.sum(axis=1) / posts_per_week,
                  bins=15,
                  label="$\\mu={:.2f}, \\sigma={:.3f}$".format(
                      (kf_df_weekly.sum(axis=1) / posts_per_week).mean(),
                      (kf_df_weekly.sum(axis=1) / posts_per_week).std()
                  ),
                  alpha=.7)
    ax[0][1].set_ylabel("# Weeks", fontweight="bold")
    ax[0][1].set_xlabel("Matches Per Post (Weekly)", fontweight="bold")
    ax[0][1].legend(loc="upper right", frameon=True, facecolor="white", framealpha=1)
    ## Largest Proportional Differences
    nplot = min(10, int(len(period_prop_change)/2))
    plot_data = (period_prop_change.nlargest(nplot).append(period_prop_change.nsmallest(nplot))).sort_values()
    values = list(plot_data.values[:nplot]) + [0] +  list(plot_data.values[nplot:])
    ax[1][0].barh(list(range(nplot*2 + 1)),
                  values,
                  color = list(map(lambda i: "darkred" if i <= 0 else "navy", values)),
                  alpha = 0.6)
    ax[1][0].set_yticks(list(range(nplot*2 + 1)))
    ax[1][0].set_yticklabels(list(plot_data.index[:nplot]) + ["..."] + list(plot_data.index[nplot:]),
                             ha="right", va="center")
    ax[1][0].set_ylim(-.5, nplot*2 + .5)
    ax[1][0].axvline(0, color="black", linestyle="--", alpha=0.5)
    if not ksub:
        ax[1][0].set_ylabel("Term", fontweight="bold")
    else:
        ax[1][0].set_ylabel("Subreddit", fontweight="bold")
    fmt = ticker.ScalarFormatter()
    fmt.set_powerlimits((-2,2))
    ax[1][0].xaxis.set_major_formatter(fmt)
    ax[1][0].set_xlabel("Absolute Change\n(Pre- vs. Post COVID-19 Start)", fontweight="bold")
    ax[1][0].ticklabel_format(axis="x", style="sci")
    ## Largest Percent Differences
    nplot = min(10, int(len(period_pct_change)/2))
    plot_data = (period_pct_change.nlargest(nplot).append(period_pct_change.nsmallest(nplot))).sort_values()
    values = list(plot_data.values[:nplot]) + [0] +  list(plot_data.values[nplot:])
    ax[1][1].barh(list(range(nplot*2 + 1)),
                  values,
                  color = list(map(lambda i: "darkred" if i <= 0 else "navy", values)),
                  alpha = 0.6)
    ax[1][1].set_yticks(list(range(nplot*2 + 1)))
    ax[1][1].set_yticklabels(list(plot_data.index[:nplot]) + ["..."] + list(plot_data.index[nplot:]),
                             ha="right", va="center")
    ax[1][1].set_ylim(-.5, nplot*2 + .5)
    ax[1][1].axvline(0, color="black", linestyle="--", alpha=0.5)
    if not ksub:
        ax[1][1].set_ylabel("Term", fontweight="bold")
    else:
        ax[1][1].set_ylabel("Subreddit", fontweight="bold")
    ax[1][1].set_xlabel("Percent Change\n(Pre- vs. Post COVID-19 Start)", fontweight="bold")
    fig.tight_layout()
    fig.suptitle(k, fontweight="bold", fontsize=14, y=.98)
    fig.subplots_adjust(top=.94)
    fig.savefig("{}summary_{}.png".format(DATA_DIR, k.replace(" ","_").lower()), dpi=300)
    plt.close(fig)
    