# Mental Health (COVID-19 Research using Social Media)

This respository exists to examine changes in mental health status due to the COVID-19 pandemic. It focuses on analysis of social media data, in particular data from Twitter and Reddit. The codebase is built assuming access to the JHU CLSP grid and requires some dependencies for modeling mental health status and user demographics; these are noted where applicable.

## Data Collection

As of July 1, 2020 this codebase supports analysis of Reddit and Twitter data. Below, we outline how user samples are constructed.

#### Reddit

1. Identify all users who posted comments in 46 of the most popular subreddits between 5/25/2020 and 6/01/2020. This search yields ~1.2 million unique Reddit users. This process is housed in `scripts/acquire/reddit/count_users.py`

2. For each of the ~1.2 million users identified, count the number of comments they made each week during January 2019. Ignoring clear bots and moderator accounts, filter down the cohort to users who posted at least 1 comment per week during the month and filter out the top 1% of active users. This process yields ~200k unique users and is housed in `scripts/acquire/reddit/sample_cohort.py`

3. For each filtered user, query their entire comment history available from January 1, 2008 to June 20, 2020. This process is housed in `scripts/acquire/reddit/retrieve_histories_api.py` and assumes access to the Pushshift.io database.

4. For each user, use their entire comment history to make geolocation inferences via the `smgeo` python package. Code here assumes access to the `smgeo` repository, models, and training data. The process is housed in `scripts/acquire/reddit/infer_geolocation.py`.

#### Twitter

1. Identify files containing the gardenhose stream of Twitter (i.e. 1% sample). Indentify files containing data from January 1, 2019 through June 18, 2020. This process uses code in `scripts/acquire/twitter/identify_filelist.py` and assumes access to Mark Dredze's sample housed on the CLSP grid at `/export/c12/mdredze/twitter/public/`.

2. Count the number of English (based on "lang" attribute), non-retweet tweets made per user, per day across the sample noted in Step 1. This process is housed in `scripts/acquire/twitter/count_users.py` and identifies ~41 million unique users across the time span.

3. Amongst the ~41 million users identifed in Step 2, identify the subset who posted at least 5 English, non-retweet tweets each month between January 1, 2019 and June 18, 2020. This only considers tweets within the 1% sample housed on the CLSP grid. We further filter out the top 5% most active users across the entire time span. This process is housed in `scripts/acquire/twitter/sample_cohort.py` and yields ~19k users.

4. For all users identified in the cohort from Step 3, find all tweets within the 1% sample these users have made between January 1, 2018 and June 18, 2020. This includes retweets and non-English tweets. This is a two step process that begins using `scripts/acquire/twitter/retrieve_timelines_sample.py` and then executes `scripts/acquite/twitter/concatenate_timelines.py`; the first script finds the subset of tweets made by users in the cohort and then the second script concatenates tweets for each user into single files per user.

5. [TODO] Use the Carmen geolocation tool to assign home locations to all users in the sample cohort based on their raw tweet data contained in the 1% Twitter sample.

## Data Processing

We tokenize all Twitter tweets and Reddit comments using code from the `mhlib` python package. The process is housed in `scripts/model/preprocess.py`. Within the preprocessed data files, we maintain only a subset of the raw data objects. Outputs are stored in `data/processed/twitter/timelines/` and `data/processed/reddit/histories/` as gzipped JSON files.

## Mental Health Status Inference

To more robustly infer mental health status, we use machine learning models trained to detect various disorders using natural language. Models are trained using code housed in the `mhlib` research repository and rely on three different datasets. For application to Twitter data, we use models trained on the "Multi-Task Learning for Mental Health" dataset from Benton et al. For application to Reddit data, we use models trained on the "SMHD" dataset from Cohan et al. and the "Topic-Restricted Text" dataset from Wolohan et al. (note: we only examine depression using the latter dataset).

We use logistic regression as the estimator for all models. As features, we ingest TF-IDF representations of unigrams, a 64-dimensional LIWC distribution, a 50-dimensional LDA topic distribution, and a 200-dimensional mean-pooled GloVe representation. Hyperparameters (regularization, class-balancing, and feature scaling) are selected that maximize F1 in a 20% held-out sample of the training data. All unigrams must have occurred in at least 10 user samples during training to be kept in the vocabulary.

All models are trained to make a binary decision on whether a social media user lives with a particular disorder. Each dataset we consider uses proxy-based annotation (e.g. self-disclosures of mental health status, community-based labeling). Please consult the original reference material for a full overview. We have access to models trained to detect the following disorders (see `/export/fs03/a08/kharrigian/mental-health/models/falconet/` on the CLSP grid):

**Reddit**

* Anxiety
* Bipolar Disorder
* Depression
* Eating Disorder
* PTSD
* Schizophrenia

**Twitter**

* Anxiety
* Bipolar Disorder
* Borderline Personality Disorder
* Depression
* Eating Disorder
* Panic Disorder
* PTSD
* Schizophrenia
* Suicidal Ideation
* (Past) Suicide Attempt

Application of these models is done using code in `scripts/model/infer.py`. Results for multiple timepoints are analyzed using code in `scripts/model/process_inferences.py`. To schedule multiple inference jobs over a regularly-spaced timespan on the CLSP grid, one can use `scripts/model/schedule_infer.py`.