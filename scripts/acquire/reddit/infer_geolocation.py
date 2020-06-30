
"""
Infer location for a list of reddit users using a pretrained inference model.
"""

## Script Configuration
SETTINGS_FILE = "../smgeo/configurations/settings.json"
RAW_DATA_DIR = "./data/raw/reddit/histories/"
MODEL_FILE = "../smgeo/models/reddit/Global_TextSubredditTime/model.joblib"
SMGEO_DIR = "../smgeo/"
NUM_JOBS = 8

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

## External Libraries
import joblib
from tqdm import tqdm
import pandas as pd
from scipy.sparse import vstack
import reverse_geocoder as rg # pip install reverse_geocoder

## Local Modules
from smgeo.acquire.reddit import RedditData
from smgeo.model import preprocess
from smgeo.util.logging import initialize_logger
from mhlib.util.multiprocessing import MyPool as Pool

#######################
### Globals
#######################

## Logger
LOGGER = initialize_logger()

## Model
LOGGER.info("Loading Geolocation Inference Model")
MODEL = joblib.load(MODEL_FILE)

#######################
### Functions
#######################

def parse_command_line():
    """

    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Infer home location of reddit users using a pretrained model.")
    ## Required Arguments
    parser.add_argument("output_csv",
                        help="Path for saving the predictions. Should be a csv file.")
    ## Optional Arguments
    parser.add_argument("--grid_cell_size",
                        default=.5,
                        type=float,
                        help="Coordinate grid cell size in degrees.")
    parser.add_argument("--posterior",
                        default=False,
                        action="store_true",
                        help="If this flag specified, the posterior across all coordinates will be included.")
    parser.add_argument("--reverse_geocode",
                        default=False,
                        action="store_true",
                        help="If this flag specified, the argmax of predictions will be reverse geocoded")
    parser.add_argument("--known_coordinates",
                         default=False,
                         action="store_true",
                         help="If specified, code will try to load in known training coordinates to restrict inference search space.")
    parser.add_argument("--max_docs",
                        default=None,
                        type=int,
                        help="Maximium number of documents to user for inference. Sorted by recency.")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check That Required Files Exist
    if not args.output_csv.endswith(".csv"):
        raise TypeError("Expected output_csv to be a .csv file")
    return args

def load_settings():
    """

    """
    LOGGER.info("Loading Settings")
    if not os.path.exists(SETTINGS_FILE):
        raise FileNotFoundError(f"Could not find setting file in expected location: {settings_file}")
    with open(SETTINGS_FILE, "r") as the_file:
        settings_config = json.load(the_file)
    return settings_config

def load_users():
    """

    """
    LOGGER.info("Loading User List")
    user_data_paths = glob(f"{RAW_DATA_DIR}/*.json.gz")
    return user_data_paths

def _prepare_data(user_file):
    """

    """
    ## Load Raw Data
    user_data = []
    with gzip.open(user_file, "r") as the_file:
        for line in the_file:
            user_data.append(json.loads(line))
    ## Apply Data Preprocessing
    user_data_list = preprocess.process_reddit_comments(user_data)
    ## Data Filtering
    user_data_list = MODEL._vocabulary._select_n_recent_documents(user_data_list)
    user_data_list = MODEL._vocabulary._select_first_n_tokens(user_data_list)
    ## Add Number of Comments to Cache
    ## Count
    user_data_counts = MODEL._vocabulary._count(user_data_list)
    ## Vectorize
    user_X = MODEL._vocabulary._vectorize_user_data(user_data_counts)
    n = len(user_data_list)
    return user_file, (user_X, n)

def prepare_data(user_data_paths,
                 jobs=NUM_JOBS):
    """
    Format raw comment data into vectorized format

    Args:
        model (GeolocationInferenceModel): Trained inference model
        user_data_paths (list): List of raw data files
    
    Returns:
        X (csr_matrix): Feature matrix for application users
        n (list): Comment counts associated with each user
    """
    ## Use Multiprocessing to Process Data
    mp = Pool(jobs)
    res = list(tqdm(mp.imap_unordered(_prepare_data, user_data_paths),
                    desc="Vectorizing User Data",
                    file=sys.stdout,
                    total=len(user_data_paths)))
    mp.close()
    ## Parse Results and Stack
    user_data_paths = [r[0] for r in res]
    X = vstack([r[1][0] for r in res]).tocsr()
    n = [r[1][1] for r in res]
    return user_data_paths, X, n

def load_known_coordinates(settings):
    """
    Load coordinates used for training the original model (e.g. known cities)

    Args:
        settings (dict): Repository-wide settings (e.g. data paths)
    
    Returns:
        coordinates (2d-array): [Lon, Lat] coordinates to use for prediction
    """
    LOGGER.info("Loading Known Coordinates")
    author_training_file = "{}author_labels.json.gz".format(SMGEO_DIR + settings["reddit"]["LABELS_DIR"])
    if not os.path.exists(author_training_file):
        raise FileNotFoundError(f"Could not identify training data for loading known coordinates at: {author_training_file}. \
                                  Check placement of the label data or turn off the --known_coordinates flag.")
    labels = pd.read_json(author_training_file)
    coordinates = labels[["longitude","latitude"]].drop_duplicates().values
    return coordinates

def reverse_search(coordinates):
    """
    Use the Geonames Database to Reverse Search Locations based on Coordinates

    Args:
        coordinates (2d-array): [Lon, Lat] values
    
    Returns:
        result (list of dict): Reverse search results
    """
    result = rg.search(list(map(tuple,coordinates[:,::-1])))
    return result

def main():
    """

    """
    ## Parse Command Line Arguments
    args = parse_command_line()
    ## Load Settings
    settings = load_settings()
    ## Load User List
    user_data_paths = load_users()
    ## Load Geolocation Inference Model
    MODEL._vocabulary._max_docs = args.max_docs
    ## Prepare User Data
    user_data_paths, X, n = prepare_data(user_data_paths)
    ## Create Coordinate Grid
    if not args.known_coordinates:
        coordinates = MODEL._create_coordinate_grid(args.grid_cell_size)
    else:
        coordinates = load_known_coordinates(settings)
    ## Make Predictions
    LOGGER.info("Making Inferences")
    _, P = MODEL.predict_proba(X, coordinates)
    y_pred = pd.DataFrame(index=user_data_paths,
                          data=coordinates[P.argmax(axis=1)],
                          columns=["longitude_argmax","latitude_argmax"])
    ## Reverse Geocoding
    if args.reverse_geocode:
        LOGGER.info("Reversing the Geolocation Inferences")
        reverse = reverse_search(y_pred[["longitude_argmax","latitude_argmax"]].values)
        for level, level_name in zip(["name","admin2","admin1","cc"],["city","county","state","country"]):
            level_data = [i[level] for i in reverse]
            y_pred[f"{level_name}_argmax"] = level_data
    ## Add Posterior
    if args.posterior:
        P = pd.DataFrame(P, index=user_data_paths, columns=list(map(tuple, coordinates)))
        y_pred = pd.merge(y_pred, P, left_index=True, right_index=True)
    ## Cache
    LOGGER.info("Caching Inferences")
    y_pred.to_csv(args.output_csv, index=True)
    ## Done
    LOGGER.info("Script complete.")

#######################
### Execute
#######################

if __name__ == "__main__":
    main()

