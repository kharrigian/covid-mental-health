
"""
Schedule Falconet Mental Health Pipeline
- Supports Serial or Parallel Processing of Input Files
- Supports Selection of Date Boundaries and Automatic Input Detection
"""

########################
### Configuration
########################

## Run Configuration
OUTPUT_PREFIX="falconet"
START_DATE="2018-01-01"
END_DATE="2020-08-01"
SAMPLE_RATE=0.01
RANDOM_STATE=42
DRY_RUN=False

## Pipeline Configuration
DATA_TYPE="twitter"
INPUT_FOLDER="/export/c12/mdredze/twitter/public/"
OUTPUT_FOLDER="/export/fs03/a08/kharrigian/covid-mental-health/data/results/twitter/2018-2020/falconet-full/"
CONFIG="/export/fs03/a08/kharrigian/lab-resources/falconet/pipelines/mental_health/all.json"

## Processing Parameters
MEMORY=32
NJOBS=1
PARALLEL=True
TEMP_DIR = "./temp/"

########################
### Imports
########################

## Standard Library
import os
import sys
import logging
import subprocess
from datetime import datetime
from textwrap import dedent
from glob import glob

## External
import pandas as pd
from pandas import to_datetime
import matplotlib.pyplot as plt

########################
### Helpers
########################

## Logger Initialization
def initialize_logger(level=logging.INFO):
    """
    Create a logger object for outputing
    to standard out
    Args:
        level (int or str): Logging filter level
    
    Returns:
        logger (Logging object): Python logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger

def get_size_of(filename):
    """
    Get size of a file in gigabytes
    """
    return os.path.getsize(filename) / 1e9

########################
### Globals
########################

## Globals
LOGGER=initialize_logger()

########################
### Functions
########################

def get_header(NFILES,
               NJOBS=1,
               MEMORY=8):
    """

    """
    HEADER=f"""
    #$ -cwd
    #$ -S /bin/bash
    #$ -m eas
    #$ -N {OUTPUT_PREFIX}
    #$ -t 1-{NFILES}
    #$ -e /home/kharrigian/gridlogs/python/covid_falconet_mental_health/
    #$ -o /home/kharrigian/gridlogs/python/covid_falconet_mental_health/
    #$ -pe smp {NJOBS}
    #$ -l 'gpu=0,mem_free={MEMORY}g,ram_free={MEMORY}g'
    """
    return HEADER

def get_init_env():
    """

    """
    INIT_ENV="""
    ENV="falconet"
    RUN_LOC="/export/fs03/a08/kharrigian/covid-mental-health/"
    ## Move to Home Directory (Place Where Virtual Environments Live)
    cd /home/kharrigian/
    ## Activate Conda Environment
    source .bashrc
    conda activate ${ENV}
    ## Move To Run Directory
    cd ${RUN_LOC}
    """
    return INIT_ENV

## File Finder
def get_filenames(input_folder,
                  start_date=None,
                  end_date=None):
    """

    """
    ## Get Files
    files = sorted(glob(f"{input_folder}*/*/*.gz"))
    ## Dates
    if start_date is not None:
        start_date = to_datetime(start_date)
    if end_date is not None:
        end_date = to_datetime(end_date)
    ## Filter By Date
    date_extract = lambda f: datetime(*list(map(int, os.path.basename(f).split(".")[0].split("_"))))
    files_filtered = []
    for f in files:
        fdate = date_extract(f)
        if start_date is not None and fdate < start_date:
            continue
        if end_date is not None and fdate >= end_date:
            continue
        files_filtered.append(f)
    return files_filtered

## Scheduler
def run_falconet(input_file=None,
                 nfiles=1,
                 njobs=1,
                 memory=8,
                 parallel=True):
    """

    """
    ## Variable Formatting
    if input_file is not None:
        INPUT_PREFIX = os.path.basename(input_file).split(".")[0]
    else:
        INPUT_PREFIX = "parallel"
    ## Header (In case of Parallel Processing on the Grid)
    HEADER = ""
    if parallel:
        HEADER=get_header(NFILES=nfiles,
                          NJOBS=njobs,
                          MEMORY=memory)
        HEADER=dedent(HEADER)
    ## Environment
    INIT_ENV=""
    if parallel:
        INIT_ENV=get_init_env()
        INIT_ENV=dedent(INIT_ENV)
    ## Input File
    input_command="INPUT_FILE=$(<./temp/filenames/$SGE_TASK_ID.txt)" if parallel else f"INPUT_FILE={input_file}"
    ## Full Object File Removal
    if parallel:
        TRAILER="""
    identifier={}
    identifier={}
    rm {}{}_{}_out.json.gz
    """.format(
        "${INPUT_FILE##*/}",
        "${identifier%.*}",
        OUTPUT_FOLDER,
        OUTPUT_PREFIX,
        "${identifier}"
    )
    else:
        TRAILER=f"rm {OUTPUT_FOLDER}{OUTPUT_PREFIX}_{INPUT_PREFIX}_out.json.gz"
    ## Bash Script
    script="""
    #!/bin/bash
    {}
    {}
    {}
    python -m falconet.cli.run \
        {} \
        --output_prefix {} \
        --output_folder {} \
        --pipeline_conf {} \
        --message_type {} \
        --sample_rate {} \
        --random_state {}
    {}
    """.format(
        HEADER,
        INIT_ENV,
        input_command,
        "${INPUT_FILE}",
        OUTPUT_PREFIX,
        OUTPUT_FOLDER,
        CONFIG,
        DATA_TYPE,
        SAMPLE_RATE,
        RANDOM_STATE,
        TRAILER
    )
    script = dedent(script)
    ## Write File
    filename=f"{TEMP_DIR}falconet_pipeline_{INPUT_PREFIX}.sh"
    with open(filename,"w") as the_file:
        the_file.write(script)
    ## Execute or Schedule
    if not parallel:
        command = f"bash {filename}"
        status = os.system(command)
        LOGGER.info(f"\tFinished pipeline for {input_file} with exit status {status}")
    else:
        command = f"qsub {filename}"
        job_id = subprocess.check_output(command, shell=True)
        LOGGER.info(f"\tScheduled pipeline at {job_id}")

########################
### Wrapper
########################

def main():
    """

    """
    ## Get Files
    FILENAMES = get_filenames(INPUT_FOLDER,
                              start_date=START_DATE,
                              end_date=END_DATE)
    n = len(FILENAMES)
    LOGGER.info("Found {} files".format(n))
    ## Dry Run -- Exit Early
    if DRY_RUN:
        exit()
    ## Setup Temporary Folders
    if not os.path.exists(TEMP_DIR):
        _ = os.makedirs(TEMP_DIR)
    if PARALLEL and not os.path.exists(f"{TEMP_DIR}filenames/"):
        _ = os.makedirs(f"{TEMP_DIR}filenames/")
    ## Save Data Filenames to Individual Files for Parallel Processing
    if PARALLEL:
        for f, filename in enumerate(FILENAMES):
            with open(f"{TEMP_DIR}filenames/{f+1}.txt","w") as the_file:
                the_file.write(filename)
    ## Get Expected Sizes
    filesizes = pd.DataFrame([{"filename":f, "size":get_size_of(f)} for f in FILENAMES])
    filesizes["timestamp"] = filesizes["filename"].map(os.path.basename).\
                                              str.split(".").\
                                              map(lambda i: i[0]).\
                                              str.split("_").\
                                              map(lambda i: datetime(*list(map(int, i))))
    filesizes["date"] = filesizes["timestamp"].map(lambda i: i.date())
    filesizes_by_date = filesizes.groupby(["date"])["size"].sum().sort_index()
    ## Plot Filesizes (to see expected outliers)
    fig, ax = plt.subplots()
    _ = filesizes_by_date.plot(ax=ax, marker="o", linestyle="--", alpha=0.5, markersize=1)
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Filesize (GB)", fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{TEMP_DIR}filesizes.png", dpi=300)
    plt.close(fig)
    ## Execution
    if PARALLEL:
        _ = run_falconet(input_file=None,
                         nfiles=n,
                         njobs=NJOBS,
                         memory=MEMORY,
                         parallel=True)
    else:
        for f, input_file in enumerate(FILENAMES):
            LOGGER.info(f"\tWorking on: {input_file} ({f+1}/{n})")
            _ = run_falconet(input_file=input_file,
                             nfiles=n,
                             njobs=NJOBS,
                             memory=MEMORY,
                             parallel=False)
    LOGGER.info("Script complete! Remember to remove temporary files.")

########################
### Execution
########################

if __name__ == "__main__":
    _ = main()

