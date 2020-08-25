
######################
### Configuration
######################

## CLSP Grid Configuration
USERNAME = "kharrigian"
MODEL_DIR = "/export/fs03/a08/kharrigian/mental-health/models/falconet_v2/"

## Inference Configuration
# MODEL_FILE = "20200824135305-SMHD-Depression/model.joblib"
# MODEL_FILE = "20200824135147-SMHD-Anxiety/model.joblib"
# START_DATE = "2019-01-01"
# END_DATE = "2020-06-15"
# FREQ = "W-Mon"
# WINDOW_SIZE = 4
# STEP_SIZE = 1
# RUN_NAME = "monthly-weekly_step"
# PLATFORM = "reddit"
# INPUT_DIR = "/export/fs03/a08/kharrigian/covid-mental-health/data/processed/reddit/2017-2020/histories/"
# OUTPUT_DIR = "/export/fs03/a08/kharrigian/covid-mental-health/data/results/reddit/2017-2020/"

MODEL_FILE = "20200824134720-Multitask-Depression/model.joblib"
# MODEL_FILE = "20200824135027-Multitask-Anxiety/model.joblib"
START_DATE = "2019-01-01"
END_DATE = "2020-06-15"
FREQ = "W-Mon"
WINDOW_SIZE = 4
STEP_SIZE = 1
RUN_NAME = "monthly-weekly_step"
PLATFORM = "twitter"
INPUT_DIR = "/export/fs03/a08/kharrigian/covid-mental-health/data/processed/twitter/2018-2020/timelines/"
OUTPUT_DIR = "/export/fs03/a08/kharrigian/covid-mental-health/data/results/twitter/2018-2020/"

## Hold For Complete
HOLD_FOR_COMPLETE = False

######################
### Imports
######################

## Standard Library
import os
import subprocess
from time import sleep
from uuid import uuid4

## External Library
from pandas import date_range, to_datetime
from mhlib.util.logging import initialize_logger

######################
### Globals
######################

## Logger
LOGGER = initialize_logger()

######################
### Parameter Checks
######################

## Alert User to Model Window
if STEP_SIZE > WINDOW_SIZE:
    LOGGER.warning("WARNING: Using a STEP_SIZE > WINDOW_SIZE will result in non-consecutive modeling windows")

######################
### Helpers
######################

def get_running_jobs():
    """
    Identify all jobs running for the user specified in this script's configuration
    Args:
        None
    Returns:
        running_jobs (list): List of integer job IDs on the CLSP grid
    """
    jobs_res = subprocess.check_output(f"qstat -u {USERNAME}", shell=True)
    jobs_res = jobs_res.decode("utf-8").split("\n")[2:-1]
    running_jobs = [int(i.split()[0]) for i in jobs_res]
    return running_jobs

def hold_for_complete_jobs(scheduled_jobs):
    """
    Sleep until all jobs scheduled have been completed
    Args:
        scheduled_jobs (list): Output of schedule_jobs
    
    Returns:
        None
    """
    ## Sleep Unitl Jobs Complete
    complete_jobs = []
    sleep_count = 0
    while sorted(complete_jobs) != sorted(scheduled_jobs):
        ## Get Running Jobs
        running_jobs = get_running_jobs()
        ## Look for Newly Completed Jobs
        newly_completed_jobs = []
        for s in scheduled_jobs:
            if s not in running_jobs and s not in complete_jobs:
                newly_completed_jobs.append(s)
        ## Sleep If No Updates, Otherwise Update Completed Job List and Reset Counter
        if len(newly_completed_jobs) == 0:
            if sleep_count % 5 == 0:
                LOGGER.info("Inference jobs still running. Continuing to sleep.")
            sleep(20)
            sleep_count += 1
        else:
            complete_jobs.extend(newly_completed_jobs)
            LOGGER.info("Newly finished jobs: {}".format(newly_completed_jobs))
            n_jobs_complete = len(complete_jobs)
            n_jobs_remaining = len(scheduled_jobs) - n_jobs_complete
            LOGGER.info(f"{n_jobs_complete} jobs complete. {n_jobs_remaining} jobs remaining.")
            sleep_count = 0

######################
### Execution
######################

## Base Script
BASE_SCRIPT = """
#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m eas
#$ -e /home/kharrigian/gridlogs/python/covid_infer_{}_{}.err
#$ -o /home/kharrigian/gridlogs/python/covid_infer_{}_{}.out
#$ -pe smp 8
#$ -l 'gpu=0,mem_free=32g,ram_free=32g'

## Move to Home Directory (Place Where Virtual Environments Live)
cd /home/kharrigian/
## Activate Conda Environment
source .bashrc
conda activate covid-mental-health
## Move To Run Directory
cd /export/fs03/a08/kharrigian/covid-mental-health/
## Run Script
python ./scripts/model/infer.py \\
       {}{} \\
       --input {} \\
       --output_folder {}inference/{}/{}_{}/ \\
       --min_date {} \\
       --max_date {}
"""

## Dates
DATE_RANGE = list(date_range(START_DATE, END_DATE, freq=FREQ))
if to_datetime(START_DATE) < DATE_RANGE[0]:
    DATE_RANGE = [to_datetime(START_DATE)] + DATE_RANGE
if to_datetime(END_DATE) > DATE_RANGE[-1]:
    DATE_RANGE = DATE_RANGE + [to_datetime(END_DATE)]
DATE_RANGE = [i.date().isoformat() for i in DATE_RANGE]

## Model Windows
DATE_WINDOWS = []
date_ind = 0
while date_ind < len(DATE_RANGE) - 1:
    DATE_WINDOWS.append((DATE_RANGE[date_ind], DATE_RANGE[min(date_ind+WINDOW_SIZE, len(DATE_RANGE)-1)]))
    date_ind += STEP_SIZE

## Temporary Script Directory
rand_id = str(uuid4())
temp_dir = f"./temp_scheduler_{rand_id}/"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

## Create Scripts
script_files = []
for min_date, max_date in DATE_WINDOWS:
    ## Format Script
    date_script = BASE_SCRIPT.format(
        RUN_NAME,
        min_date,
        RUN_NAME,
        min_date,
        MODEL_DIR,
        MODEL_FILE,
        INPUT_DIR,
        OUTPUT_DIR,
        RUN_NAME,
        min_date,
        max_date,
        min_date,
        max_date
    )
    ## Write Script
    date_file = f"{temp_dir}covid_infer_{min_date}_{max_date}.sh"
    with open(date_file, "w") as the_file:
        the_file.write(date_script)
    script_files.append(date_file)

## Schedule Jobs
jobs = []
for sf in script_files:
    qsub_call = f"qsub {sf}"
    LOGGER.info(f"Scheduling job defined in {sf}")
    job_id = subprocess.check_output(qsub_call, shell=True)
    job_id = int(job_id.split()[2])
    jobs.append(job_id)

## Hold Till Completion
if HOLD_FOR_COMPLETE:
    ## Sleep and Wait
    _ = hold_for_complete_jobs(jobs)
    ## Remove Temp Directory
    _ = os.system(f"rm -rf {temp_dir}")

LOGGER.info("Script complete! Remember to remove temporary directory if not done already.")