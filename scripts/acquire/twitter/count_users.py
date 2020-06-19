
"""
Count posts per user

Command-line Args:
- file_index (int): which file in the source list to process
- filelist (str): where the source list is cached
- outdir (str): where the counts should be cached
"""

####################
### Configuration
####################

## Globals
IGNORE_RETWEETS = True
IGNORE_NON_ENGLISH = True
FREQ = "D" # H, D, W, M, Y
SAMPLE_FREQ = 1
RANDOM_SEED = 42

####################
### Imports
####################

## Standard Library
import os
import sys
import gzip
import json
import random
from datetime import datetime
from collections import Counter

####################
### Globals
####################

## Random Number Generator
SEED = random.Random(x=RANDOM_SEED)

####################
### Helpers
####################

def get_time_bin(ts, freq="D"):
	"""

	"""
	tbin = [ts.year, ts.month, ts.day, ts.hour]
	indmap = {"y":1,"m":2,"d":3,"h":4}
	if freq.lower() not in indmap:
		raise ValueError("freq must be one of h, d, m, y (hour, day, month, year)")
	return "_".join(list(map(str,tbin[:indmap[freq.lower()]])))
	
####################
### Setup
####################

## Parse Command Line (Input File and Output Directory)
file_index = int(sys.argv[1]) - 1
filelist = sys.argv[2]
outdir = sys.argv[3]

## Ensure Output Directory Exists
if not os.path.exists(outdir):
	os.makedirs(outdir)

## Identify Filename
filelist = list(map(lambda x: x.strip(), open(filelist).readlines()))
if file_index > len(filelist) - 1:
	exit()
filename = filelist[file_index]

####################
### Data Procesing
####################

## Initialize Counter
counts = dict()

## Load File and Parse
with gzip.open(filename,"r") as the_file:
	try:
		for l, line in enumerate(the_file):
			## Random Sampling (Tweet-level)
			if SEED.uniform(0, 1) > SAMPLE_FREQ:
				continue
			## Parse JSON Line
			try:
				data = json.loads(line)
			except:
				continue
			## Check Filters
			if "user" not in data:
				continue
			if IGNORE_RETWEETS and data["text"].startswith("RT"):
				continue
			if IGNORE_NON_ENGLISH and data["lang"] != "en":
				continue
			## Identify Metadata
			user = data["user"]["id_str"]
			ts = datetime.fromtimestamp(int(data["timestamp_ms"]) / 1e3)
			## Cache
			tb = get_time_bin(ts, FREQ)
			if tb not in counts:
				counts[tb] = Counter()
			counts[tb][user] += 1
	## Catch File Error Exceptions
	except OSError as e:
		pass
	except Exception as e:
		pass

# ## Write Counts
# if len(counts) > 0:
# 	prefix = os.path.basename(filename).replace(".gz","")
# 	outfile = f"{outdir}{prefix}_processed.json"
# 	with open(outfile, "w") as the_file:
# 		json.dump(counts, the_file)

## Done
print("Script Complete")
