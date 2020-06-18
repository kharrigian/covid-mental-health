
"""
Identify a list of gzipped JSON files containing 
tweets, write to a list .txt file

Command-line Args:
- rootpath (str): Directory where all of the .gz files live
- outpath (str): directory where the list of files should be stored
"""

#################
### Imports
#################

## Imports
import os
import sys
from glob import iglob

#################
### Helpers
#################

def find_files(rootdir, suffix=".gz"):
	"""

	"""
	filenames = list(iglob(f"{rootdir}**/*{suffix}", recursive=True))
	return sorted(filenames)

#################
### Find Files
#################

## Parse Command-Line
rootpath = sys.argv[1]
outpath = sys.argv[2]

## Get Filepaths
filepaths = find_files(rootpath, "*.gz")
print("Found {} files in {}".format(len(filepaths), rootpath))

#################
### Write
#################

## Write Filepaths to a .txt File
outfile = f"{outpath}source_filenames.txt"
print("Saved list to {}".format(outfile))
with open(outfile,"w") as the_file:
	for f in filepaths:
		the_file.write(f"{f}\n")


