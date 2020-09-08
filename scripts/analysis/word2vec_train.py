
############################
### Imports
############################

## Standard Library
import os
import sys
import json
import gzip
import argparse
import random
from glob import glob
from uuid import uuid4

## External
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

## Mental Health
from mhlib.util.helpers import flatten
from mhlib.util.logging import initialize_logger
from mhlib.model.data_loaders import LoadProcessedData
from mhlib.preprocess.tokenizer import Tokenizer

############################
### Globals
############################

## Logger
LOGGER = initialize_logger()

## Initialize Tokenizer
TOKENIZER = Tokenizer(stopwords=None,
                      keep_case=False,
                      negate_handling=False,
                      negate_token=False,
                      keep_punctuation=False,
                      keep_numbers=False,
                      expand_contractions=True,
                      keep_user_mentions=False,
                      keep_pronouns=True,
                      keep_url=False,
                      keep_hashtags=True,
                      keep_retweets=False,
                      emoji_handling=None,
                      strip_hashtag=False)

############################
### Functions
############################

def parse_arguments():
    """
    Parse command-line to identify configuration filepath.

    Args:
        None
    
    Returns:
        args (argparse Object): Command-line argument holder.
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Preprocess raw Twitter or Reddit data")
    ## Generic Arguments
    parser.add_argument("indir", type=str, help="Path to processed data files for training")
    parser.add_argument("outdir", type=str, help="Path for saving model")
    parser.add_argument("--start_date", type=str, default=None, help="Lower date bound")
    parser.add_argument("--end_date", type=str, default=None, help="Upper date bound")
    parser.add_argument("--sample_size", type=int, default=None, help="Maximum number of training files")
    parser.add_argument("--sample_rate", type=float, default=1, help="Post-level sample rate (0,1]")
    parser.add_argument("--pretokenize", action="store_true", default=False, help="Run tokenization once at the start and cache on disk.")
    parser.add_argument("--model_dim", type=int, default=200, help="Vector dimensionality")
    parser.add_argument("--model_min_freq", type=int, default=5, help="Minimum token frequency")
    parser.add_argument("--model_context_size", type=int, default=5, help="Context window")
    parser.add_argument("--model_use_skipgram", action="store_true", default=False, help="Use skipgram instead of CBOW")
    parser.add_argument("--model_use_softmax", action="store_true", default=False, help="Use softmax instead of negative sampling")
    parser.add_argument("--model_negative", type=int, default=10, help="Number of words to use for negative sampling")
    parser.add_argument("--model_learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model_max_vocab_size", type=int, default=250000, help="Maximum vocabulary size")
    parser.add_argument("--model_sort_vocab", action="store_true", default=False, help="Sort vocabulary by frequency")
    parser.add_argument("--model_epochs", type=int, default=10, help="Number of Training Epochs")
    parser.add_argument("--model_shuffle", action="store_true", default=False, help="Shuffle filenames between epochs")
    parser.add_argument("--jobs", default=8, type=int, help="Number of processes to use")
    parser.add_argument("--random_state", default=42, type=int, help="Random Seed")
    ## Parse Arguments
    args = parser.parse_args()
    return args


class FileStream(object):
    
    """
    FileStream (Iterable)
    """

    def __init__(self,
                 filenames,
                 min_date=None,
                 max_date=None,
                 pretokenized=False,
                 randomized=True,
                 n_samples=None,
                 shuffle=False,
                 random_state=42):
        """
        FileStream (Iterable) used for passing tokenized sentences
        to Word2Vec model

        Args:
            filenames (list of str): Training file paths
            min_date (datetime or None): Lower date boundary
            max_date (datetime or None): Upper date boundary
            pretokenized (bool): Specifies whether to use existing "text_tokenized" field
            randomized (bool): Whether any random sampling of posts should be uniform
            n_samples (int, float, or None): Post-level sampling
            shuffle (bool): Whether to shuffle filenames at start of an iteration
            random_state (int): Random seed

        Returns:
            self
        """
        ## Initialize Loader
        self.loader = LoadProcessedData(random_state=random_state)
        ## Class Attributes
        self.filenames = filenames
        self.min_date = min_date
        self.max_date = max_date
        self.pretokenized = pretokenized
        self.randomized = randomized
        self.n_samples = n_samples
        self.shuffle = shuffle
        self.random_state = random_state
        ## Set Random Seed
        np.random.seed(self.random_state)
        self.random = random.Random(self.random_state)


    def __iter__(self):
        """
        Iterable method to pass tokenized sentences from a list of filenames
        to a Word2Vec model

        Args:
            None
        
        Yields:
            sentence (list of str): Tokens for a single sentence
        """
        ## Shuffle (If Desired)
        if self.shuffle:
            filenames = sorted(self.filenames, key=lambda k: self.random.uniform(0,1))
        else:
            filenames = self.filenames
        ## Cycle Through Filenames
        for file in filenames:
            if self.pretokenized:
                ## Load Tokens
                sentences = self.load_tokens(file,
                                             n_samples=self.n_samples)
            else:
                ## Load Original File
                file_data = self.load_text(file,
                                           min_date=self.min_date,
                                           max_date=self.max_date,
                                           randomized=self.randomized,
                                           n_samples=self.n_samples)
                ## Parse and Tokenize Sentences
                sentences = self.tokenize_user_data(file_data)
            ## Yield Sentences
            for sentence in sentences:
                yield sentence
    
    def load_tokens(self,
                    filename,
                    n_samples=None):
        """
        Load Tokens (Assume date-based filter already completed)
        """
        ## Load Tokens
        file_data = []
        with gzip.open(filename,"r") as the_file:
            for line in the_file:
                file_data.append(json.loads(line))
        ## Post-level Sampling
        if n_samples is not None:
            file_data = self.loader._select_documents_randomly(file_data,
                                                               n_samples)
        ## Flatten Sentences
        sentences = flatten([i["text"] for i in file_data])
        return sentences

    def load_text(self,
                  filename,
                  min_date=None,
                  max_date=None,
                  randomized=True,
                  n_samples=None):
        """
        Load a processed file, filtering and sampling
        as desired

        Args:
            filename (str): Path to processed data file (.gzip)
            min_date (datetime or None): Lower date boundary
            max_date (datetime or None): Upper date boundary
            randomized (bool): Whether to use uniform random sampling instead of recency sampling
            n_samples (float, int, or None): Amount of post-level sampling
        
        Returns:
            user_data (list of str): Raw post text
        """
        ## Load the Gzipped File
        with gzip.open(filename, "r") as the_file:
            user_data = json.load(the_file)
        ## Data Amount Filtering
        user_data = self.loader._select_n_recent_documents(user_data)
        ## Date-basd Filtering
        user_data = self.loader._select_documents_in_date_range(user_data,
                                                                min_date,
                                                                max_date)
        ## Post-level Sampling
        if n_samples is not None:
            if randomized:
                user_data = self.loader._select_documents_randomly(user_data,
                                                                   n_samples)
            else:
                user_data = user_data[:min(len(user_data), n_samples)]
        ## Isolate Text
        user_data = list(map(lambda i: i.get("text"), user_data))
        return user_data

    def tokenize_user_data(self,
                           user_data):
        """
        Tokenize user data into separate sentences

        Args:
            user_data (list of str): Unique posts
        
        Returns:
            sentences (list of str): Posts, tokenized into sentences and words
        """
        ## Sentence Tokenization
        sentences = flatten(list(map(sent_tokenize, user_data)))
        ## Word Tokenization
        sentences = list(map(TOKENIZER.tokenize, sentences))
        sentences = list(filter(lambda x: len(x) > 0, sentences))
        return sentences

class LossCallback(CallbackAny2Vec):
    
    """
    Callback to print loss after each epoch.
    """

    def __init__(self):
        """
        Initialize callback

        Args:
            None
        
        Returns:
            None
        """
        self.epoch = 0
        self.loss = []
    
    def __repr__(self):
        """
        Clean Class Name

        Args:
            None

        Returns:
            desc (str): LossCallback()
        """
        return "LossCallback()"

    def on_epoch_end(self,
                     model):
        """
        Functions to run after epoch completes

        Args:
            model (word2vec model): Model being trained
        
        Returns:
            None, updates epochs inplace
        """
        loss = model.get_latest_training_loss()
        LOGGER.warning("~"*50 + '\n~~~ Loss after epoch {}/{}: {:.3f}\n'.format(self.epoch+1, model.epochs, loss) + "~"*50)
        self.loss.append(loss)
        self.epoch += 1

def main():
    """

    """
    ## Parse Command Line
    args = parse_arguments()
    ## Output Directory
    if not os.path.exists(args.outdir):
        _ = os.makedirs(args.outdir)
    ## Set Random Seed
    np.random.seed(args.random_state)
    ## Get Data Files
    LOGGER.info("Finding Files")
    filenames = sorted(glob(f"{args.indir}*.json.gz"))
    ## Downsample Files
    if args.sample_size is not None:
        filenames = sorted(np.random.choice(filenames, min(args.sample_size, len(filenames)), replace=False))
    ## Pre-tokenization
    if args.pretokenize:
        ## Initialize Temporary Cache Directory
        temp_dir = "temp_{}".format(str(uuid4()))
        while os.path.exists(temp_dir):
            temp_dir = "temp_{}".format(str(uuid()))
        _ = os.makedirs(temp_dir)
        ## Initialize Helper Class
        file_stream = FileStream(filenames,
                                 min_date=pd.to_datetime(args.start_date) if args.start_date is not None else None,
                                 max_date=pd.to_datetime(args.end_date) if args.end_date is not None else None,
                                 random_state=args.random_state)
        ## Tokenize All Available Text in File
        pretokenized_filenames = []
        for f in tqdm(filenames, file=sys.stdout, desc="Tokenization"):
            f_data = file_stream.load_text(filename=f,
                                           min_date=pd.to_datetime(args.start_date) if args.start_date is not None else None,
                                           max_date=pd.to_datetime(args.end_date) if args.end_date is not None else None,
                                           n_samples=None)
            f_tokens = [{"text":tokens} for tokens in list(map(lambda x: file_stream.tokenize_user_data([x]), f_data))]
            fname = "{}/{}".format(temp_dir, os.path.basename(f))
            with gzip.open(fname,"wt", encoding="utf-8") as the_file:
                for post in f_tokens:
                    the_file.write(json.dumps(post) + "\n")
            pretokenized_filenames.append(fname)
        ## Update Filesnames
        filenames = pretokenized_filenames
    ## Initialize Word2Vec
    LOGGER.info("Initializing Word2Vec Model")
    word2vec = Word2Vec(size=args.model_dim,
                        alpha=args.model_learning_rate,
                        window=args.model_context_size,
                        min_count=args.model_min_freq,
                        max_vocab_size=args.model_max_vocab_size,
                        sorted_vocab=args.model_sort_vocab,
                        seed=args.random_state,
                        workers=args.jobs,
                        sg=args.model_use_skipgram,
                        hs=args.model_use_softmax,
                        negative=args.model_negative)
    ## Build Vocabulary
    LOGGER.info("Building Vocabulary")
    vocab_stream = FileStream(filenames,
                              min_date=pd.to_datetime(args.start_date) if args.start_date is not None else None,
                              max_date=pd.to_datetime(args.end_date) if args.end_date is not None else None,
                              pretokenized=args.pretokenize,
                              randomized=True,
                              n_samples=args.sample_rate if args.sample_rate < 1 else None,
                              shuffle=False,
                              random_state=args.random_state)
    word2vec.build_vocab(sentences=vocab_stream,
                         keep_raw_vocab=True)
    ## Train Model
    LOGGER.info("Beginning Training")
    train_stream = FileStream(filenames,
                              min_date=pd.to_datetime(args.start_date) if args.start_date is not None else None,
                              max_date=pd.to_datetime(args.end_date) if args.end_date is not None else None,
                              pretokenized=args.pretokenize,
                              randomized=True,
                              n_samples=args.sample_rate if args.sample_rate < 1 else None,
                              shuffle=args.model_shuffle,
                              random_state=args.random_state)
    word2vec.train(sentences=train_stream,
                   total_examples=word2vec.corpus_count,
                   epochs=args.model_epochs,
                   compute_loss=True,
                   callbacks=[LossCallback()])
    LOGGER.info("Finished Training")
    ## Save Model (Resetting Callback Attribute)
    LOGGER.info("Saving Model")
    word2vec.callbacks = ()
    word2vec.save(f"{args.outdir}word2vec.model")
    ## Remove any temporary directories
    if args.pretokenize:
        LOGGER.info("Removing Temporary Data Directories")
        _ = os.system("rm -rf {}".format(temp_dir))
    LOGGER.info("Script Complete!")    

############################
### Execute
############################

if __name__ == "__main__":
    _ = main()