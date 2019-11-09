import argparse
import os
import numpy as np
import pandas as pd
import re
from pathlib import Path
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def list_all_extension_files(directory_path, extension='.csv'):
    """

    List all the files in the directory directory_path that have .csv as extension.
    :param directory_path:
    :return:
    """
    files_paths = []
    for r, d, f in os.walk(directory_path):
        f = [os.path.join(r,file) for file in f if file.find(extension) != -1]
        files_paths.append(f)

    files_paths = list(np.hstack(files_paths))
    return files_paths

def remove_punctuations(text):
    text = ''.join([character for character in text if character not in string.punctuation])
    return text

def remove_stop_words(text):
    #TODO: Scrivere da qualceh parte che bisogna fare questi comandi prima:
    """
        import nltk
        nltk.download('stopwords')

    """
    text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])
    return text


def lemmatize_text(text):
    #TODO: Scrivere da qualceh parte che bisogna fare questi comandi prima:
    """
        import nltk
        nltk.download('wordnet')

    """
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(i) for i in text.split()])
    return text


def lower_case(text):
    text = ' '.join([word.lower() for word in text.split()])
    return text


def tokenizer(text):

    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text.lower())


def preprocess(text):

    text = remove_punctuations(text)
    text = lemmatize_text(text)
    text = remove_stop_words(text)
    text = lower_case(text)
    return text

def cluster(input_file_paths):
    dict_centroids_speaker = {}
    for file_path in input_file_paths:
        file_df = pd.read_csv(file_path)
        #for ind in file_df.index:
        tmp_df = file_df[file_df.columns.difference(["name"])].applymap(preprocess)
        tmp_df['name'] = file_df['name']

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Segment file')
    parser.add_argument('--input_files_path', type=str, required=True, help='Directory containig the .csv files' )

    args = parser.parse_args()

    print(f"Starting segmentation job for Path: {args.input_files_path}")
    cluster(input_file_paths=list_all_extension_files(args.input_files_path))
    print(f"Job finished. Path: {args.input_files_path} completed")
