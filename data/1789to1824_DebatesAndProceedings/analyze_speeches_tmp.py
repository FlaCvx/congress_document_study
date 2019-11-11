import argparse
import os
import numpy as np
import pandas as pd
import re
from pathlib import Path
import string
import dask.dataframe as dd
import d6tstack.combine_csv
from pandas.io.common import EmptyDataError
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


def remove_empty_df_from_list_dfs(input_file_paths):
    new_paths = []
    for file in input_file_paths:
        try:
            df = pd.read_csv(file)
            new_paths.append(file)
        except EmptyDataError:
            pass
    return new_paths


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


def parse_names(names):
    #TODO: Some names may be the same, but due to ocr they are written badly.
    names = names.str.lower().apply(remove_punctuations, meta=('name', 'object'))
    return names


def remove_punctuations(text):
    text = ''.join([character for character in text if character not in string.punctuation])
    return text


def lower_case(text):
    text = ' '.join([word.lower() for word in text.split()])
    return text


def remove_stop_words(text):
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


def preprocess_text(text):

    if text is not np.nan:
        text = remove_punctuations(text)
        text = lemmatize_text(text)
        text = remove_stop_words(text)
        text = lower_case(text)
    return text


def loadGloveModel(gloveFile="../../glove.6B/glove.6B.300d.txt"):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def create_df_from_csv(input_file_paths):

    if not os.path.exists("speeches_aligned"):
        input_file_paths = remove_empty_df_from_list_dfs(input_file_paths)
        _ = d6tstack.combine_csv.CombinerCSV(input_file_paths).to_csv_align("speeches_aligned/")

    all_files = dd.read_csv(list_all_extension_files("./speeches_aligned"), dtype={'0': 'object',
                                                                                   '1': 'object',
                                                                                   '2': 'object',
                                                                                   '3': 'object',
                                                                                   '4': 'object',
                                                                                   '5': 'object',
                                                                                   '6': 'object'})
    for i in all_files.columns.values:
        all_files[i] = all_files[i].astype(str)

    info = all_files[['name', 'filename', 'filepath']]

    all_files = all_files[all_files.columns.difference(['name', 'filename', 'filepath'])].applymap(preprocess_text)
    all_files = all_files.apply('|NEW_SPEECH|'.join, axis=1).reset_index(drop=True).to_frame().rename(columns={0: 'speeches'})
    all_files['name'] = parse_names(info['name'])
    all_files = all_files.groupby(['name'])['speeches'].apply('|NEW_SPEECH|'.join).to_frame()
    speeches_df = all_files['speeches'].apply(lambda x: x.split("|NEW_SPEECH|"), meta=('speeches', 'object')).compute()
    prova_df = speeches_df
    print(prova_df.iloc[0])
    speeches_df = speeches_df.applymap(embed_speeches)
    speeches_df = speeches_df.compute()
    return speeches_df


def embed_speeches(speeches, mean_speeches=True):
    model = loadGloveModel()
    embedded_speeches = [[model(word) for word in speech] for speech in speeches]
    embedded_speeches = [np.mean(speech, axis=1) for speech in embedded_speeches]
    if mean_speeches:
        embedded_speeches = np.mean(embedded_speeches)
    return embedded_speeches


def cluster_speeches(speeches_df):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Segment file')
    parser.add_argument('--input_files_path', type=str, required=True, help='Directory containig the .csv files' )

    args = parser.parse_args()

    print(f"Starting analysis job for Path: {args.input_files_path}")
    speeches_df = create_df_from_csv(input_file_paths=list_all_extension_files(args.input_files_path))
    embedded_speeches_df = embed_speeches(speeches_df=speeches_df)

    print(f"Job finished. Analysis of path: {args.input_files_path} completed")
