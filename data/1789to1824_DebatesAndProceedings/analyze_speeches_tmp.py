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


def preprocess_text(text):
    #TODO Add stopwords removal etc!!!
    if text is not np.nan:
        text = remove_punctuations(text)
        text = lower_case(text)
    return text


def create_df_from_csv(input_file_paths):
    dict_centroids_speaker = {}
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

    info = all_files[['name','filename','filepath']]

    all_files = all_files[all_files.columns.difference(['name', 'filename', 'filepath'])].applymap(preprocess_text)
    all_files = all_files.apply('|NEW_SPEECH|'.join, axis=1).reset_index(drop=True).to_frame().rename(columns={0: 'speeches'})
    all_files['name'] = parse_names(info['name'])
    all_files = all_files.groupby(['name'])['speeches'].apply('|NEW_SPEECH|'.join)
    speeches_df = all_files['speeches'].apply(lambda x: x.split("|NEW_SPEECH|"), meta=('speeches', 'object')).compute()

    return speeches_df

def embed_speeches(speeches_df):
    pass

    return

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
