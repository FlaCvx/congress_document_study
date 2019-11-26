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
    import csv
    model = pd.read_table(gloveFile, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    return model

def create_df_from_csv(input_file_paths):


    input_file_paths = remove_empty_df_from_list_dfs(input_file_paths)
    all_parents_directories = pd.Series(['/'.join(file.split("/")[:-1]) for file in input_file_paths]).unique()
    for parent_dir in all_parents_directories:
        tmp_input_file_paths = [file for file in input_file_paths if '/'.join(file.split("/")[:-1]).find(parent_dir)>=0]
        if not os.path.exists(parent_dir.replace("speeches","speeches_aligned")):
            _ = d6tstack.combine_csv.CombinerCSV(tmp_input_file_paths).to_csv_align(parent_dir.replace("speeches","speeches_aligned"))

    _ = d6tstack.combine_csv.CombinerCSV(input_file_paths).to_csv_align("all_speeches_aligned")

    all_files = dd.read_csv(list_all_extension_files("./all_speeches_aligned"), dtype={'0': 'object',
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
    speeches_df = all_files['speeches'].apply(lambda x: x.split("|NEW_SPEECH|"), meta=('speeches', 'object')).to_frame()
    if not os.path.exists("speeches_df.csv"):
        speeches_df = speeches_df.compute()
        speeches_df.to_csv("speeches_df.csv")
    speeches_df = pd.read_csv("speeches_df.csv")
    speeches_df = embed_speeches(speeches_df)
    return speeches_df

def remove_out_of_vocab(model,speeches):
    final = [str(' '.join([word for word in speech.split() if word in model.index])) for speech in speeches]
    return final

def embed_and_average_speeches(model,speeches, mean_speeches):
    final = [[model.loc[word] for word in speech[0].split()] for speech in speeches]
    final = [np.array(speech).mean(axis=0) for speech in final if len(speech)>0]
    if mean_speeches:
        final = np.array(final).mean(axis=0)
    return final

def embed_speeches(speeches, mean_speeches=True):
    model = loadGloveModel("../../glove.840B.300d.txt")
    speeches = speeches.groupby(['name'])['speeches'].apply(lambda x: remove_out_of_vocab(model, x)).reset_index()
    embedded_speeches = speeches.groupby(['name'])['speeches'].apply(lambda x: embed_and_average_speeches(model, x, mean_speeches)).reset_index()

    return embedded_speeches


def cluster_speeches(speeches_df):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Segment file')
    parser.add_argument('--input_files_path', type=str, required=True, help='Directory containig the .csv files' )

    args = parser.parse_args()

    if not os.path.exists("~/nltk_data"):
        import nltk
        nltk.download('wordnet')
        nltk.download('stopwords')

    print(f"Starting analysis job for Path: {args.input_files_path}")
    speeches_df = create_df_from_csv(input_file_paths=list_all_extension_files(args.input_files_path))
    speeches_df.to_csv("embedded_speeches.csv")

    print(f"Job finished. Analysis of path: {args.input_files_path} completed")


def parse_speech(text):
    if text is not np.nan:
        text = text.replace("[", "").replace("\n", "").replace("]","")
        text = [ float(num) for num in text.split()]
    return text

def transform_in_df(df_of_lists):
    tmp = pd.DataFrame()
    for ind in df_of_lists.index:
        if df_of_lists.iloc[ind] is not np.nan:
            tmp=tmp.append(pd.DataFrame(np.array(df_of_lists.iloc[ind]).reshape((1, 300))))
        else:
            tmp=tmp.append(pd.DataFrame())
    return tmp

def write_tsv(df):
    import csv
    f = open("emb.tsv", "w")
    for ind in df.index:
        f.write('\t'.join([str(value) for value in df.iloc[ind].values]))
        f.write("\n")
    f.close()
    return