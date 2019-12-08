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
from nltk import ngrams

def remove_empty_df_from_list_dfs(input_file_paths):
    new_paths = []
    for file in input_file_paths:
        try:
            df = pd.read_csv(file)
            new_paths.append(file)
        except EmptyDataError:
            pass
    return new_paths


def list_all_extension_files(directory_path, extension='.csv', remove_empty=False):
    """

    List all the files in the directory directory_path that have .csv as extension.
    :param directory_path:
    :return:
    """
    files_paths = []
    for r, d, f in os.walk(directory_path):
        f = [os.path.join(r, file) for file in f if file.find(extension) != -1]
        if remove_empty:
            f = remove_empty_df_from_list_dfs(f)
        if len(f) > 0:
            files_paths.append(f)

    files_paths = list(np.hstack(files_paths))
    return files_paths


def list_all_extension_files_per_directory(directory_path, extension='.csv'):
    """

    List all the files in the directory directory_path that have .csv as extension.
    :param directory_path:
    :return:
    """
    files_paths = []
    for r, d, f in os.walk(directory_path):
        f = [os.path.join(r, file) for file in f if file.find(extension) != -1]
        if len(f) > 0:
            files_paths.append(f)

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
        text = text.split()
        text = list(ngrams(text, 2))

    return text


def loadGloveModel(gloveFile="../../glove.6B/glove.6B.300d.txt"):
    print("Loading Glove Model")
    import csv
    model = pd.read_table(gloveFile, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    return model


def save_tuples_df(input_file_paths):
    dir_all_aligned = input_file_paths.replace("speeches","all_speeches_aligned")

    if not os.path.exists(dir_all_aligned):
        input_file_paths_allSubdirectories = list_all_extension_files(input_file_paths, remove_empty=True)
        print(f"There are {len(input_file_paths_allSubdirectories)} non empty Dataframes")
        if not os.path.exists(dir_all_aligned):
            os.makedirs(dir_all_aligned)
        _ = d6tstack.combine_csv.CombinerCSV(input_file_paths_allSubdirectories
                                             ).to_csv_align(dir_all_aligned)


    all_aligned_speeches = list_all_extension_files_per_directory(dir_all_aligned)
    all_aligned_speeches = [element for element in all_aligned_speeches if len(element)>0]

    for aligned_speeches in all_aligned_speeches:
        out_p = str(Path(aligned_speeches[0]).parent).replace("all_speeches_aligned", "df_tuples")
        file_out_p = os.path.join(out_p, "tuples_counts.csv")
        if not os.path.exists(file_out_p):
            if not os.path.exists(out_p):
                os.makedirs(out_p)
            all_files = dd.read_csv(aligned_speeches, dtype={'0': 'object', '1': 'object',
                                                                                      '2': 'object', '3': 'object',
                                                                                      '4': 'object', '5': 'object',
                                                                                      '6': 'object', '7': 'object',
                                                                                      '8': 'object', '9': 'object',
                                                                                      '10': 'object', '11': 'object',
                                                                                      '12': 'object', '13': 'object',
                                                                                      '14': 'object', '15': 'object',
                                                                                      '16': 'object', '17': 'object',
                                                                                      '18': 'object', '19': 'object',
                                                                                      '20': 'object', '21': 'object',
                                                                                      '22': 'object', '23': 'object',
                                                                                      '24': 'object', '25': 'object',
                                                                                      '26': 'object', '27': 'object',
                                                                                      '28': 'object', '29': 'object',
                                                                                      '30': 'object', '31': 'object',
                                                                                      '32': 'object', '33': 'object',
                                                                                      '34': 'object', '35': 'object',
                                                                                      '36': 'object', '37': 'object',
                                                                                      '38': 'object', '39': 'object',
                                                                                      '40': 'object', '41': 'object',
                                                                                      '42': 'object', '43': 'object',
                                                                                      '44': 'object', '45': 'object'})

            for i in all_files.columns.values:
                all_files[i] = all_files[i].astype(str)

            info = all_files[['name', 'filename', 'filepath']]

            all_files = all_files[all_files.columns.difference(['name', 'filename', 'filepath'])].applymap(preprocess_text).compute()

            all_files['name'] = parse_names(info['name'])
            c_all_speakers = {}
            for speaker in all_files.name.unique():
                tmp_speeches = pd.DataFrame(all_files[all_files.name==speaker][all_files.columns.difference(['name'])].values.reshape(1, -1))
                tmp_speeches = tmp_speeches.applymap(lambda x: np.nan if len(x)==0 else x)
                tmp_speeches = tmp_speeches.dropna(axis=1).T.reset_index(drop=True).T
                c_speaker = {}
                for col in tmp_speeches:
                    for tup in tmp_speeches[col][0]:
                        if tup not in c_speaker.keys():
                            c_speaker[tup] = 1
                        else:
                            c_speaker[tup] = c_speaker[tup]+1
                c_all_speakers[speaker] = c_speaker


            print(f"Writing: {file_out_p}")
            pd.DataFrame(c_all_speakers).to_csv(file_out_p)
    return


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

    print(f"Starting bi-gram analysis for Path: {args.input_files_path}")
    save_tuples_df(input_file_paths=args.input_files_path)


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