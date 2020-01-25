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
from bigram_analysis import read_congressmen_info, group_volumes_by_congresses

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

def align_speeches(input_files_path):
    aligned_speeches = input_files_path.replace("speeches","all_speeches_aligned")
    if not os.path.exists(aligned_speeches):
        input_file_paths = list_all_extension_files(input_files_path)
        input_file_paths = remove_empty_df_from_list_dfs(input_file_paths)
        _ = d6tstack.combine_csv.CombinerCSV(input_file_paths).to_csv_align("all_speeches_aligned")


    # all_parents_directories = pd.Series(['/'.join(file.split("/")[:-1]) for file in input_file_paths]).unique()
    # for parent_dir in all_parents_directories:
    #     tmp_input_file_paths = [file for file in input_file_paths if '/'.join(file.split("/")[:-1]).find(parent_dir)>=0]
    #     if not os.path.exists(parent_dir.replace("speeches","speeches_aligned")):
    #         _ = d6tstack.combine_csv.CombinerCSV(tmp_input_file_paths).to_csv_align(parent_dir.replace("speeches","speeches_aligned"))

    return aligned_speeches


def remove_out_of_vocab(model,speeches):
    final = [str(' '.join([word for word in speech.split() if word in model.index])) for speech in speeches]
    return final

def embed_and_average_speeches(model,speeches, mean_speeches):
    final = [[model.loc[word] for word in speech[0].split()] for speech in speeches]
    final = [np.array(speech).mean(axis=0) for speech in final if len(speech)>0]
    if mean_speeches:
        final = np.array(final).mean(axis=0)
    return final

def embed_speeches(speeches, glove_model, mean_speeches=True):
    speeches = speeches.groupby(['name'])['speeches'].apply(lambda x: remove_out_of_vocab(glove_model, x)).reset_index()
    embedded_speeches = speeches.groupby(['name'])['speeches'].apply(lambda x: embed_and_average_speeches(glove_model, x, mean_speeches)).reset_index()

    return embedded_speeches

def match_congressmen(all_files, df_congressmen):


    all_files['name'] = parse_names(all_files['name'])
    all_files['name'] = all_files.name.str.replace("mr","").str.strip()
    df_congressmen['match_surname'] = df_congressmen.bioname.str.split(",", expand=True)[0].str.lower().values

    all_files = all_files[all_files.name.isin(df_congressmen.match_surname)]
    #Filter only names existing in the congressmen dataframe
    all_files = all_files.merge(df_congressmen[["match_surname", "party_code"]], how="left", left_on='name',
                                right_on='match_surname').drop(columns=['match_surname'])

    return all_files

def embed_speeches_congress(input_file_paths, output_path, df_congressmen, glove_model):

    all_files = dd.read_csv(input_file_paths, dtype={'0': 'object', '1': 'object', '2': 'object', '3': 'object',
                                                     '4': 'object', '5': 'object', '6': 'object', '7': 'object',
                                                     '8': 'object', '9': 'object', '10': 'object'})

    for i in all_files.columns.values:
        all_files[i] = all_files[i].astype(str)

    all_files = match_congressmen(all_files=all_files, df_congressmen=df_congressmen)
    info = all_files[['name', 'filename', 'filepath', 'party_code']]

    all_files = all_files[all_files.columns.difference(['name', 'filename', 'filepath', 'party_code'])].applymap(preprocess_text)
    all_files = all_files.apply('|NEW_SPEECH|'.join, axis=1).reset_index(drop=True).to_frame().rename(columns={0: 'speeches'})
    all_files['name'] = info['name']
    all_files = all_files.groupby(['name'])['speeches'].apply('|NEW_SPEECH|'.join).to_frame()
    speeches_df = all_files['speeches'].apply(lambda x: x.split("|NEW_SPEECH|"), meta=('speeches', 'object')).to_frame()

    speeches_df = embed_speeches(speeches_df, glove_model)

    speeches_df.to_csv(output_path)
    return speeches_df





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Segment file')
    parser.add_argument('--input_files_path', type=str, required=True, help='Directory containig the .csv files' )
    parser.add_argument('--congressmen_csv', type=str, default="./HSall_members.csv", help='Path to the congressmen csv file')
    parser.add_argument('--glove_path', type=str, required=True, help='Path to the congressmen csv file')

    args = parser.parse_args()

    if not os.path.exists("~/nltk_data"):
        import nltk
        nltk.download('wordnet')
        nltk.download('stopwords')

    df_congressmen = read_congressmen_info(args.congressmen_csv)

    dict_congresses_volumes = {1: ['Volume_1','Volume_2'],
                                   2: ['Volume_3'],
                                   3: ['Volume_4'],
                                   4: ['Volume_5','Volume_6'],
                                   5: ['Volume_7', 'Volume_8', 'Volume_9'],
                                   6: ['Volume_10'],
                                   7: ['Volume_11', 'Volume_12'],
                                   8: ['Volume_13', 'Volume_14'],
                                   9: ['Volume_15', 'Volume_16'],
                                   10: ['Volume_17', 'Volume_18', 'Volume_19'],
                                   11: ['Volume_20','Volume_21','Volume_22'],
                                   12: ['Volume_23', 'Volume_24', 'Volume_25'],
                                   13: ['Volume_26', 'Volume_27', 'Volume_28'],
                                   14: ['Volume_29', 'Volume_30'],
                                   15: ['Volume_31', 'Volume_32','Volume_33', 'Volume_34'],
                                   16: ['Volume_35', 'Volume_36', 'Volume_37'],
                                   17: ['Volume_38', 'Volume_39', 'Volume_40'],
                                   18: ['Volume_41','Volume_42']}

    aligned_speeches_path = align_speeches(input_files_path=args.input_files_path)
    congresses_files = group_volumes_by_congresses(input_path=aligned_speeches_path, dict_congresses=dict_congresses_volumes)

    dir_output_path = args.input_files_path.replace("speeches","congresses_embedded")
    for congress in congresses_files.keys():
        print(f"Creation of embeddings of speeches of congress {congress}")
        file_output_path = os.path.join(dir_output_path, "embedded_congress_"+str(congress)+".csv")
        if not os.path.exists(file_output_path):
            df_congressmen_filtered = df_congressmen[df_congressmen.congress==congress]
            glove_model = loadGloveModel(args.glove_path)
            embedded_congresses_speeches = embed_speeches_congress(input_file_paths=congresses_files[congress],
                                                                   output_path=file_output_path,
                                                                   df_congressmen=df_congressmen_filtered,
                                                                   glove_path=args.glove_path)

        else:
            embedded_congresses_speeches = pd.read_csv(file_output_path, index_col=0)
        #X, y = load_data(bigrams_count)
        #analysis_RandomizedSearch(X, y)


    print(f"Job finished. Analysis of path: {args.input_files_path} completed")
