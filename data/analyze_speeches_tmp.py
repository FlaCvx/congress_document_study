import argparse
import pdb
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
    print(speeches)
    final = [[model.loc[word] for word in speech.split()] for speech in speeches.iloc[0]]
    final = [np.array(speech).mean(axis=0) for speech in final if len(speech)>0]
    if mean_speeches:
        final = np.array(final).mean(axis=0)
    return final

def embed_speeches(speeches, glove_model, mean_speeches=False):
    speeches = speeches.groupby(['name'])['speeches'].apply(lambda x: remove_out_of_vocab(glove_model, x.iloc[0]) if len(x)>0 else []).reset_index()
    print(speeches.shape)
    embedded_speeches = speeches.groupby(['name'])['speeches'].apply(lambda x: embed_and_average_speeches(glove_model, x, mean_speeches) if len(x)>0 else []).reset_index()

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

def embed_speeches_congress(input_file_paths, output_path, df_congressmen, glove_path, universalSE, SIF_weighted):

    all_files = dd.read_csv(input_file_paths, dtype={'0': 'object', '1': 'object', '2': 'object', '3': 'object',
                                                     '4': 'object', '5': 'object', '6': 'object', '7': 'object',
                                                     '8': 'object', '9': 'object', '10': 'object', '11': 'object',
                                                     '12': 'object','13': 'object', '14': 'object', '15': 'object',
                                                     '16': 'object','17': 'object', '18': 'object', '19': 'object',
                                                     '20': 'object', '21': 'object', '22': 'object', '23': 'object',
                                                     '24': 'object', '25': 'object', '26': 'object', '27': 'object',
                                                     '28': 'object', '29': 'object', '30': 'object', '31': 'object',
                                                     '32': 'object', '33': 'object', '34': 'object', '35': 'object',
                                                     '36': 'object', '37': 'object', '38': 'object', '39': 'object',
                                                     })

    for i in all_files.columns.values:
        all_files[i] = all_files[i].astype(str)

    #all_files = all_files.compute()
    all_files = match_congressmen(all_files=all_files, df_congressmen=df_congressmen)
    info = all_files[['name', 'filename', 'filepath', 'party_code']]

    all_files = all_files[all_files.columns.difference(['name', 'filename', 'filepath', 'party_code'])].applymap(preprocess_text)
    all_files = all_files.apply('|NEW_SPEECH|'.join, axis=1).reset_index(drop=True).to_frame().rename(columns={0: 'speeches'})
    all_files['name'] = info['name']
    all_files['party_code'] = info['party_code']
    all_files = all_files.groupby(['name','party_code'])['speeches'].apply('|NEW_SPEECH|'.join).to_frame()

    speeches_df = all_files['speeches'].apply(lambda x: x.split("|NEW_SPEECH|"), meta=('speeches', 'object')).to_frame()
    speeches_df = speeches_df.compute()

    if glove_path is not None:
        tmp_output_path = output_path.replace("congresses_embedded", "glove_embedded")
        if not os.path.exists(tmp_output_path):
            print(f"Creation of embeddings: {tmp_output_path}")
            glove_model = loadGloveModel(glove_path)
            speeches_df = embed_speeches(speeches_df, glove_model)
            speeches_df[['name','party_code']] = all_files.compute().reset_index()[['name','party_code']]
            speeches_df.to_csv(tmp_output_path)
            print(f"Wrote file: {tmp_output_path}")

    if universalSE:
        try:
            tmp_output_path = output_path.replace("congresses_embedded","universalSE_embedded")
            #TODO: Create directory if it does not exists
            if not os.path.exists(tmp_output_path):
                print(f"Creation of embeddings: {tmp_output_path}")
                speeches_df = universal_sentence_embedding(speeches_df)
                speeches_df.to_csv(tmp_output_path)
                print(f"Wrote file: {tmp_output_path}")
        except:
            count = 0


    if SIF_weighted:
        raise NotImplementedError
        tmp_output_path = output_path.replace("congresses_embedded", "sif_weighted_embedded")
        if not os.path.exists(tmp_output_path):
            speeches_df = SIF_weighted_embedding(speeches_df)
            speeches_df[['name','party_code']] = all_files.compute().reset_index()[['name','party_code']]
            speeches_df.to_csv(tmp_output_path)
            print(f"Wrote file: {tmp_output_path}")

    del speeches_df
    del all_files

    return

def embed_universal(speeches, embed_model, session):
    print(speeches)
    x = speeches.iloc[0]
    x = [ele for ele in x if ele!='nan' ]
    if len(x) > 0:
        x_e = session.run(embed_model(x))
    else:
        x_e = []
    return x_e 
    

def universal_sentence_embedding(speeches):
    import tensorflow_hub as hub
    import tensorflow as tf

    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embedded_speeches = speeches.groupby(['name','party_code'])['speeches'].apply(
            lambda x: embed_universal(x, embed, session) if len(x) > 0 else []).reset_index()
        embedded_speeches = embedded_speeches.reset_index()
        embedded_speeches = embedded_speeches.drop(columns=['index'])
    return embedded_speeches

def SIF_weighted_embedding(speeches):
    raise NotImplementedError
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Segment file')

    parser.add_argument('--input_files_path', type=str, required=True, help='Directory containig the .csv files' )
    parser.add_argument('--congressmen_csv', type=str, default="./HSall_members.csv", help='Path to the congressmen csv file')
    parser.add_argument('--glove_path', type=str, required=False, default=None, help='Path to the congressmen csv file')
    parser.add_argument('--universal_sentence_encoder', action='store_true', help='')
    parser.add_argument('--SIF_weighted_embedding', action='store_true', help='')

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
    if not os.path.exists(dir_output_path):
        os.makedirs(dir_output_path)

    for congress in congresses_files.keys():

        file_output_path = os.path.join(dir_output_path, "embedded_congress_"+str(congress)+".csv")
        tmp_1 = os.path.exists(file_output_path.replace("congresses_embedded", "universalSE_embedded")) if args.universal_sentence_encoder is not False else True 
        tmp_2 = os.path.exists(file_output_path.replace("congresses_embedded", "sif_weighted_embedded")) if args.SIF_weighted_embedding is not False else True

        if (not os.path.exists(file_output_path)) | (not tmp_1) | (not tmp_2):
            print(f"Creation of embeddings of speeches of congress {congress}")
            df_congressmen_filtered = df_congressmen[df_congressmen.congress==congress]
            embed_speeches_congress(input_file_paths=congresses_files[congress],
                                                                   output_path=file_output_path,
                                                                   df_congressmen=df_congressmen_filtered,
                                                                   glove_path=args.glove_path,
                                                                   universalSE= args.universal_sentence_encoder,
                                                                   SIF_weighted=args.SIF_weighted_embedding
                                                                   )
        #analysis_RandomizedSearch(X, y)
        else:
            print(f"Embeddings of speeches of congress {congress} already exists. ")

    print(f"Job finished. Analysis of path: {args.input_files_path} completed")

