import argparse
import os
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
import numpy as np
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
    return "./1789to1824_DebatesAndProceedings/df_tuples/"
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

    out_p = str(Path(all_aligned_speeches[0][0]).parent).replace("all_speeches_aligned", "df_tuples") if len(all_aligned_speeches)>0 else None
    if out_p is not None:
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

        return out_p


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


def analysis(X, y):
    print(__doc__)

    # build a classifier
    clf = RandomForestClassifier(n_estimators=20)

    # Utility function to report best scores
    def report(grid_scores, n_top=3):
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                score.mean_validation_score,
                np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")

    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # # run randomized search
    # n_iter_search = 20
    # random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
    #                                    n_iter=n_iter_search)
    #
    # start = time()
    # random_search.fit(X, y)
    # print("RandomizedSearchCV took %.2f seconds for %d candidates"
    #       " parameter settings." % ((time() - start), n_iter_search))
    # report(random_search.grid_scores_)

    # use a full grid over all parameters
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10, 20, 30],
                  "min_samples_split": [1, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()
    grid_search.fit(X, y)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.grid_scores_)))
    report(grid_search.grid_scores_)

def analysis(X, y):
    estimator = EstimatorSelectionHelper()
    models = {
        # 'ExtraTreesClassifier': ExtraTreesClassifier(),
        'SVC': SVC(),
        # 'LogisticRegression': LogisticRegression()
    }

    params = {
        # 'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
        # 'LogisticRegression' : {'C':[0.1,0.5,1,2,5,10]},
        'SVC': [
            {'kernel': ['linear', 'rbf'], 'C': [0.1, 0.5, 1, 2, 5, 10]}
        ]
    }
    helper = EstimatorSelectionHelper(models, params)
    helper.fit(X, y, scoring='accuracy', n_jobs=2)
    helper.score_summary(sort_by='mean_score')



class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=5, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=5, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            print(X.shape)
            gs.fit(X,y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

def match_congressmen(bigrams_count, df_congressmen):

    #Create the surname to match those ones of the congressmen
    df_congressmen['match_surname'] = df_congressmen.bioname.str.split(",", expand=True)[0].str.lower().values

    print(f"Shape of the original bigrams matrix: {bigrams_count.shape}")
    print(f"Columns of the original bigrams matrix: {bigrams_count.columns}")

    #Filter only the columns of people in the df_congressmen dataframe, this will make the merge easier.
    columns_congressmen = bigrams_count.columns.difference(["w0","w1"]).values
    matches = [man for man in columns_congressmen if man.split("mr ")[-1] in df_congressmen.match_surname]
    matches = matches + ["w0","w1"]
    bigrams_count = bigrams_count[matches]
    print(f"Shape of the filtered bigrams matrix: {bigrams_count.shape}")

    print("Original: ", bigrams_count.head())
    #Create the match_surname column for te bigrmas dataframe as well
    bigrams_count = bigrams_count.T
    print("Transposed: ", bigrams_count.head())
    bigrams_count['match_surname'] = bigrams_count.index
    bigrams_count['match_surname'] = bigrams_count.match_surname.str.split("mr ", expand=True)[1]
    print("Matched surnames: ", bigrams_count.head())

    bigrams_count = bigrams_count.merge(df_congressmen[["match_surname","party_code"]], left_on='match_surname', right_on='match_surname')
    print("Merged: ", bigrams_count.head())
    #Could remove outliers

    X = bigrams_count[[bigrams_count.columns.difference(["match_surname","party_code"])]].values
    y = bigrams_count.party_code.values
    return X,y


def keep_common_bigrams(bigrams_count, threshold=1):

    keep_not_rare_indexes = np.where((bigrams_count.sum(axis=1).values) > threshold)[0] #Remove tuples of w0,w1 used only by one congressmen
    bigrams_count = bigrams_count[bigrams_count.index.isin(keep_not_rare_indexes)]

    return bigrams_count


def extract_data(input_file_paths, df_congressmen):

    all_paths = list(np.hstack(list_all_extension_files_per_directory(directory_path=input_file_paths, extension='csv')))

    aligned_csvs = input_file_paths.replace("df_tuples","df_tuples_aligned")
    if not os.path.exists(aligned_csvs):
        _ = d6tstack.combine_csv.CombinerCSV(all_paths).to_csv_align(aligned_csvs)

    bigrams_count = dd.read_csv(list(np.hstack(list_all_extension_files_per_directory(directory_path=aligned_csvs, extension='csv'))))
    bigrams_count = bigrams_count.drop(columns=["filename","filepath"])
    bigrams_count = bigrams_count.reset_index(drop=True).rename(columns={"Unnamed: 0":"w0", "Unnamed: 1":"w1"})
    bigrams_count = bigrams_count.fillna(0).compute()

    bigrams_count = keep_common_bigrams(bigrams_count=bigrams_count, threshold=1)

    bigrams_count = bigrams_count.groupby(["w0","w1"]).sum() #Sums the bigrams of the same congressmen
    bigrams_count = bigrams_count.reset_index()
    # dd reads the csv by appending the new rows, if they have a column already, otherwise creates the column
    # therefore with this operation I count if the bigram is present elsewhere for the same congressmen

    #Match congressmen
    X, y = match_congressmen(bigrams_count, df_congressmen)

    assert isinstance(X, np.array)
    assert isinstance(y, np.array)

    return X, y


def read_congressmen_info(congressmen_csv):

    info_congressmen = pd.read_csv(congressmen_csv)
    dict_parties = {5000:'Pro-Administration', 4000:'Anti-Administration', 1:'Federalist Party',
                    13: 'Democratic-Republican Party', 1346: 'Jackson Democratic-Republican',
                    8888: 'Adams-Clay Democratic-Republican', 6000: 'Crawford Federalist',
                    7777: 'Crawford Democratic-Republican', 8000: 'Adams-Clay Federalist', 7000: 'Jackson Federalist',
                    22: 'Anti-Jacksonian', 555: 'Jacksonian', 1275: 'Anti-Jacksonian', 26: 'Anti-Masonic',
                    44:'Nullifier', 29: 'Whig', 100: 'Democratic', 328: 'Independent', 112: 'Conservative',
                    329: 'Independent Democratic', 603: 'Independent Whig', 403: 'Law and Order', 310: 'American',
                    1111: 'Liberty', 300: 'Free Soil', 4444: 'Unionist', 46: 'States\' Rights', 3333: 'Opposition',
                    200: 'Republican', 3334: 'Opposition', 108: 'Anti-Lecompton Democratic', 206: 'Unionist',
                    37: 'Constitutional Unionist', 203: 'Unconditional Unionist', 331: 'Independent Republican',
                    1116: 'Conservative Republican', 208: 'Liberal Republican', 326: 'Greenback', 117: 'Democratic',
                    114: 'Readjuster', 355: 'Labor', 356: 'Socialist Labor', 340: 'Populist', 1060: 'Silver',
                    354: 'Silver Republican', 213: 'Democratic', 380: 'Socialist', 370: 'Progressive',
                    347: 'Prohibitionist', 537: 'Farmer–Labor', 523: 'Republican', 522: 'American Labor',
                    402: 'Liberal'}
    info_congressmen['party_name'] = info_congressmen.party_code.apply(lambda x: dict_parties[x])
    return info_congressmen


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bigrams Analysis file')
    parser.add_argument('--input_files_path', type=str, required=True, help='Directory containig the .csv files')
    parser.add_argument('--congressmen_csv', type=str, default="./HSall_members.csv", help='Path to the congressmen csv file')

    args = parser.parse_args()

    if not os.path.exists("~/nltk_data"):
        import nltk
        nltk.download('wordnet')
        nltk.download('stopwords')

    print(f"Starting bi-gram analysis for Path: {args.input_files_path}")
    output_path = save_tuples_df(input_file_paths=args.input_files_path)

    assert (output_path is not None)

    #Do here the division by sessions.

    df_congressmen = read_congressmen_info(args.congressmen_csv)
    X, y = extract_data(input_file_paths=output_path, df_congressmen=df_congressmen)

    #Run gridSearch
    analysis(X, y)

    print(f"Job finished. Analysis of path: {args.input_files_path} completed")
