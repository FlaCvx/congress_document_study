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
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold

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
    #return input_file_paths.replace("speeches","df_tuples")
    #return "./1789to1824_DebatesAndProceedings/df_tuples/"
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

        return input_file_paths.replace("speeches","df_tuples")


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


def analysis_GridSearch(X, y):
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
                  "min_samples_split": [3, 10],
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

def explore_classes(y):
    from collections import Counter
    classes = Counter(y)
    print("Classes: ")
    print(classes)
    return

def explore_features(X):
    features_occurrences = []
    for j in range(0, X.shape[1], 1):
        features_occurrences.append(len(np.where(X[:, j] > 1)[0]))
    hist, bin_edges = np.histogram(features_occurrences, density=True)
    #print("Histogram: ", hist)

    import matplotlib.pyplot as plt


    # the histogram of the data
    n, bins, patches = plt.hist(features_occurrences,density=True, facecolor='g', alpha=0.75)

    plt.xlabel('feature occurrence')
    plt.ylabel('Probability')
    plt.title('Histogram of Features Occurrences')
    plt.xlim(40, 160)
    plt.ylim(0, 0.03)
    plt.grid(True)
    plt.show()
    i=input()
    return

def analysis_RandomizedSearch(X, y):
    #print(__doc__)

    # build a classifier
    from sklearn import preprocessing
    explore_features(X)
    explore_classes(y)

    X_normalized = preprocessing.normalize(X)
    clf = RandomForestClassifier(n_estimators=20)

    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3,10, 20, None],
                  "max_features": [1, 3, 10, 20, 30, 50, 100],
                  "min_samples_split": [3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)

    start = time()
    random_search.fit(X_normalized, y)
    #print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % ((time() - start), n_iter_search))
    print(f"Accuracy score: {random_search.best_score_}")
    best_model = random_search.best_estimator_

    # kf = KFold(n_splits=5)
    # from sklearn.metrics import accuracy_score
    # from sklearn.metrics import multilabel_confusion_matrix
    # from sklearn.metrics import precision_score
    # from sklearn.metrics import recall_score
    #
    # accuracy = []
    # precision = []
    # recall = []
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     best_model.fit(X_train, y_train)
    #     y_predicted = best_model.predict(X_test)
    #     accuracy.append(accuracy_score(y_test, y_predicted))
    #     precision.append(precision_score(y_test, y_predicted, average='micro' ))
    #     recall.append(recall_score(y_test, y_predicted, average='micro'))
    #     #print(f"Confusion Matrix: {multilabel_confusion_matrix(y_test, y_predicted)}\n")
    #     print(f"Mean accuracy score: {np.mean(accuracy)}")
    #     print(f"Mean precision score: {np.mean(precision)}")
    #     print(f"Mean recall score: {np.mean(recall)}")

    print()
    return random_search

def third_analysis(X, y):
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        return y_true, y_pred
    # Note the problem is too easy: the hyperparameter plateau is too flat and the
    # output model is the same for precision and recall with ties in quality.


def analysis(X, y):

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

    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    X_new = SelectKBest(chi2, k=100).fit_transform(X, y)

    helper.fit(X_new, y, scoring='accuracy', n_jobs=2)
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

    bigrams_count = bigrams_count.rename(columns={i: i.replace("mr ","") for i in bigrams_count.columns})

    #Filter only the columns of people in the df_congressmen dataframe, this will make the merge easier.
    columns_congressmen = bigrams_count.columns.difference(["w0","w1"]).values
    matches = [man for man in columns_congressmen if man in df_congressmen.match_surname.values]
    matches = ["w0","w1"] + matches
    bigrams_count = bigrams_count[matches]


    bigrams_count = bigrams_count.set_index(["w0","w1"]).T
    #bigrams_count = bigrams_count.merge(df_congressmen.set_index("match_surname")["party_code"], how="left", left_index=True, right_index=True)


    return bigrams_count


def keep_common_bigrams(bigrams_count, threshold=1):

    bigrams_count = bigrams_count[bigrams_count.sum(axis=1)>threshold]

    return bigrams_count

def extract_data_old(input_file_paths, parent_path, df_congressmen):

    aligned_csvs = parent_path.replace("df_tuples","df_tuples_aligned")
    all_paths_aligned = list_all_extension_files_per_directory(directory_path=aligned_csvs, extension='csv')
    all_paths_aligned = list(np.hstack(all_paths_aligned)) if len(all_paths_aligned)>0 else []
    if not os.path.exists(aligned_csvs) or len(all_paths_aligned)!=len(input_file_paths):
        _ = d6tstack.combine_csv.CombinerCSV(input_file_paths).to_csv_align(aligned_csvs)

    bigrams_count = dd.read_csv(all_paths_aligned)
    if 'filepath' in bigrams_count.columns:
        bigrams_count = bigrams_count.drop(columns=["filepath"])
    if 'filename' in bigrams_count.columns:
        bigrams_count = bigrams_count.drop(columns=["filename"])
    bigrams_count = bigrams_count.reset_index(drop=True).rename(columns={"Unnamed: 0":"w0", "Unnamed: 1":"w1"})
    bigrams_count = bigrams_count.fillna(0).compute()

    print(f"Before first filter: {bigrams_count.shape}")
    bigrams_count = keep_common_bigrams(bigrams_count=bigrams_count, threshold=1)
    print(f"After first filter: {bigrams_count.shape}")
    bigrams_count = match_congressmen(bigrams_count, df_congressmen)
    print(f"After match congressmen: {bigrams_count.shape}")
    bigrams_count = keep_common_bigrams(bigrams_count=bigrams_count, threshold=1)
    print(f"After second filter: {bigrams_count.shape}")


    bigrams_count = bigrams_count.groupby(["w0","w1"]).sum() #Sums the bigrams of the same congressmen
    bigrams_count = bigrams_count.reset_index()


    X = bigrams_count[bigrams_count.columns.difference(["match_surname","party_code"])].values
    y = bigrams_count.party_code.values

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0]==y.shape[0]
    return X, y

def clean_bigrams_matrix(bigrams_count, df_congressmen, threshold_bigrams, threshold_speaker):
    print(f"Before filter on bigrams: {bigrams_count.shape}")
    bigrams_count = keep_common_bigrams(bigrams_count=bigrams_count, threshold=threshold_bigrams)
    print(f"After filter on bigrams: {bigrams_count.shape}")
    bigrams_count = match_congressmen(bigrams_count, df_congressmen)
    print(f"After match congressmen: {bigrams_count.shape}, the matrix is transposed ")
    # Since it was transposed this keep only the congressmen with a certain number of bigrams.
    bigrams_count = keep_common_bigrams(bigrams_count=bigrams_count, threshold=threshold_speaker)
    print(f"After speaker filter: {bigrams_count.shape}")
    return bigrams_count

def filter_bigrams_matrix(bigrams_count, threshold_bigrams, threshold_speaker):
    bigrams_count = bigrams_count.T
    print(f"Before filter on bigrams: {bigrams_count.shape}")
    bigrams_count = keep_common_bigrams(bigrams_count=bigrams_count, threshold=threshold_bigrams)
    print(f"After filter on bigrams: {bigrams_count.shape}")
    bigrams_count = bigrams_count.T
    # Since it was transposed this keep only the congressmen with a certain number of bigrams.
    bigrams_count = keep_common_bigrams(bigrams_count=bigrams_count, threshold=threshold_speaker)
    print(f"After speaker filter: {bigrams_count.shape}")
    return bigrams_count


def extract_data(input_file_paths, output_path, df_congressmen):
    df_congressmen['match_surname'] = df_congressmen.bioname.str.split(",", expand=True)[0].str.lower().values

    if len(input_file_paths)>0:
        all_bigrams_count = None
        for file in input_file_paths:
            print(f"File: {file}")
            bigrams_count = pd.read_csv(file)
            if 'filepath' in bigrams_count.columns:
                bigrams_count = bigrams_count.drop(columns=["filepath"])
            if 'filename' in bigrams_count.columns:
                bigrams_count = bigrams_count.drop(columns=["filename"])

            bigrams_count = bigrams_count.rename(columns={"Unnamed: 0": "w0", "Unnamed: 1": "w1"})
            bigrams_count = bigrams_count.fillna(0)

            bigrams_count = clean_bigrams_matrix(bigrams_count=bigrams_count, df_congressmen=df_congressmen,
                                                 threshold_bigrams=1, threshold_speaker=20)

            if all_bigrams_count is None:
                all_bigrams_count = bigrams_count
            else:
                if bigrams_count.shape[0]>0:
                    all_bigrams_count = pd.concat([all_bigrams_count, bigrams_count])

        all_bigrams_count = all_bigrams_count.fillna(0.0)
        all_bigrams_count = all_bigrams_count.groupby(level=0).sum(axis=1)
        all_bigrams_count = filter_bigrams_matrix(all_bigrams_count, threshold_bigrams=1, threshold_speaker=20)

        all_bigrams_count = all_bigrams_count.merge(df_congressmen.set_index("match_surname")["party_code"], how="left",
                                            left_index=True, right_index=True)

        #TODO: Make sure that one speaker is associated to ONLY one party code, it can happen that is associated with more
        all_bigrams_count.to_csv(output_path)
        return all_bigrams_count
    else:
        raise NotImplementedError

def load_data(bigrams_df_path):

    y = bigrams_count.party_code.values
    all_bigrams_count = bigrams_count.drop(columns=['party_code'])
    X = all_bigrams_count.values

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
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


def group_volumes_by_congresses(input_path, dict_congresses):
    all_files = list_all_extension_files(input_path)
    files_divided_by_congress = {i: list(np.hstack([[file for file in all_files if file.find("/"+ele+"/")>=0] for ele in dict_congresses[i]])) for i in dict_congresses}
    return files_divided_by_congress


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bigrams Analysis file')
    parser.add_argument('--input_files_path', type=str, required=True, help='Directory containig the .csv files')
    parser.add_argument('--congressmen_csv', type=str, default="./HSall_members.csv", help='Path to the congressmen csv file')
    parser.add_argument('--type', type=str, required=True, help='Can be "volumes", "items", "congresses"')
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

    dir_output_path = output_path.replace("df_tuples", "bigrams")
    if not os.path.exists(dir_output_path):
        os.makedirs(dir_output_path)
    if args.type=='volumes':
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

        congresses_files = group_volumes_by_congresses(input_path=output_path, dict_congresses=dict_congresses_volumes)
        for congress in congresses_files.keys():
            print(f"Analysis of congress {congress}")
            file_output_path = os.path.join(dir_output_path, "congress_"+str(congress)+".csv")
            if not os.path.exists(file_output_path):
                df_congressmen_filtered = df_congressmen[df_congressmen.congress==congress]
                bigrams_count = extract_data(input_file_paths=congresses_files[congress], output_path=file_output_path,
                                    df_congressmen=df_congressmen_filtered)

            else:
                bigrams_count = pd.read_csv(file_output_path, index_col=0)
            X, y = load_data(bigrams_count)
            analysis_RandomizedSearch(X, y)


    print(f"Job finished. Analysis of path: {args.input_files_path} completed")

