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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import make_scorer

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

def parse_vector(column_string):
    vectors = column_string.str.split(expand=True)
    new_vectors = []
    for row in vectors.iterrows():
        row = row[1].astype(float).values
        new_vectors.append(row)
    return new_vectors


def parse_embedding_info(speeches_df):

    sp = speeches_df['speeches'].str.replace("[","").str.replace("]","").str.replace("\n","")\
        .str.replace("(","").str.replace(")","").str.replace(",","").str.split("array", expand=True)
    sp = sp.drop(columns=0)

    for col in sp.columns:
        sp[col] = parse_vector(sp[col])

    return sp


def mean_speeches(speeches):

    averages_by_row = []
    for row in speeches.iterrows():
        row = row[1]
        avgs = [ele for ele in row if not np.isnan(ele).any()]
        if len(avgs)==0:
            averages_by_row.append(row.iloc[0])
        else:
            averages_by_row.append(np.mean(avgs, axis=0).reshape((1,300)))

    averages_by_row = np.array(averages_by_row)
    averages_by_row = np.vstack(averages_by_row)
    return averages_by_row


def load_speeches(file_path):
    speeches = pd.read_csv(file_path, index_col=0)
    names_party = speeches[['name','party_code']]
    speeches = parse_embedding_info(speeches)
    speeches = mean_speeches(speeches)

    remove_elements = np.unique(np.where(np.isnan(speeches))[0])
    X = np.delete(speeches, remove_elements, 0)
    y = np.delete( names_party.party_code.values, remove_elements, 0)
    return X, y

def analysis_RandomizedSearch(X, y):

    clf = RandomForestClassifier(n_estimators=20)

    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3,10, 20, None],
                  "max_features": [1, 5, 10],
                  "min_samples_split": [3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)

    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % ((time() - start), n_iter_search))
    print(f"Accuracy score: {random_search.best_score_}")

    best_model = random_search.best_estimator_

    print(best_model)
    return random_search

def analysis_GridSearch(X, y):

    # build a classifier
    clf = RandomForestClassifier(n_estimators=20)
    # Utility function to report best scores
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})"
                      .format(results['mean_test_score'][candidate],
                              results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    # use a full grid over all parameters
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10, 20, 30],
                  "min_samples_split": [3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run grid search
    cv = KFold(10)
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv, scoring='balanced_accuracy')
    start = time()
    grid_search.fit(X, y)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)

    best_model = grid_search.best_estimator_
    def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
    def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
    def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
    def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

    from sklearn.metrics import recall_score, balanced_accuracy_score

    def my_precision(y_true, y_pred):
        if np.unique(y_true).__len__()==2:
            return precision_score(y_true, y_pred, pos_label=y_true[0])
        else:
            return precision_score(y_true, y_pred, pos_label=y_true[0], average='weighted')

    def my_recall(y_true, y_pred):
        if np.unique(y_true).__len__()==2:
            return recall_score(y_true, y_pred, pos_label=y_true[0])
        else:
            return recall_score(y_true, y_pred, pos_label=y_true[0], average='weighted')

    def my_f1(y_true, y_pred):
        if np.unique(y_true).__len__()==2:
            return f1_score(y_true, y_pred, pos_label=y_true[0])
        else:
            return f1_score(y_true, y_pred, pos_label=y_true[0], average='weighted')

    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
               'fp': make_scorer(fp), 'fn': make_scorer(fn),
               'accuracy':'accuracy', 'balanced_accuracy':'balanced_accuracy', 'precision': make_scorer(my_precision),
               'recall':make_scorer(my_recall), 'f1_score':make_scorer(my_f1)}

    cv_results = cross_validate(best_model, X, y, cv=5, scoring=scoring)
    # Getting the test set true positive scores
    for score in scoring.keys():
        scr = 'test_'+score
        mean_scr = np.mean(cv_results[scr])
        print(f"Mean {scr}: {mean_scr}")

    print()
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--congresses_embedded', type=str, required=True, help='Directory containig the .csv files of '
                                                                               'the congressmen speeches embedded' )

    args = parser.parse_args()

    congresses_files = list_all_extension_files(directory_path=args.congresses_embedded, extension='.csv')
    for file in congresses_files:
        print(f"Analysis of congress file: {file}")
        X, y = load_speeches(file)
        analysis_GridSearch(X,y)