import argparse
import os
import numpy as np
import pandas as pd
import re
from pathlib import Path


def list_all_extension_files(directory_path, extension='.csv'):
    """

    List all the files in the directory directory_path that have .txt as extension, check if the corresponding .csv file
    path exists. This path has also "speeches" instaed of "text_volumes.

    :param directory_path:
    :return:
    """
    files_paths = []
    for r, d, f in os.walk(directory_path):
        f = [os.path.join(r,file) for file in f if file.find(extension) != -1]
        files_paths.append(f)

    files_paths = list(np.hstack(files_paths))

    return files_paths


def reformat_name(name, congressmen_df):
    if name not in list(congressmen_df.surname):

        regex_one = "^(M|m)(R|r)(\.)?(\s)?[a-zA-Z]?" #Covers Mr. Potter/MR.Potter/mR.Potter/Mr Potter...
        regex_two = "^(M|m)(R|r)(\.)?(\s)?[a-zA-Z]+(\.)(\s)?[a-zA-Z]+" #Covers Mr. H. Potter
        regexes = [regex_one, regex_two]
        for reg in regexes:
            pattern = re.search(reg, name)
            new_name = name[pattern.start():pattern.end()]

    else:
        if congressmen_df[congressmen_df.bioname==name].size[0]>1:
            raise("Too many congMen associated with this name")


def reformat_df(input_path, congressmen_df):
    speeches_df = pd.read_csv(input_path)

    new_names = speeches_df.name.apply(lambda name: reformat_name(name, congressmen_df))
    pass


def load_congressmen_df(congressmen_csv_path):
    congressmen_df = pd.read_csv(congressmen_csv_path)
    congressmen_df['surname'] = congressmen_df.bioname.str.split(" ", expand=True)[0].to_list()
    return congressmen_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Segment file')
    parser.add_argument('--input_files_path', type=str, required=True, help='Directory containig the .txt files' )
    parser.add_argument('--congressmen_csv', type=str, required=True, help='Directory containig the .csv of the congressmen info' )
    args = parser.parse_args()

    input_paths = list_all_extension_files(args.input_files_path)
    congressmen_df = load_congressmen_df(args.congressmen_csv)

    for path in input_paths:
        print(f"Refactoring df in: {path}")
        reformat_df(input_path=path, congressmen_df=congressmen_df)
