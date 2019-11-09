import argparse
import argparse
import subprocess
import sys
import os
import glob
from PIL import Image, ImageOps
from bs4 import BeautifulSoup
from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
from operator import itemgetter
import scipy.cluster.hierarchy as hcluster
from pathlib import Path
import csv


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Implement the whole pipeline for speech extraction')
    parser.add_argument('--num_volume', type=int, required=False, help='Number of the volume where to take the .hocr files.' )
    #parser.add_argument('--data_path', type=int, required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--name_vol', type=int, required=True, help='Name of the volumes. ')
    args = parser.parse_args()
    name_vol = args.name_vol+str(args.year)

    if args.year == 1789:
        data_path = '/home/fla/Desktop/Research_Assistantship/Congress_Documents_Scraping/data/' \
                    '1789to1824_DebatesAndProceedings/volumes'
    elif args.year == 1824:
        data_path = '/home/fla/Desktop/Research_Assistantship/Congress_Documents_Scraping/data/' \
                    '1789to1824_DebatesAndProceedings/volumes'
        #TODO: Change this path
    elif args.year == 1833:
        data_path = '/home/fla/Desktop/Research_Assistantship/Congress_Documents_Scraping/data/' \
                    '1789to1824_DebatesAndProceedings/volumes'
        # TODO: Change this path
    else:
        raise ValueError('The inserted value for --year is incorrect. Choices: [1789, 1833, 1894]')

    assert data_path.split("/")[-1]=='volumes'



    command = 'tesseract  ' + str(input_page) + ' ' + str(input_page) + ' -l eng hocr'
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

