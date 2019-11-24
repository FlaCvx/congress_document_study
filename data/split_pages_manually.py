import argparse
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
import string

def list_all_files(directory_path, extension='png'):
    """

    List all the files in the directory directory_path that have .hocr as extension

    :param directory_path:
    :return:
    """
    files_paths = []
    for r, d, f in os.walk(directory_path):
        f = [os.path.join(r,file) for file in f if file.find(extension) > -1]
        files_paths.append(f)

    return list(np.hstack(files_paths))


def rearrange_page(lines_split, up, bottom, num_columns, original_file, output_file):
    original_file = Image.open(original_file)

    if not os.path.exists(Path(output_file).parent):
        os.makedirs(Path(output_file).parent)
    if num_columns==2:
        # Cut the left page.
        middle_point = np.mean(lines_split[0])
        left_border = (0, up-10, (original_file.size[0] - middle_point), 10)  # left, up, right, bottom
        # ImageOps.crop(original_file, left_border).show()
        left_imm = ImageOps.crop(original_file, left_border)

        # Cut the right page.
        right_border = (middle_point, up-10, -1, 10)  # left, up, right, bottom
        # ImageOps.crop(original_file, border).show()
        right_imm = ImageOps.crop(original_file, right_border)
        images = [left_imm, right_imm]
        widths, heights = zip(*(i.size for i in images))
        total_width = max(widths)
        max_height = sum(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]
        #new_im.show()
        new_im.save(output_file)
    elif(num_columns==1):
        original_file.save(output_file)
    else:
        raise NotImplementedError
    return

def split_pages_manually(image_path, splitting_point, save_imm=False):
    original_file = Image.open(image_path)
    if original_file.size[0]>600:
        print(f"Splitting the page: {image_path}")
        if splitting_point is None:
            splitting_point=int(original_file.size[0]/2)

        left_border = (0, 0, (original_file.size[0] - splitting_point), 10)  # left, up, right, bottom
        # ImageOps.crop(original_file, left_border).show()
        left_imm = ImageOps.crop(original_file, left_border)

        right_border = (splitting_point, 0, -1, 10)  # left, up, right, bottom
        # ImageOps.crop(original_file, border).show()
        right_imm = ImageOps.crop(original_file, right_border)
        images = [left_imm, right_imm]
        widths, heights = zip(*(i.size for i in images))
        total_width = max(widths)
        max_height = sum(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]

        if save_imm:
            new_im.save(image_path)
        else:
            new_im.show()
    else:
        pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Splitting pages')
    parser.add_argument('--image_path', type=str, required=True, help='Path where to find the image to split')
    parser.add_argument('--splitting_point', type=int, default=None, required=False, help='Pixels location where to perform the cut')
    parser.add_argument('--save', action='store_true', help='Put the flag if you want to save, do not put it if you only want to show the imm')

    args = parser.parse_args()

    for file in list_all_files(args.image_path):
        split_pages_manually(image_path=file, splitting_point=args.splitting_point, save_imm=args.save)
