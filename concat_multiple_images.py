import argparse
import os
import numpy as np
import pandas as pd
import re
from pathlib import Path
from PIL import Image
# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
import io
from operator import itemgetter, attrgetter


def list_all_extension_files_and_sort(directory_path, extension='.txt'):
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

    files_paths = [paths for paths in files_paths if len(paths)>0]
    files_paths = [[tuple((path, int(path.replace(".png","").split("/")[-1].split("_")[-1]))) for path in paths] for paths in files_paths]
    files_paths = [sorted(paths, key=itemgetter(1)) for paths in files_paths]
    files_paths = [[path[0] for path in paths]for paths in files_paths]


    return files_paths


def rearrange_page(input_file_paths, num_pages):


    all_directories_paths = list_all_extension_files_and_sort(input_file_paths, 'png')

    for dir_paths in all_directories_paths:
        count = -1
        output_directory = str(Path(dir_paths[0]).parent).replace("one_column_oriented", "concat_pages")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for i in range(0,len(dir_paths),num_pages):
            count += 1
            output_path = os.path.join(output_directory, "page_" + str(count) + ".png")
            if not os.path.exists(output_path):
                concat_pages_paths = dir_paths[i:i+num_pages]
                images = []
                for page_path in concat_pages_paths:
                    try:
                        images.append(Image.open(page_path))
                    except:
                        pass
                widths, heights = zip(*(i.size for i in images))
                max_width = max(widths)
                total_height = sum(heights)
                new_im = Image.new('RGB', (max_width, total_height))
                y_offset = 0
                for im in images:
                    try:
                        new_im.paste(im, (0, y_offset))
                        y_offset += im.size[1]
                    except:
                        pass
                #new_im.show()


                new_im.save(output_path)
                print(f"Saved image: {output_path}")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Segment file')
    parser.add_argument('--input_files_path', type=str, required=True, help='Directory containig the .txt files' )
    parser.add_argument('--num_pages', type=int, required=False, default=3, help='Directory containig the .txt files' )

    args = parser.parse_args()

    initial_path = args.input_files_path

    rearrange_page(input_file_paths=args.input_files_path, num_pages=args.num_pages)
