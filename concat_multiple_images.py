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

def list_all_extension_files(directory_path, extension='.txt'):
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

def rearrange_page(input_file_paths, num_pages, try_double=False):

    count = 0
    input_paths = list_all_extension_files(input_file_paths, 'png')

    input_paths = input_paths[:num_pages]
    #TODO: Need to sort
    images = []
    for im_path in input_paths:
        images.append(Image.open(im_path))

    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)
    if try_double:
        new_im = Image.new('RGB', (2*max_width, total_height))
    else:
        new_im = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    if try_double:
        new_im.paste(new_im, (max_width, 0))

    new_im.show()
    path = os.path.join("/home/fla/Desktop/Research_Assistantship/congress_document_study/data/"
                        "1789to1824_DebatesAndProceedings/scratch_dir", str(num_pages)+"_pages.png")
    new_im.save(path)
    print(f"Image dimensions: {new_im.size}")
    print(f"Saved in: {path}")


    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/fla/Desktop/Research_Assistantship/congress_document_study/data/credentials.json"  # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs text detection on the image file
    response = client.text_detection(image=image)
    texts = response.text_annotations
    print(texts[0].description)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Segment file')
    parser.add_argument('--input_files_path', type=str, required=True, help='Directory containig the .txt files' )
    parser.add_argument('--num_pages', type=int, required=False, default=3, help='Directory containig the .txt files' )

    args = parser.parse_args()

    initial_path = args.input_files_path
    final_path = args.input_files_path.replace("text_volumes","speeches")

    rearrange_page(input_file_paths=args.input_files_path, num_pages=args.num_pages)
