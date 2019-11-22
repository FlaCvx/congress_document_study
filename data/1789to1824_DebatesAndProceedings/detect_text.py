import io
import os
import argparse
import numpy as np
from pathlib import Path

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types


def list_all_pngs_files(directory_path):
    """

    List all the files in the directory directory_path that have .hocr as extension

    :param directory_path:
    :return:
    """
    files_paths = []
    for r, d, f in os.walk(directory_path):
        f = [os.path.join(r,file) for file in f if file.find(".png")!=-1]
        files_paths.append(f)

    return list(np.hstack(files_paths))


def detect_document(path):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                print('Paragraph confidence: {}'.format(
                    paragraph.confidence))

                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    print('Word text: {} (confidence: {})'.format(
                        word_text, word.confidence))

                    for symbol in word.symbols:
                        print('\tSymbol: {} (confidence: {})'.format(
                            symbol.text, symbol.confidence))


    print(response)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Detect text in image')

    parser.add_argument('--path_pngs', type=str, required=True, help='Path containing .pngs\' files ')

    args = parser.parse_args()


    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"] = "/home/fla/Desktop/Research_Assistantship/Congress_Documents_Scraping" \
                                            "/data/credentials.json"  # Instantiates a client
    client = vision.ImageAnnotatorClient()


    files_paths = list_all_pngs_files(args.path_pngs)

    for file_path in files_paths:
        new_directory = Path(file_path.replace("one_column_oriented","text_volumes")).parent
        new_name = file_path.replace("one_column_oriented","text_volumes").replace("png","txt")
        if not os.path.exists(new_name):
            if not os.path.exists(new_directory):
                os.makedirs(new_directory)

            # Loads the image into memory
            with io.open(file_path, 'rb') as image_file:
                content = image_file.read()

            image = types.Image(content=content)

            # Performs text detection on the image file
            response = client.text_detection(image=image)
            texts = response.text_annotations
            if len(texts)==0:
                description = ""
            else:
                description = texts[0].description

            with io.open(new_name, 'w') as text_file:
                text_file.write(description)
                print(f" Wrote the file: {new_name}")
