import argparse
import subprocess
import os
import shutil


def list_all_files(directory_path):
    """
    List all the files in the directory directory_path
    It returns a list of files as a complete os path
    :param directory_path:
    :return:
    """
    files_paths = []
    for r, d, f in os.walk(directory_path):
        for file in f:
            files_paths.append(os.path.join(r,file))
    return files_paths

def list_all_files_hocr(directory_path):
    """
    List all the files in the directory directory_path
    It returns a list of files as a complete os path
    :param directory_path:
    :return:
    """
    files_paths = []
    for r, d, f in os.walk(directory_path):
        for file in f:
            if file.find(".hocr")>=0:
                files_paths.append(os.path.join(r,file))
    return files_paths


def create_hocr(path, path_type):
    """
    Given a path the objective of this function is to create the .hocr for each image inside the "path"
    directory.
    First list all the files in path, then filters out the pages for which the .hocr already exists.
    Then creates the .hocr for these remaining pages.

    :param path:
    :return:
    """
    input_pages = list_all_files(path)

    for input_page in input_pages:
        out_dir = str(input_page).replace(path_type, "hocrs_files").replace(str(input_page).split("/")[-1], "")
        out_name = str(input_page).replace(path_type, "hocrs_files") + ".hocr"

        if not os.path.exists(out_name):
            try:
                command = 'tesseract  ' + str(input_page) + ' ' + str(input_page) + ' -l eng hocr'
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                shutil.move(str(input_page)+".hocr", out_name)
                print("The file: \"%s\" has been created" % out_name)

            except KeyboardInterrupt:
                if os.path.exists(str(input_page + '.hocr')):
                    print("Before the INTERRUPT, this file was being created: " + str(input_page + '.hocr'))
                    print("This file is now being deleted.")
                    os.remove(input_page + '.hocr')

    print("All the files of path: " + str(path.split("/")[-1]) + " have been created")

    try:
        for input_page in list_all_files_hocr(path):
            shutil.move(str(input_page), str(input_page).replace(path_type, "hocrs_files"))
    except KeyboardInterrupt:
        print("Process of moving .hocr files not finished yet")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract .hocr files')

    parser.add_argument('--path_files', type=str, required=True, help='Paths of the files to do .hocr')


    args = parser.parse_args()

    print(f"Creating .hocr for path: { args.path_files}")

    if args.path_files.find("/volumes/")>=0:
        path_type='volumes'
    elif args.path_files.find("/items/")>=0:
        path_type = 'items'
    elif args.path_files.find("/congresses/")>=0:
        path_type = 'congresses'
    else:
        raise ValueError(f"Path not valid, because it not contains neither \"volumes\" or \"items\" or \"congresses\"")

    create_hocr(args.path_files, path_type)

