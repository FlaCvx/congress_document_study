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

def filter_pages_without_hocr(paths):
    """
    Removes from paths all the paths that have the hocr already. It returns a list of path for which the .hocr
     needs to be created

    :param paths:
    :return:
    """
    input_pages = [fi for fi in paths if ((fi.find(".hocr") == -1)&(fi.find("one_column") == -1))]
    output_pages = [fi.replace(".hocr","") for fi in paths if fi.find(".hocr") != -1]
    return list(set(input_pages)-set(output_pages))


def create_hocr(path_volume):
    """
    Given a path_volume the objective of this function is to create the .hocr for each image inside the "path_volume"
    directory.
    First list all the files in path_volume, then filters out the pages for which the .hocr already exists.
    Then creates the .hocr for these remaining pages.

    :param path_volume:
    :return:
    """
    files_paths = list_all_files(path_volume)
    input_pages = filter_pages_without_hocr(files_paths)

    for input_page in input_pages:
        try:
            command = 'tesseract  ' + str(input_page) + ' ' + str(input_page) + ' -l eng hocr'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            if not os.path.exists(str(input_page).replace("volumes", "hocrs_files").replace(input_page.split("/")[-1],"")):
                os.makedirs(str(input_page).replace("volumes", "hocrs_files").replace(input_page.split("/")[-1],""))
            shutil.move(str(input_page)+".hocr", str(input_page).replace("volumes", "hocrs_files") + ".hocr")
            print("The file: \"%s\" has been created" % (str(input_page).replace("volumes", "hocrs_files") + ".hocr"))

        except KeyboardInterrupt:
            if os.path.exists(str(input_page + '.hocr')):
                print("Before the INTERRUPT, this file was being created: " + str(input_page + '.hocr'))
                print("This file is now being deleted.")
                os.remove(input_page + '.hocr')

    print("All the files of volume " + str(path_volume.split("/")[-1]) + " have been created")

    try:
        for input_page in list_all_files_hocr(path_volume):
            shutil.move(str(input_page), str(input_page).replace("volumes", "hocrs_files"))
    except KeyboardInterrupt:
        print("Process of moving .hocr files not finished yet")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract .hocr files')

    parser.add_argument('--num_volume', type=int, required=False, help='Number of the volume where to convert the files in .hocr .')

    default_path = "./volumes"

    args = parser.parse_args()
    if args.num_volume:
        print(f"Creating .hocr of Volume: { args.num_volume}")
        create_hocr(default_path+"/Volume_"+str(args.num_volume))
    else:
        for volume_path in os.listdir(default_path):
            print(f"Creating .hocr of Volume: {volume_path}")
            create_hocr(os.path.join(default_path, volume_path))
