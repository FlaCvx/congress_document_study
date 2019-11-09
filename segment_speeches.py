import argparse
import os
import numpy as np
import pandas as pd
import re
from pathlib import Path

WEEK_DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']


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
    files_paths = [tuple((in_file, in_file.replace("text_volumes","speeches").replace(".txt",".csv"))) for in_file in files_paths]

    files_paths = [tuple((in_file,out_file)) for in_file, out_file in files_paths if not os.path.exists(out_file)]
    return files_paths


def find_all_mrs_separator(text):
    occurrences = [m.start(0) for m in re.finditer("Mr.", text)]
    return occurrences


def find_all_weekdays_separator(text):
    occurrences = np.hstack([[m.start(0) for m in re.finditer(day, text)] for day in WEEK_DAYS])
    return occurrences


def find_closest_stop(start_index, all_occurrences):
    tmp_occ = [ele for ele in all_occurrences if ele>start_index]
    if len(tmp_occ) == 0:
        return None
    return int(min(tmp_occ))


def calculate_start_and_stop_occurences(text):
    start_and_stop = []
    occurrences_mrs = find_all_mrs_separator(text)
    occurrences_days = find_all_weekdays_separator(str.lower(text))
    occurrences = list(np.hstack([occurrences_days, occurrences_mrs]))
    occurrences.sort(reverse=True)

    for occ_mr in occurrences_mrs:
        start_and_stop.append(tuple((occ_mr, find_closest_stop(start_index=occ_mr, all_occurrences=occurrences))))

    return start_and_stop


def extract_following_name(text):
    # TODO: May implement a fancier way of doing this. But not at the moment.
    # try:
    #     import re
    #     NameRegex = re.compile(r'Mr. [A-Z][A-z]+')
    #     p = NameRegex.search(text)
    #     name = p.group()
    # except:
    #     print(text)

    name = (' '.join(text.split(" ", maxsplit=2)[:2])).strip()
    return name.strip()


def extract_name_and_speech(text, start, stop):
    name = extract_following_name(text[start:stop])
    return name.strip().upper(), text[start:stop].replace(name, "").strip()


def segment(input_file_paths):

    input_output_file_paths = list_all_extension_files(input_file_paths)

    for input_file_path, output_file_path in input_output_file_paths:
        speaker_speeches = {}
        if not os.path.exists(Path(output_file_path).parent):
            os.makedirs(Path(output_file_path).parent)

        with open(input_file_path, 'r') as input_file:
            loaded_text = input_file.read(-1)
            loaded_text = loaded_text.replace("-\n", "").replace("\n", " ")
            start_stop_indexes = calculate_start_and_stop_occurences(loaded_text)
            all_speakers = pd.DataFrame()

            for start_stop in start_stop_indexes:
                name, speech = extract_name_and_speech(loaded_text, start=start_stop[0], stop=start_stop[1])
                if name not in speaker_speeches:
                    speaker_speeches[name] = []
                speaker_speeches[name].append(speech)

            for name in speaker_speeches.keys():
                tmp = pd.DataFrame(data=[speaker_speeches[name]])
                tmp['name'] = name
                all_speakers = all_speakers.append(tmp)
            all_speakers.reset_index(drop=True).to_csv(path_or_buf=output_file_path, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Segment file')
    parser.add_argument('--input_files_path', type=str, required=True, help='Directory containig the .txt files' )

    args = parser.parse_args()

    initial_path = args.input_files_path
    final_path = args.input_files_path.replace("text_volumes","speeches")

    print(f"Starting segmentation job for Path: {initial_path}")
    segment(input_file_paths=args.input_files_path)
    print(f"Job finished. Path: {final_path} completed")
