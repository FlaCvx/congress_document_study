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

def list_all_hocr_files(directory_path):
    """

    List all the files in the directory directory_path that have .hocr as extension

    :param directory_path:
    :return:
    """
    files_paths = []
    for r, d, f in os.walk(directory_path):
        f = [os.path.join(r,file) for file in f if file.find(".hocr")!=-1]
        files_paths.append(f)

    return list(np.hstack(files_paths))

def most_common_borders(soup, num_occurrences=2, n_most_common=10):
    """
    The objective of this function is to find the borders of the line that are the closest to the column separator.

    1- It finds the information about the position of each line.
    2- It collects the values of the left and right borders.
    3- It keeps the most common borders, which are probably the true borders of the column.

    :param soup:
    :param num_occurrences:
    :return:
    """
    span_lines_classes = [span for span in soup.find_all('span') if span.get("id").find("line")!=-1 ]
    lines = [(span_line_class.get("title").split(";")[0]).replace("bbox", "").strip() for span_line_class in span_lines_classes]
    approx_left_border = [int(line.split(" ")[0]) for line in lines]
    approx_right_border = [int(line.split(" ")[2]) for line in lines]
    boxes = [{'l': int(line.split(" ")[0]), 'u': int(line.split(" ")[1]), 'r': int(line.split(" ")[2]),
              'b': int(line.split(" ")[3])} for line in lines]

    #approx_left_border = [k for k in approx_left_border if Counter(approx_left_border)[k] < num_occurrences]
    #approx_right_border = [k for k in approx_right_border if Counter(approx_right_border)[k] < num_occurrences]
    left_most_common = [ele[0] for ele in Counter(approx_left_border).most_common(n_most_common)]
    approx_left_border = [k for k in approx_left_border if k in left_most_common]

    right_most_common = [ele[0] for ele in Counter(approx_right_border).most_common(n_most_common)]
    approx_right_border = [k for k in approx_right_border if k in right_most_common]

    return boxes, approx_left_border, approx_right_border

def find_borders_ranges(approx_border, num_cluster):
    X = np.array(approx_border).reshape(-1, 1)
    kmeans_class = KMeans(n_clusters=num_cluster, random_state=0).fit(X)

    borders_ranges = {}
    for cluster in list(range(num_cluster)):
        b = [ind for ind in np.where(kmeans_class.labels_ == cluster)[0]]
        a = approx_border
        borders_ranges[cluster] = (
        min(itemgetter(*b)(a)), kmeans_class.cluster_centers_[cluster][0], max(itemgetter(*b)(a)))


    sorted_borders_ranges = {}
    for element, pos in zip(sorted(borders_ranges.items(), key = lambda x : x[1]), list(range(0,len(borders_ranges),1))):
        sorted_borders_ranges[pos] = element[1]

    return sorted(kmeans_class.cluster_centers_), sorted_borders_ranges

def find_border_of_columns(left_borders, right_borders):
    #Left and right should be ordered list of one-element array.
    return np.concatenate((left_borders, right_borders), axis=1)

def find_columns_split(boxes, columns_borders, left_borders_ranges, right_borders_ranges ):
    if(columns_borders.shape[0]>1):
        up=sys.maxsize
        bottom=0
        middle_line = np.empty((columns_borders.shape[0], 0)).tolist()
        for box in boxes:
            for column in list(range(columns_borders.shape[0])):
                if( left_borders_ranges[column][0] <= box['l'] <= left_borders_ranges[column][2]):
                    if (right_borders_ranges[column][0] <= box['r'] <= right_borders_ranges[column][2]):
                        up = box['u'] if box['u']< up else up
                        bottom = box['b'] if box['b']>bottom else bottom
                        if (column >= 1 ):
                            middle_line[column-1].append(box['l'])
                            middle_line[column].append(box['r'])
                        else:
                            if(column==0):
                                middle_line[column].append(box['r'])
        return middle_line, up, bottom
    else:   #There is only one column, no split needed.
        return -1, -1, -1

def check_that_are_valid_column(columns_borders):
    return all(element == True for element in np.diff(np.hstack((columns_borders)))>=0)

def find_num_cluster(threshold, approx_left_border, approx_right_border, page):

    data_left = np.array(approx_left_border).reshape(-1, 1)
    data_right = np.array(approx_right_border).reshape(-1, 1)
    # clustering
    clusters_left = hcluster.fclusterdata(data_left, t=threshold, criterion="distance")
    num_clusters_left = len(set(clusters_left))
    for label in set(clusters_left):
        new_l = len(np.argwhere(clusters_left==label))
        if (new_l < 0.2*len(clusters_left)):
            num_clusters_left -= 1

    clusters_right = hcluster.fclusterdata(data_right, t=threshold, criterion="distance")
    num_clusters_right = len(set(clusters_right))
    for label in set(clusters_right):
        new_l = len(np.argwhere(clusters_right==label))
        if (new_l < 0.2*len(clusters_right)):
            num_clusters_right -= 1

    if (num_clusters_left ==num_clusters_right):
        if(num_clusters_left<=2):
            return num_clusters_left
        else:
            raise ValueError('The number of cluster for page: '+str(page)+'is too much.')
    else:
        raise ValueError('The number of cluster for page: '+str(page)+'was not uniquely found.')

def rearrange_page(lines_split, up, bottom, num_columns, page):
    original_file = Image.open(page)

    p = Path(page)
    new_image_path = str(p.parent).replace("volumes","one_column_volumes")
    if not os.path.exists(new_image_path):
        os.makedirs(new_image_path)
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
        new_im.save(os.path.join(new_image_path, p.stem+'.png'))
    elif(num_columns==1):
        original_file.save(os.path.join(new_image_path, p.stem+'.png'))
    else:
        raise NotImplementedError
    return


def split_pages(dir_path, default_path):
    """
    Algorithm based on a clustering technique.

    :param path:
    :return:
    """
    num_occurrences=5 #Number of time that a border has to appear in order to be consider a border.
    threshold=50

    bad_pages = []
    hocr_files_paths = list_all_hocr_files(dir_path)

    for path in hocr_files_paths:
        basepath = path.replace(".hocr","")
        if not os.path.exists(path.replace(default_path,"one_column_volumes").replace(".hocr",".png")):
            hocr_file = open(path, "r")
            soup = BeautifulSoup(hocr_file, 'html.parser')
            boxes, approx_left_border, approx_right_border = most_common_borders(soup=soup, num_occurrences=num_occurrences)
            try:
                num_cluster=find_num_cluster(threshold=threshold, approx_left_border=approx_left_border,
                                             approx_right_border=approx_right_border, page=basepath)

                left_borders, left_borders_ranges = find_borders_ranges(approx_left_border, num_cluster)
                right_borders, right_borders_ranges = find_borders_ranges(approx_right_border, num_cluster)
                columns_borders = find_border_of_columns(left_borders, right_borders)   # It is a matrix. Where each row is a tuple of (left_border, right_border).
                if ( check_that_are_valid_column(columns_borders)):
                    lines_split, up, bottom = find_columns_split(boxes=boxes, columns_borders=columns_borders,
                                       left_borders_ranges=left_borders_ranges, right_borders_ranges=right_borders_ranges)
                else:
                    raise ValueError('The order of the columns border is not increasing in page: '+str(basepath))
                rearrange_page(lines_split=lines_split, up=up, bottom=bottom, num_columns=num_cluster,
                               page=basepath.replace("hocrs_files","volumes"))
                print("Page " + str(Path(basepath).stem) + ", completed.")
            except (ValueError, TypeError):
                bad_pages.append(basepath)
    print("The following pages could not be modified: ",bad_pages)
    with open(dir_path+"/one_column_not_modified_pages.csv", 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for element in bad_pages:
            wr.writerow([element])

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract .hocr files')
    parser.add_argument('--num_volume', type=int, required=False, help='Number of the volume where to take the '
                                                                       '.hocr files.')
    default_path = "./hocrs_files"

    args = parser.parse_args()

    if args.num_volume:
        split_pages(os.path.join(default_path,"Volume_"+str(args.num_volume)), default_path=default_path)
    else:
        for volume_path in os.listdir(default_path):
            split_pages(os.path.join(default_path,"Volume_"+volume_path), default_path=default_path)
