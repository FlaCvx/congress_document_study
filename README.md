# Polarization of Speeches Project.
This project is ideally divided in three parts. First part with speeches from 1789 to 1824, a second part 
from 1824 to 1837, the third from 1833 to 1873. There are five main steps before the final analysis.
1- Dataset Creation (Scraping of the websites)
2- Creation of image format information.
3- Page splitting
3- Text detection
4- Speech segmentation
5- Bigram analysis


#### 1. DATASET CREATION
The dataset can be found in the ./data directory, it is divided into three subdirectories.
1- "./data/1789to1824_DebatesAndProceedings" scraped from the link: "https://memory.loc.gov/ammem/amlaw/lwaclink.html"
2- "./data/1824to1837_DebatesAndProceedings" scraped from the link: "https://memory.loc.gov/ammem/amlaw/lwrdlink.html"
3- "./data/1833to1873_Debates andProceedings". scraped from the link: "https://memory.loc.gov/ammem/amlaw/lwcglink.html"

Each of the three data subdirectories contain a scraping file named "<date1>to<date2>_DebatesAndProceedings.py". 
This file can be executed in parallel, it scrapes the web and saves the images.
```console
python <date1>to<date2>_DebatesAndProceedings.py
```

#### 1.2 CREATION OF IMAGE FORMAT INFORMATION.
In order to take spatial information of the lines of the document, we use a script called "Create_hocrs.py".
This file uses the libray "tesseract" to create the .hocr file of each image. 
The .hocr file given as output will be used later to split the image, because it contains
spatial information of each line of the page.
```console
python Create_hocrs.py --path_files ./1789to1824_DebatesAndProceedings/volumes/
```
 
#### 1.3 PAGE SPLITTING
Almost all the images are oriented column-wise. In order to use the Google Cloud Vision API, 
we need to segment these columns and create a single column file. This is done in the "split_pages.py" file.
```console
python split_pages.py --hocr_paths ./1789to1824_DebatesAndProceedings/hocrs_files --page_type volumes
```
Future improvement: This algorithm fails some times if the page is a little be inclined. Could be adjusted.  


#### 1.4 CONCAT MULTIPLE IMAGES
From now, all the steps MUST be run once all the previous steps are completed. At least once every
subdirectory is completed, otherwise it will concatenate images and may skip others that are not yet available.. 

This step is executed to reduce the number of ocr queries. The parameter "num_pages" specify how many files from the
previous step should be concatenated to create one single image.
```console
python concat_multiple_images.py --input_files_path ./data/1789to1824_DebatesAndProceedings/one_column_oriented/ --num_pages 4
```

#### 1.4 TEXT DETECTION
For each file uses the Google Cloud Vision API to do an ocr and extract the text. The text
will be saved in a directory called "text_volumes" and the file will have .txt as extension.
```console
python detect_text.py --path_pngs ./1789to1824_DebatesAndProceedings/concat_pages
```

#### 1.4 SPEECH SEGMENTATION
For each .txt file created with the previous step, this script creates a corresponding .csv file with the speaker and 
the list of speeches he made on that .txt file.
```console
python segment_speech.py --input_files_path ./1789to1824_DebatesAndProceedings/text_volumes
```

#### 1.4 BIGRAM ANALYSIS
This script will take the previous created csv files, concatenate them together, merge all the speeches
made by a single speaker and then perform an embedding of these speeches and study the distribution
of these speeches
 
```console
python bigram_analysis.py --input_files_path /home/fla/remote/cnb/Desktop/RA/congress_document_study/data/1789to1824_DebatesAndProceedings/speeches --type volumes
```