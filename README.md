# Polarization of Speeches Project.
The project aims at analyzing the polarization of the speeches of congressmen between 1789 and 1873.
In order to do this, we need to first build the dataset. We do this by scraping the web to obtain the 
scans of the congresses' documents.
Once we have all the scans, we need to perform OCR on these documents. In order to
do this, we use the Google Cloud Vision API. Unfortunately this API is not able to
recognize if a page is organized in multiple columns, which is our case. Therefore, 
before feeding the pages to the API we split and recompose the page into one-column-oriented
pages. Then we feed this pages to the API in order to obtain text data.
The text data is finally cleaned and it will be used to create a new dataset on which we will perform 
our analysis.
This new dataset will associate all the speeches to their appropriate congressman.
Since we know the party of each congressman, we will then run ML on this new dataset to
see whether we can predict the party of a congressman solely based on his speech.

This project has three datasets. First dataset with speeches from 1789 to 1824, 
a second datasets 
from 1824 to 1837 and the third one from 1833 to 1873. 
There are five main steps before the final analysis.
1- Dataset Creation (Scraping of the websites)
2- Creation of image format information.
3- Page splitting
3- Text detection
4- Speech segmentation
5- Bigram analysis


#### 1.1 DATASET CREATION
The dataset can be found in the ./data directory, it is divided into three subdirectories.<br />
1- "./data/1789to1824_DebatesAndProceedings" scraped from the link: "https://memory.loc.gov/ammem/amlaw/lwaclink.html"<br />
2- "./data/1824to1837_DebatesAndProceedings" scraped from the link: "https://memory.loc.gov/ammem/amlaw/lwrdlink.html"<br />
3- "./data/1833to1873_Debates andProceedings". scraped from the link: "https://memory.loc.gov/ammem/amlaw/lwcglink.html"<br />

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
Future improvement: This algorithm fails some times if the page is inclined. Could be improved.  


#### 1.4 CONCAT MULTIPLE IMAGES
From now, all the next steps MUST be run once all the previous steps are completed. 
Otherwise the 
 will concatenate images and may skip others that are not yet available.. 

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

#### 1.5 SPEECH SEGMENTATION
For each .txt file created with the previous step, this script creates a corresponding .csv file with the speaker and 
the list of speeches he made on that .txt file.
```console
python segment_speech.py --input_files_path ./1789to1824_DebatesAndProceedings/text_volumes
```

#### 1.6 BIGRAM ANALYSIS
This script will take the previous created csv files, concatenate them together, merge all the speeches
made by a single speaker and then perform a bygram analysis of these speeches.
 
```console
python bigram_analysis.py --input_files_path /home/fla/remote/cnb/Desktop/RA/congress_document_study/data/1789to1824_DebatesAndProceedings/speeches --type volumes
```