#Polarization of Speeches Project.
This project is ideally divided in three parts. First part with speeches from 1789 to 1824, a second part 
from 1824 to 1837, the third from 1833 to 1873. There are five main steps before the final analysis.
1- Dataset Creation (Scraping of the websites)
2- Creation of image format information.
3- Page splitting
3- Text detection
4- Speech segmentation
Since the data directories of these three datasets change slightly, to preserve this information we will 
different scripts for each part. Ideally these scripts are just copies but with small modifications.



#The segment_speeches.py can be used to segment speeches. In order to do this the dataset has to be created.

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
Each of the three data subdirectories contain a "Create_hocrs.py" file, which uses the libray "tesseract" 
to create the .hocr file of each image. This file will be used later to split the image, because it contains
spacial information of each line of the page.
```console
python Create_hocrs.py
```
 
#### 1.3 PAGE SPLITTING
Almost all the images are oriented column-wise. In order to use the Google Cloud Vision API, 
we need to segment these columns and create a single column file. This is done in the "split_pages.py" file.
```console
python split_pages.py
```
Future improvement: This algorithm fails some times if the page is a little be inclined. Could be adjusted.  


#### 1.4 TEXT DETECTION
For each file uses the Google Cloud Vision API to do an ocr and extract the text. The text
will be saved in a directory called "text_volumes" and the file will have .txt as extension.
```console
python detect_text.py
```

#### 1.4 SPEECH SEGMENTATION
For each .txt file created with the previous step, this script creates a corresponding .csv file with the speaker and 
the list of speeches he made on that .txt file.
```console
python segment_speech.py
```

#### 1.4 SPEECH ANALYSIS
This script will take the previous created csv files, concatenate them together, merge all the speeches
made by a single speaker and then perform an embedding of these speeches and study the distribution
of these speeches
 
```console
python analyze_speeches.py
```