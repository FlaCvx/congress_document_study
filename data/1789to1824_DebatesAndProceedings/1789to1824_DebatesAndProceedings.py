# coding: utf-8

# In[11]:


#FOR EXTERNAL USE: CHANGE THE "destination_path"


# In[12]:


## Scraping images for all the volumes


# In[13]:


import os
import requests
import urllib
from PIL import Image

#Local path where you want the images to be saved. The image will be saved based in the volume, and they will be saved as
# "page_0, page_1, ..."
destination_path = "/home/fla/Desktop/Research_Assistantship/Congress_Documents_Scraping/1789to1824_DebatesAndProceedings/Volume_"


#Name of the image saved in the webste. Lookup with "Inspection element"
name_of_the_image = "img src=\"/ll/llac" 

#The lenghts of each volume need to be manually set"
#FILLED manually, unfortunately the lengths change.
lengths = [638, 570, 723, 764, 754, 725, 603, 600, 778, 785,
           686, 801, 656, 847, 644, 645, 715, 718, 921, 635,
           654, 675, 588, 591, 676, 715, 709, 981, 956, 669,
           652, 646, 605, 669, 637, 675, 911, 620, 698, 705,
           844, 792]


appendix_start = [ -1,  378, 480, 640, -1,  427, -1,  -1,  322, 539,
                   347, -1,  -1,  611, -1,  340, -1,  424, -1,  -1, 
                   389, 571, -1,  266, 582, -1,  312, 638, 731, 532, 
                   -1,  237, -1,  116, -1,  488, 653, -1,  326, 587, 
                   -1, 612 ]


num_volumes = len(lengths)

#Used to fill the dictionary automatically
first_part_url="https://memory.loc.gov/cgi-bin/ampage?collId=llac&fileName="
second_part_url="/llac"
third_part_url=".db&recNum="


#The following for loop creates the incremental names for the links to scrape, in this case we were
# lucky because the names have incremental numbers in a link's part.

vol = {}
for i in range(1, num_volumes+1, 1):
    vol[i] = []
    if (i<10):
        vol.get(i).append(first_part_url+"00"+str(i)+second_part_url+"00"+str(i)+third_part_url)
    elif(i<100):
        vol.get(i).append(first_part_url+"0"+str(i)+second_part_url+"0"+str(i)+third_part_url)
    else:
        vol.get(i).append(first_part_url+str(i)+second_part_url+str(i)+third_part_url)
        
    vol.get(i).append(name_of_the_image)
    vol.get(i).append(lengths[i-1])
    vol.get(i).append(appendix_start[i-1])




# In[14]:


def scrap_volume(url, pattern, start, end, destination_path ):
    for count in range(start,end+1, 1):
        new_url = url+str(count)
        f = requests.get(new_url)
        name = extract_name(f.text, (f.text).find(pattern)  )
        complete_path = "https://memory.loc.gov"+str(name) # Found out by inspecting the web page.
        urllib.request.urlretrieve(complete_path, destination_path+"/page_"+str(count))
    for count in range(start,end+1, 1):
        im = Image.open(destination_path+"/page_"+str(count))
        im.save( destination_path+"/page_"+str(count),'PNG')
        
def extract_name(raw_line, start_index):
    end = start_index
    while (raw_line[end:end+3]!="gif"):
        end += 1
    end +=3
    return raw_line[start_index+9:end]


# In[15]:


for volumes in range(1,num_volumes+1,1):
#for volumes in range(1,3,1):
    if(vol.get(volumes)[3]!=-1):
        if not os.path.exists(destination_path+str(volumes)):
            os.makedirs(destination_path+str(volumes))
#           scrap_volume(url, pattern, start, end, destination_path ):
            scrap_volume(vol.get(volumes)[0], 
                         vol.get(volumes)[1], 
                         0,
                         vol.get(volumes)[3], #Start of the appendix
                         destination_path+str(volumes))
        path=destination_path+str(volumes)+'/Appendix'
        if not os.path.exists(path):
            os.makedirs(path)
#           scrap_volume(url, pattern, start, end, destination_path ):
            scrap_volume(vol.get(volumes)[0], 
                         vol.get(volumes)[1], 
                         vol.get(volumes)[3],
                         vol.get(volumes)[2], 
                         path)
        
        
    else:
        #There is no appendix.
        if not os.path.exists(destination_path+str(volumes)):
            os.makedirs(destination_path+str(volumes))
            scrap_volume(vol.get(volumes)[0], 
                         vol.get(volumes)[1], 
                         0,
                         vol.get(volumes)[2], 
                         destination_path+str(volumes))

