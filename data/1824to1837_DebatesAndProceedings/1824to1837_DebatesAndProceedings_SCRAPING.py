
# coding: utf-8

# In[5]:


## Scraping images for all the items


# In[6]:


import os
import requests
import urllib
from PIL import Image

#Local path where you want the images to be saved. The image will be saved based in the item, and they will be saved as
# "page_0, page_1, ..."
destination_path = "/home/fla/Desktop/Research_Assistantship/Congress_Documents_Scraping/1824to1837_DebatesAndProceedings/Item_"


#Name of the image saved in the webste. Lookup with "Inspection element"
name_of_the_image = "img src=\"/ll/llrd" 

num_items = 29
#The lenghts of each item need to be manually set"
lenghts = [507, 831, 717, 840, 739, 753, 485, 667, 647, 677,
           737, 747, 717, 670, 645, 699, 695, 722, 697, 643,
           651, 643, 643, 643, 599, 723, 705, 596, 641]

appendix_start = [376, 819, 536, 784, -1,  652,  404, -1,  506, 430,
                  714, -1,  516, -1,  318, -1,   -1,  -1,  346,  -1, 
                  206, -1,  670, -1,  438, 708 , 404, 588, 292]

#Used to fill the dictionary automatically
first_part_url="https://memory.loc.gov/cgi-bin/ampage?collId=llrd&fileName="
second_part_url="/llrd"
third_part_url=".db&recNum="


vol = {}
for i in range(1, num_items+1, 1):
    vol[i] = []
    if (i<10):
        vol.get(i).append(first_part_url+"00"+str(i)+second_part_url+"00"+str(i)+third_part_url)
    elif(i<100):
        vol.get(i).append(first_part_url+"0"+str(i)+second_part_url+"0"+str(i)+third_part_url)
    else:
        vol.get(i).append(first_part_url+str(i)+second_part_url+str(i)+third_part_url)
        
    vol.get(i).append(name_of_the_image)
    vol.get(i).append(lenghts[i-1])
    vol.get(i).append(appendix_start[i-1])




# In[7]:


def scrape_item(url, pattern, start, end, destination_path ):
    for count in range(start,end+1, 1):
        new_url = url+str(count)
        f = requests.get(new_url)
        name = extract_name(f.text, (f.text).find(pattern)  )
        complete_path = "https://memory.loc.gov"+str(name)
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


# In[8]:


for item in range(1,len(vol)+1,1):
#for item in range(1,2,1):
    if(vol.get(item)[3]!=-1):
        if not os.path.exists(destination_path+str(item)):
            os.makedirs(destination_path+str(item))
#           scrape_item(url, pattern, start, end, destination_path ):
            scrape_item(vol.get(item)[0], 
                         vol.get(item)[1], 
                         0,
                         vol.get(item)[3], #Start of the appendix
                         destination_path+str(item))
        path=destination_path+str(item)+'/Appendix'
        if not os.path.exists(path):
            os.makedirs(path)
#           scrape_item(url, pattern, start, end, destination_path ):
            scrape_item(vol.get(item)[0], 
                         vol.get(item)[1], 
                         vol.get(item)[3],
                         vol.get(item)[2], 
                         path)
        
        
    else:
        #There is no appendix.
        if not os.path.exists(destination_path+str(item)):
            os.makedirs(destination_path+str(item))
            scrape_item(vol.get(item)[0], 
                         vol.get(item)[1], 
                         0,
                         vol.get(item)[2], 
                         destination_path+str(item))

