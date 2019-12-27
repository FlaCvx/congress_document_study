
# coding: utf-8

# In[3]:


## Scraping images for all the volumes


# In[39]:


import os
import requests
import urllib
from PIL import Image
#Local path where you want the images to be saved. The image will be saved based in the volume, and they will be saved as
# "page_0, page_1, ..."
destination_path = "/home/fla/Desktop/Research_Assistantship/Congress_Documents_Scraping/1833to1873_Debates andProceedings/"


#Name of the image saved in the webste. Lookup with "Inspection element"
name_of_the_image = "img src=\"/ll/llcg" 

#TODO: ADD APPENDIXES FOR EACH SESSION
congress = {
    23: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=001/llcg001.db&recNum=", 0, 493],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=001/llcg001.db&recNum=", 494, 849]
        },
    24: { 1: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=111/llcg111.db&recNum=", 0, 663],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=112/llcg112.db&recNum=", 0, 795]
            ],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=003/llcg003.db&recNum=", 0, 254]
        },
    25: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=004/llcg004.db&recNum=", 0, 159],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=005/llcg005.db&recNum=", 0, 539],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=006/llcg006.db&recNum=", 0, 262]
        },
    26: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=007/llcg007.db&recNum=", 0, 571], 
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=008/llcg008.db&recNum=", 0, 279]
        },
        
    27: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=009/llcg009.db&recNum=", 0, 467],
      2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=010/llcg010.db&recNum=", 0, 1002],
      3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=012/llcg012.db&recNum=", 0, 423],
      4: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=008/llcg008.db&recNum=", 246, 246]
    },
    28: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=013/llcg013.db&recNum=", 0, 719],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=015/llcg015.db&recNum=", 0, 423]
        },
    29: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=016/llcg016.db&recNum=", 0, 1271],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=018/llcg018.db&recNum=", 0, 607]
        },    
    30: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=019/llcg019.db&recNum=", 0, 1139],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=021/llcg021.db&recNum=", 0, 739]
        },
    31: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=022/llcg022.db&recNum=", 0, 1095],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=026/llcg026.db&recNum=", 0, 881]
        },
    32: { 1: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=027/llcg027.db&recNum=", 0, 913],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=028/llcg028.db&recNum=", 0, 865],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=029/llcg029.db&recNum=", 0, 857]
             ],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=031/llcg031.db&recNum=", 0, 1207],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=026/llcg026.db&recNum=", 1279, 1306]
        },
    33: { 1: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=033/llcg033.db&recNum=", 0, 785],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=034/llcg034.db&recNum=", 0, 859],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=035/llcg035.db&recNum=", 0, 789]
             ], 
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=037/llcg037.db&recNum=", 0, 1233],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=032/llcg032.db&recNum=", 254, 341]
        },
    34: { 1: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=039/llcg039.db&recNum=", 0, 802],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=040/llcg040.db&recNum=", 0, 875],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=041/llcg041.db&recNum=", 0, 642]
             ],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=041/llcg041.db&recNum=", 644, 734],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=043/llcg043.db&recNum=", 0, 1155]
        },
    35: { 1: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=045/llcg045.db&recNum=", 0, 1087],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=046/llcg046.db&recNum=", 0, 1025],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=047/llcg047.db&recNum=", 0, 1014]
             ],
          2: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=049/llcg049.db&recNum=", 0, 1083],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=050/llcg050.db&recNum=", 0, 653]   
             ],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=044/llcg044.db&recNum=", 384, 411],
          4: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=047/llcg047.db&recNum=", 1004, 1014],
        },
    36: { 1: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=051/llcg051.db&recNum=", 0, 1021],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=052/llcg052.db&recNum=", 0, 945],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=053/llcg053.db&recNum=", 0, 945],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=054/llcg054.db&recNum=", 0, 473]
             ],
          2: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=055/llcg055.db&recNum=", 0, 993],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=056/llcg056.db&recNum=", 0, 567]
             ],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=050/llcg050.db&recNum=", 646, 653],
          4: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=054/llcg054.db&recNum=", 472, 480]
        },
    37: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=057/llcg057.db&recNum=", 0, 476],
          2: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=058/llcg058.db&recNum=", 0, 1023],
             ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=059/llcg059.db&recNum=", 0, 961],
             ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=060/llcg060.db&recNum=", 0, 79],
             ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=060/llcg060.db&recNum=", 81, 600],
             ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=060/llcg060.db&recNum=", 601, 961],
             ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=061/llcg061.db&recNum=", 0, 528]
             ],
          3: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=062/llcg062.db&recNum=", 0, 939],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=063/llcg063.db&recNum=", 0, 653]
             ],
          4: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=056/llcg056.db&recNum=", 474, 567]
        },
    38: { 1: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=064/llcg064.db&recNum=", 0, 1047],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=065/llcg065.db&recNum=", 0, 977],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=066/llcg066.db&recNum=", 0, 977],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=067/llcg067.db&recNum=", 0, 353],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=067/llcg067.db&recNum=", 354, 620]
             ],
          2: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=068/llcg068.db&recNum=", 0, 787],
		["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=068/llcg068.db&recNum=", 788, 817],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=069/llcg069.db&recNum=", 0, 675]
             ],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=063/llcg063.db&recNum=", 642, 653]
        },
    39: { 1: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=070/llcg070.db&recNum=", 0, 1065],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=071/llcg071.db&recNum=", 0, 961],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=072/llcg072.db&recNum=", 0, 961],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=073/llcg073.db&recNum=", 0, 961],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=074/llcg074.db&recNum=", 0, 471],
              ],
          2: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=075/llcg075.db&recNum=", 0, 765],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=076/llcg076.db&recNum=", 0, 369],
		["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=076/llcg076.db&recNum=", 371, 753],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=077/llcg077.db&recNum=", 0, 578],
              ],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=069/llcg069.db&recNum=", 659, 675]
        },
    40: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=078/llcg078.db&recNum=", 0, 986],
          2: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=079/llcg079.db&recNum=", 364, 1387],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=080/llcg080.db&recNum=", 0, 1027],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=081/llcg081.db&recNum=", 0, 1027],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=082/llcg082.db&recNum=", 0, 1027],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=083/llcg083.db&recNum=", 0, 430],
              ],
          3: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=085/llcg085.db&recNum=", 0, 955],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=086/llcg086.db&recNum=", 0, 755],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=087/llcg087.db&recNum=", 0, 400],
              ],
          4: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=078/llcg078.db&recNum=", 956, 986]
        },
    41: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=088/llcg088.db&recNum=", 0, 902],
          2: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=089/llcg089.db&recNum=", 0,  1315],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=090/llcg090.db&recNum=", 0, 947],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=091/llcg091.db&recNum=", 0, 233],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=091/llcg091.db&recNum=", 234, 581],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=091/llcg091.db&recNum=", 582, 947],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=092/llcg092.db&recNum=", 0, 947],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=093/llcg093.db&recNum=", 0, 947],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=094/llcg094.db&recNum=", 0, 947],
              ],
          3: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=096/llcg096.db&recNum=", 0, 831],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=097/llcg097.db&recNum=", 0, 837],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=098/llcg098.db&recNum=", 0, 587],
              ]
        },
    42: { 1: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=099/llcg099.db&recNum=", 0, 707],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=100/llcg100.db&recNum=", 0, 254],
              ],
          2: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=101/llcg101.db&recNum=", 0, 899],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=102/llcg102.db&recNum=", 0, 899],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=103/llcg103.db&recNum=", 0, 301],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=103/llcg103.db&recNum=", 302, 899],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=104/llcg104.db&recNum=", 0, 279],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=104/llcg104.db&recNum=", 281, 899],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=105/llcg105.db&recNum=", 0, 923],
              ],
          3: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=108/llcg108.db&recNum=", 0, 967],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=109/llcg109.db&recNum=", 0, 963],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=110/llcg110.db&recNum=", 0, 582],
              ],
          4: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=100/llcg100.db&recNum=", 254, 340]
        }
}
num_congresses = len(congress)


appendixes = {
    23: { 1: None,
          2: None
        },
    24: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=112/llcg112.db&recNum=", 10, 795],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=003/llcg003.db&recNum=", 256, 579]
        },
    25: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=004/llcg004.db&recNum=", 159, 502],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=005/llcg005.db&recNum=", 0, 1180],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=006/llcg006.db&recNum=", 264, 685]
        },
    26: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=007/llcg007.db&recNum=", 572, 1434],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=008/llcg008.db&recNum=", 280, 679]
        },    
    27: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=009/llcg009.db&recNum=", 468, 970],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=011/llcg011.db&recNum=", 0, 996],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=012/llcg012.db&recNum=", 424, 683],
          4: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=008/llcg008.db&recNum=", 246,  247]
        },
    28: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=014/llcg014.db&recNum=", 0, 793],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=015/llcg015.db&recNum=", 424, 844]
        },
    
    29: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=017/llcg017.db&recNum=", 0, 1191],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=018/llcg018.db&recNum=", 608, 1059]
        },

    30: { 1: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=020/llcg020.db&recNum=", 10, 1222],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=026/llcg026.db&recNum=", 1307, 1327]
             ],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=021/llcg021.db&recNum=", 744, 1098]
        },
    31: { 1: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=024/llcg024.db&recNum=", 0, 867],
              ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=025/llcg025.db&recNum=", 0, 869]             ],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=026/llcg026.db&recNum=", 882, 1278]
        },
    32: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=030/llcg030.db&recNum=", 0, 1203],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=032/llcg032.db&recNum=", 0, 398],
          3: None
        },
    33: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=036/llcg036.db&recNum=", 0, 1239],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=038/llcg038.db&recNum=", 0, 453],
          3: None
        },
    34: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=042/llcg042.db&recNum=", 0, 1319],
          2: None,
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=044/llcg044.db&recNum=", 0, 470]
        },
    35: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=048/llcg048.db&recNum=", 0, 1087],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=049/llcg049.db&recNum=", 0, 1083],
          3: None,
          4: None
        },
    36: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=054/llcg054.db&recNum=", 500, 1033],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=056/llcg056.db&recNum=", 586, 953],
          3:  None,
          4:  None
        },
    37: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=057/llcg057.db&recNum=", 494, 545],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=061/llcg061.db&recNum=", 562, 988],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=063/llcg063.db&recNum=", 680, 922],
          4:  None
        },
    38: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=067/llcg067.db&recNum=", 654, 926],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=069/llcg069.db&recNum=", 694, 857],
          3:  None
        },
    39: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=074/llcg074.db&recNum=", 522, 965],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=077/llcg077.db&recNum=", 620, 878],
          3:  None
        },
    40: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=078/llcg078.db&recNum=", 1000, 1050],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=083/llcg083.db&recNum=", 480, 1081],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=087/llcg087.db&recNum=", 428, 767],
          4:  None
        },
    41: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=088/llcg088.db&recNum=", 918, 973],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=095/llcg095.db&recNum=", 74, 844],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=098/llcg098.db&recNum=", 624, 1037],
        },
    42: { 1: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=100/llcg100.db&recNum=", 0, 354],
          2: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=106/llcg106.db&recNum=", 56, 899],
          3: ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=110/llcg110.db&recNum=", 632, 582],
          4: [["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=100/llcg100.db&recNum=", 254, 690],
	       ["https://memory.loc.gov/cgi-bin/ampage?collId=llcg&fileName=100/llcg100.db&recNum=", 691, 693]
	      ]
        }
}


# In[41]:


def scrape_congress(url, pattern, num_min, num_max, destination_path ):
    #print('url: '+url)
    #print('pattern: '+pattern)
    #print('num_min: '+str(num_min))
    #print('num_max: '+str(num_max))
    #print('destination_path: '+str(destination_path))
    
    for count in range(num_min,num_max+1, 1):
        new_url = url+str(count)
        f = requests.get(new_url)
        #print(new_url)
        #print(f.text)
        #print((f.text).find(pattern))
        
        name = extract_name(f.text, (f.text).find(pattern)  )
        complete_path = "https://memory.loc.gov"+str(name)
        #print(destination_path+"/page_"+str(count))
        urllib.request.urlretrieve(complete_path, destination_path+"/page_"+str(count))
        
    for count in range(num_min,num_max+1, 1):
        im = Image.open(destination_path+"/page_"+str(count))
        im.save( destination_path+"/page_"+str(count),'PNG')   
        
def extract_name(raw_line, start_index):
    end = start_index
    #print(raw_line)
    while (raw_line[end:end+3]!="gif"):
        end += 1
    end +=3
    return raw_line[start_index+9:end]


# In[44]:


#for i in range(31,32,1):
for i in range(36,43,1):
	for session in range(1,len(congress[i])+1,1):
		path = destination_path+'Congress_'+str(i)+'/Session_'+str(session)
		if(congress[i].get(session)[0][0]!='h'):
			for j in range(0,len(congress[i].get(session))):
				path = destination_path+'Congress_'+str(i)+'/Session_'+str(session)+'/Part_'+str(j)
				if not os.path.exists(path):
					os.makedirs(path)
					print('Scraping: Congress_'+str(i)+'/Session_'+str(session)+'/Part_'+str(j))
					scrape_congress(congress[i].get(session)[j][0],
						    name_of_the_image,
						    congress[i].get(session)[j][1],
						    congress[i].get(session)[j][2],
						    str(path)
						   )
		else:
			if not os.path.exists(path):
				os.makedirs(path)
				print('Scraping: Congress_'+str(i)+'/Session_'+str(session))
				scrape_congress(congress[i].get(session)[0],
					name_of_the_image,
					congress[i].get(session)[1],
					congress[i].get(session)[2],
					str(path)
				       )
		if(appendixes[i].get(session)!=None):

			if(appendixes[i].get(session)[0][0]=='h'):
				path= destination_path+'Congress_'+str(i)+'/Session_'+str(session)+'/Appendix'
				if not os.path.exists(path):
					os.makedirs(path)
					print('Scraping: Congress_'+str(i)+'/Session_'+str(session)+'/Appendix')
					scrape_congress(appendixes[i].get(session)[0],
						name_of_the_image,
						appendixes[i].get(session)[1],
						appendixes[i].get(session)[2],
						str(path)
					       )
			else:
				for j in range(0,len(appendixes[i].get(session)[0])-1):
					path= destination_path+'Congress_'+str(i)+'/Session_'+str(session)+'/Appendix'
					path += '/Part_'+str(j)
					if not os.path.exists(path):
						os.makedirs(path)
						print('Scraping: Congress_'+str(i)+'/Session_'+str(session)+'/Appendix')
						scrape_congress(appendixes[i].get(session)[j][0],
							name_of_the_image,
							appendixes[i].get(session)[j][1],
							appendixes[i].get(session)[j][2],
							str(path)
						       )


# for count in range(10,42+1, 1):
#         im = Image.open('/home/fla/Desktop/Research_Assistantship/Congress_Documents_Scraping/1833to1873_Debates andProceedings/Congress_30/Session_1/Appendix/Part_0/'+"page_"+str(count))
#         im.save( '/home/fla/Desktop/Research_Assistantship/Congress_Documents_Scraping/1833to1873_Debates andProceedings/Congress_30/Session_1/Appendix/Part_0/'+"page_"+str(count),'PNG')   
# 
