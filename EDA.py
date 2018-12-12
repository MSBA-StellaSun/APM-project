# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:35:01 2018

@author: ayuan
"""

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os
from os import listdir
from PIL import Image


root_dir = 'C:/Users/ayuan/OneDrive/Documents/000APM/images/'
rows = 17
cols = 6
fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(15, 25))
fig.suptitle('Random Image from Each Food Class with Descriptions', fontsize=15)
sorted_food_dirs = sorted(os.listdir(root_dir))
for i in range(rows):
    for j in range(cols):
        try:
            food_dir = sorted_food_dirs[i*cols + j]
        except:
            break
        
        path=os.path.join(root_dir, food_dir)
        onlyfiles = [f for f in listdir(path)]
        onlyfiles = [f for f in onlyfiles if f.endswith('.jpg')]
        data = {}
        images_count= len(onlyfiles)
        min_width = 10**100  
        max_width = 0
        min_height = 10**100 
        max_height = 0
        for filename in onlyfiles:
            pic = Image.open(os.path.join(root_dir, food_dir, filename))
            width, height = pic.size
            min_width = min(width, min_width)
            max_width = max(width, max_height)
            min_height = min(height, min_height)
            max_height= max(height, max_height)

        all_files = os.listdir(os.path.join(root_dir, food_dir))
        rand_img = np.random.choice(all_files)
        img = plt.imread(os.path.join(root_dir, food_dir, rand_img))
        ax[i][j].imshow(img)
        ec = (0, .6, .1)
        fc = (0, .7, .2)
        string1=food_dir+','+''.join(['img count:',str(images_count)])
        string2=' '.join(['min width/height:',str(min_width),str(min_height)])
        string3=' '.join(['max width/height:',str(max_width),str(max_height)])
        string='\n'.join([string1,string2])
        ax[i][j].text(10, 10, string, size=8, rotation=0,
                ha="left", va="top", 
                bbox=dict(boxstyle="round",facecolor='white', edgecolor='black'))
plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



