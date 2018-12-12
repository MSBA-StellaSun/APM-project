
![Food img](https://www.ocf.berkeley.edu/~sather/wp-content/uploads/2018/01/food--1200x600.jpg "Food image")
# Food Recognition with Nutrition Facts
## Team members
    Mengying Yu  
    Shuyuan Sun (ss86757)  
    Wanxue Dong  
    Wenying Hu  (wh7893)
    Xueru Rong  
## Content
- [Abstract](#1-abstract)
- [Introduction and Background](#2-introduction-and-background)
  - Project Goal and Why we care
  - Related work
  - Outline of approach and rationale (high level)
  - Contributions or novel characteristics of project
- [Data Collection and Description](#3-data-collection-and-description)
  - Source
  - Methods of acquisition
  - Relevant Characteristics
- [Data Exploration and Preprocessing](#4-data-exploration-and-preprocessing)
  - Feature engineering/selection
  - Relevant Plots
- Learning/Modeling
  - Chosen models and why
  - Training methods (validation, parameter selection)
  - Other design choices
- Results
  - Key findings and evaluation
  - Comparisons from different approaches
  - Plots and figures
- Conclusion
  - Summarize everything above
  - Lessons learned
  - Future work - continuations or improvements
- References
- Relevant project links (i.e. Github, Bitbucket, etc…)

## 1. Abstract
(say something here)

## 2. Introduction and Background
### Project Goal and Why We Care

Sometimes delicious food can be satisfactory, while it could become a barrier to establish healthy dietary habits.          Traditional way to know about the nutrition from the food is to search the name from a database. The main goal of our project, however, aims to design a more convenient way to let people understand the nutrition fact from what they eat. Imagine how convenient when you just take a picture from your phone, and it will recognize the food and return the nutrition fact from the picture. 

### Related work

There are several papers that use the same dataset as ours—food 101. The following papers and blogs were able to achieve certain accuracy based on CNN or Random Forest. 
  
Inception-ResNet (full layer training) 72.55% Accuracy:
  
https://pdfs.semanticscholar.org/6dbb/4f5a00f81971b7bc45f670f3706071a9db20.pdf
  
Random Forest 50.76% Accuracy:
  
https://link.springer.com/chapter/10.1007/978-3-319-10599-4_29

### Outline of approach and rationale (high level)
    - Data preprocessing 
    - Train AlexNet
    - Model modifications
    - Obtain FatSecret API
    - Model Evaluations
  
### Contributions or novel characteristics of project

One of the novelties of our project is that recognized food picture is associated with a food ID from the FatSecret database. We then wrote a python code linking this food ID to the database in the FatSecret in order to get the nutrition fact from the recognized picture.  

## 3. Data Collection and Description

### Source

  We obtained our data from the original paper using random forest (2014) [linkhere](https://www.vision.ee.ethz.ch/datasets_extra/food-101/). 
  
```
@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}
```
  We use this dataset because it can be downloaded directly. Also, this dataset is quite large for us to lower the variance and train a better model.

### Methods of acquisition

  Since the dataset is publicly available from the website [downloadhere](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz), we simply downloaded the dataset and then unzipped it into our laptops.
  The unzipped dataset contains two folders:  
```
images
meta
```
  Images can be found in the "images" folder and are organized per class.
  
  <p align="center">
  <img src="https://github.com/MSBA-StellaSun/APM-project/blob/master/Data/categories.png">
  </p>
  
  For example, in the folder "apple_pie", we can see images of apple pie:
  
  <p align="center">
  <img src="https://github.com/MSBA-StellaSun/APM-project/blob/master/Data/sampleimg.png">
  </p>
  
  All image IDs are unique and correspond to the foodspotting.com review IDs. Thus the original articles can retrieved trough [foodspotting](http://www.foodspotting.com/reviews) or through the [foodspotting api](http://www.foodspotting.com/api).  
  The class labels and test/train splitting used in the experiment can be found in the "meta" directory:
  
  <p align="center">
  <img src="https://github.com/MSBA-StellaSun/APM-project/blob/master/Data/metafile.png">
  </p>
  
  The class labels of the images under different categories are like this:
  
  <p align="center">
  <img src="https://github.com/MSBA-StellaSun/APM-project/blob/master/Data/classlabel.png">
  </p>
  
  The train set IDs are like this:
  
  <p align="center">
  <img src="https://github.com/MSBA-StellaSun/APM-project/blob/master/Data/trainlabel.png">
  </p>
  
  And the test set IDs are the rest of those IDs:
  
  <p align="center">
  <img src="https://github.com/MSBA-StellaSun/APM-project/blob/master/Data/testlabel.png">
  </p>
  
  There are also two txt files providing some instructions on the use of this dataset:
```
license_agreement.txt
README.txt
```

### Relevant Characteristics

  There are 101 food categories in total in our dataset, each with 1000 images. 750 of them in each category are training images and the rest 250 are testing images. 
  All images were originally rescaled to have a maximum side length of 512 pixels.
  This dataset is also said to have some amount of noises on purpose to reflect real-world situations. For example, there is an image of iPad under the food category "apple_pie". So the noises may add some difficulties for us to train the model.
  
## 4. Data Exploration and Preprocessing

### Relevant Plots

  We explored and processed our data using Spyder(Python 3.6).
  First we import relevant packages needed:
```Python
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os
from os import listdir
from PIL import Image
```
  Then for all 101 categories, we randomly show an image, stating the name of that food category, the number of images and the minimal side length of images in that category.
```Python
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
```
  <p align="center">
  <img src="https://github.com/MSBA-StellaSun/APM-project/blob/master/Data/EDA.png">
  </p>
  The [EDA](/example/profile.md)	
### Feature engineering/selection

  First we import the packages needed:
```Python
import numpy as np
import cv2
import gc
import random
```
  Then the train/test metadata are loaded using this function:
```Python
# return the data in the file.
def read_file(fn):
    data = []
    with open(fn, "r") as fd:
        for line in fd:
            data.append(line.strip())
    return data
```
  From the exploration we can see that the side length of all the images varies from 193 to 512 pixels. So we need to first pad each image with an edge to a fixed side length of 512 pixels, 
  <p align="center">
  <img src="https://github.com/MSBA-StellaSun/APM-project/blob/master/Data/padding.png">
  </p>
  and then resize all images to 128*128.
  <p align="center">
  <img src="https://github.com/MSBA-StellaSun/APM-project/blob/master/Data/resize.png">
  </p>
  
```Python
# return the image padded to (512, 512, 3) and downsize to (128, 128, 3).
W = 128
def read_image(fn):
    image = cv2.imread(fn,cv2.IMREAD_COLOR)
    x, y, _ = image.shape
    padded_image = np.pad(image, ((0, 512 - x), (0, 512 - y), (0, 0)), 'edge')
    return cv2.resize(padded_image,(W, W))
```
  Then we write a function to map all the labels to distinct index values:
```Python
# prepare a map of labels to their result index.
def prepare_labels(dataset):
    labels = set()
    for i in dataset:
        labels.add(i.split("/")[0])
    
    labels_map = {}
    for p, l in enumerate(sorted(list(labels))):
        labels_map[l] = p
    return labels_map
```
  And then we store all the image data and all the corresponding labels into numpy arrays, with corresponding category in the list turned into 1 while others remain 0.
  <p align="center">
  <img src="https://github.com/MSBA-StellaSun/APM-project/blob/master/Data/array.png">
  </p>
  
```Python
# prepare the data into numpy array format.
def prepare_data(dataset, labels_map):
    images = np.zeros((len(dataset), W, W, 3), dtype=np.uint8)
    labels = np.zeros((len(dataset), 101))
    for p, i in enumerate(dataset):
        if p % 5000 == 0:
            gc.collect()
            print("Processed {} images.".format(p))
        images[p] = read_image("images/" + i + ".jpg")
        labels[p, labels_map[i.split("/")[0]]] = 1
    return images, labels
```
  Then we write the main function to link all the defined functions above for us to prepare the train/test data into npy files. Besides, we also randomly shuffle the images at first, otherwise the images would be by categories in alphabetical order, which will adversly influence our model fitting.
```Python
# preprocess all data into npy format.
def process_data(dataset_name):
    print("\nProcess {} set:".format(dataset_name))
    dataset = read_file("meta/{}.txt".format(dataset_name))
    random.Random(10086).shuffle(dataset)
    labels_map = prepare_labels(dataset)
    x, y = prepare_data(dataset, labels_map)
    np.save("data/{}_x.npy".format(dataset_name), x)
    np.save("data/{}_y.npy".format(dataset_name), y)
def main():
    process_data("train")
    process_data("test")
```
  After excuting the main function:
```Python
if __name__ == "__main__":
    main()
```
  We get our feature matrix npy files for train and test set.
  <p align="center">
  <img src="https://github.com/MSBA-StellaSun/APM-project/blob/master/Data/xnpy.png">
  </p>
  And also our label npy files for train and test set.
  <p align="center">
  <img src="https://github.com/MSBA-StellaSun/APM-project/blob/master/Data/ynpy.png">
  </p>
  
## 5. Learning/Modeling
### Chosen models and why
### Training methods (validation, parameter selection)
### Other design choices

## 6. Results
### Key findings and evaluation
### Comparisons from different approaches
### Plots and figures

## 7. Conclusion
### Summarize everything above

We aggregated image data from Food 101 dataset and made image pre-processing to convert the images into proper size. After splitting train and test sets, we conducted AlexNet models to perform image recognition. The accuracy of the AlexNet model is 52.6%, which is better than the result of original paper, 50.67%. After predicting the food class from image recognition, we got the nutrition and calorie facts of the food from FatScret Platform API and showed the information to people.

### Lessons learned



### Future work - continuations or improvements

Ensemble learning algorithm can be used for future work. Ensemble method combines multiple models to obtain better predictive performance. 
Other architectures of convolutional networks, such as ResNet and VGG, can be applied in the future. ResNet makes use of Residual module and make it easier for network layers to represent the identity mapping. So ResNet have more layers and is able to go deeper but takes much more time. Compared to AlexNet, VGG uses multiple stacked smaller size kernel. These non-linear layers help increase the depth of the network, which enables the VGG to learn more complex features with a lower cost. Thus, VGG performs well on image feature extraction.

## 8. References

[1] Pouladzadeh, P., Kuhad, P., Peddi, S. V. B., Yassine, A., & Shirmohammadi, S. (2016, May). Food calorie measurement using deep learning neural network. In IEEE International Instrumentation and Measurement Technology Conference Proceedings (I2MTC) (pp. 1-6).

[2] Bossard, L., Guillaumin, M., & Van Gool, L. (2014, September). Food-101–mining discriminative components with random forests. In European Conference on Computer Vision (pp. 446-461). Springer, Cham. Retrieved from https://www.vision.ee.ethz.ch/datasets_extra/food-101/ 

[3] Simayijiang, Z., & Grimm, S. Segmentation with Graph Cuts. Matematikcentrum Lunds Universitet.[Online]. Available: http://www.maths.lth.se/matematiklth/personal/petter/rapporter/graph. pdf.[Diakses 8 Mei 2017].  
(should have more references here)


  
(Below are from his blog requirement)  
Relevant project links (i.e. Github, Bitbucket, etc…)

    General Tips:
    In general, imagine your audience has a basic understanding of machine learning concepts, but is likely far from an expert. It should be an easier read than most academic research papers.
    Key code snippets can be helpful for people trying to understand how you implemented your project.
    Be sure to include images, gifs, or short videos. These will help liven up your post, make it more attractive to readers, and give you an opportunity to flex your data visualization skills. 
    Plenty of good examples can be found on Medium on TowardsDataScience.
    Make sure your writing is fluid and grammatically correct. 

    Evaluation Criteria include: 
    Clear description of project goals (and business relevance if applicable)
    Approach to pre-processing of data and feature extraction, the choice of data mining models used (and rationale for these choices)
    New theory/math (if applicable, most projects won’t have this component)
    Intelligent selection and tuning of models, addressing overfitting vs underfitting
    Novelty of approach/method/algorithm (if applicable)
    Presentation and evaluation of results
    Replicability of the results (is the description such that someone well versed in the art can obtain similar results on the same data?)
    Insights obtained from the effort
    Potential business impact or how the results can be “actioned upon” (if applicable)
    Appropriate/relevant reference list
    Quality of the writing (grammar and style)
    Visuals to aid in understanding of project elements 

Examples:  
https://towardsdatascience.com/automatic-speaker-recognition-using-transfer-learning-6fab63e34e74
https://towardsdatascience.com/predict-the-number-of-likes-on-instagram-a7ec5c020203
https://towardsdatascience.com/youtube-views-predictor-9ec573090acb
https://towardsdatascience.com/a-data-science-for-good-machine-learning-project-walk-through-in-python-part-one-1977dd701dbc
https://towardsdatascience.com/predicting-school-performance-with-census-income-data-ad3d8792ac97
