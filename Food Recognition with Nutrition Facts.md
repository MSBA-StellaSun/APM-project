
![Food img](https://www.ocf.berkeley.edu/~sather/wp-content/uploads/2018/01/food--1200x600.jpg "Food image")
# Food Recognition with Nutrition Facts
## Team members
    Mengying Yu  
    Shuyuan Sun 
    Wanxue Dong  
    Wenying Hu
    Xueru Rong  
## Content
- [Abstract](#1-abstract)
- [Introduction and Background](#2-introduction-and-background)
  - Project Goal and Why we care
  - Related work
  - Outline of approach and rationale
  - Contributions or novel characteristics of project
- [Data Collection and Description](#3-data-collection-and-description)
  - Source
  - Methods of acquisition
  - Relevant Characteristics
- [Data Exploration and Preprocessing](#4-data-exploration-and-preprocessing)
  - Feature engineering/selection
  - Relevant Plots
- [Learning and Modeling](#5-learning-and-modeling)
  - Chosen models and why
  - Training methods
  - Other design choices
- [Results](#6-Results)
  - Model Performance
- [Application](#7-Application)
  - FatSecret Platform API
- [Conclusion](#8-Conclusion)
  - Summarize everything above
  - Future work - continuations or improvements
- [References](#9-References)
  - Relevant project links

## 1. Abstract
Food provides our bodies with the energy, protein, essential fats, vitamins and minerals to live, grow and function properly. People use different methods to determine how many nutrition they need and control the amount by daily intake. An accurate and convenient solution to food nutrition measurement is the key to long-term health plan. In this project, we propose an assistive food nutrition measurement model which could recognize food and provide nutrition information automatically. In order to identify the food accurately, we use deep convolutional neural networks to classify 101,000 high-quality food images for model training. The proposed methodology and measurements of the proposed model are also described below.

## 2. Introduction and Background
### Project Goal and Why We Care

Sometimes delicious food can be satisfactory, while it could become a barrier to establish healthy dietary habits.          Traditional way to know about the nutrition from the food is to search the name from a database. The main goal of our project, however, aims to design a more convenient way to let people understand the nutrition fact from what they eat. Imagine how convenient when you just take a picture from your phone, and it will recognize the food and return the nutrition fact from the picture. 

### Related work

There are several papers that use the same dataset as ours—food 101. The following papers and blogs were able to achieve certain accuracy based on CNN or Random Forest. 
  
[Inception-ResNet (full layer training) 72.55% Accuracy](https://pdfs.semanticscholar.org/6dbb/4f5a00f81971b7bc45f670f3706071a9db20.pdf)
  
[Random Forest 50.76% Accuracy](https://link.springer.com/chapter/10.1007/978-3-319-10599-4_29)

### Outline of approach and rationale
    - Data preprocessing 
    - Train AlexNet
    - Model modifications
    - Obtain FatSecret Platform API
    - Model Evaluations
  
### Contributions or novel characteristics of project

One of the novelties of our project is that recognized food picture is associated with a food ID from the FatSecret database. We then wrote a python code linking this food ID to the database in the FatSecret in order to get the nutrition fact from the recognized picture.  

## 3. Data Collection and Description

### Source

  We obtained our data from the [original paper](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) using random forest (2014). 
  
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

  Since the dataset is publicly available from the [website](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz), we simply downloaded the dataset and then unzipped it into our laptops.
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
  
  The data exploration python code is uploaded [here](https://github.com/MSBA-StellaSun/APM-project/blob/master/EDA.py).
  
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
  Then we write the main function to link all the defined functions above for us to prepare the train/test data into npy files. Besides, we also randomly shuffle the images at first, otherwise the images would be by categories in alphabetical order, which will adversely influence our model fitting.
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
  
  The data preprocessing python code is uploaded [here](https://github.com/MSBA-StellaSun/APM-project/blob/master/preprocessing.py).
  
## 5. Learning and Modeling
### Chosen models and why
  Since one of our teammates has a NVIDIA 1070TI GPU, we decide to perform all the CNN model training on her computer. Therefore, finding the balance of reasonable running time and model accuracy is the key to success in our project. We start with two pre-trained CNN models called AlexNet and ResNet, which are suitable for large color-images. AlexNet, the winner of Imagenet ILSVRC 2012, has a good performance with only right layers; ResNet launched in 2016, has very deep networks therefore hard to train. As result, AlexNet takes approximately 30 epochs to converge but ResNet needs more than 100. The running time of ResNet is about three times of AlexNet but the initial accuracies are both close to 48%. Therefore, for the scope of time in this project, we decide to focus more on AlexNet.
### Training methods 
  Like majority of others which are also doing image recognition project, we use TensorFlow as the backend. Another handy package we applied is TfLearn, which is a higher-level API for TensorFlow. If someone only wants to solve a high-level vision problem with neural network, using TfLearn can significantly reduce lines of code in the model. The basic model setting is a 3x3 kernel in all convolutional layers. For the max pooling we use a pool size of 2x2. We use relu and tanh activation functions between each layer and a softmax activation function for the last layer. Since it is a classification problem, our loss function is categorical cross-entropy. Besides of pre-trained models and hyper-parameters, we also perform a lot of modifications. We add data augmentation to increase value to base set. Image blur, four-direction flip, different angles rotation are all applied in our model. 
```python
    image_aug = ImageAugmentation()
    image_aug.add_random_blur(1)
    image_aug.add_random_flip_leftright()
    image_aug.add_random_flip_updown()
    image_aug.add_random_rotation()
    image_aug.add_random_90degrees_rotation()
```
  We also add more dropouts into conventional layers to prevent overfitting. Instead of local response normalization, we use batch normalization to improve performance and stability. We replace the optimizer from SGD to an advanced stochastic optimizer called Adam. In addition, we tune other numerical parameters like increasing the number of epochs to 100 to improve the accuracy. 
  The code for modeling can be found [here](https://github.com/MSBA-StellaSun/APM-project/blob/master/model.py). 

## 6. Results
### Model Performance
After running 60 epochs, our model begins to converge. We run 100 epochs to make the result more accurate and evaluate once every 5 epochs. The final accuracy of AlexNet model is 52.63%. This model outperforms the random forest model from original paper, which achieving highest accuracy of 50.76%.

## 7. Application
### FatSecret Platform API

Once we are able to predict the food class, we access a food and nutrition database,  FatSecret Platform API, which will return food’s nutrition fact information.  

FatSecret Platform API is “the #1 food and nutrition database in the world, utilized by more than 10,000 developers, in more than 50 countries contributing in excess of 500 million API calls every month”  provided by https://platform.fatsecret.com/api/. 

An "out of the box" FatSecret Platform API application incorporates the following features:
-	a summary of food, exercise and weight activity
-	integrated food and nutrition search and detailed results
-	a food diary - for planning and tracking foods eaten
-	an activity diary - to estimate and record energy utilization for various exercises
-	a weight chart and journal - to set goals and track progress

We applied a free but limited access to the databse. API Access Key: 8575eae8dc11485090730817b5c67c94

For simplicity and demo purpose, we wrote two python programs.  
  
Python Code could be found from following links:  
demo.py: https://github.com/MSBA-StellaSun/APM-project/blob/master/demo.py.  
PyWriteHtml.py: https://github.com/MSBA-StellaSun/APM-project/blob/master/PyWriteHtml.py.  

From command line: (image.jpg could be replaced with any given path of a image file)  
python demo.py image.jpg

(demo.py will call functions build in PyWriteHtml.py)

These two programs would do following work:
-	Read in trained and saved model 
- 	Read in labels
-	Read in a provided image and make the classification
- 	Print out top 5 class names
-	Use API Access key to connect to the database, call foods_search function from fatsecret library, pass the predicted class name who has highest probability as the parameter
-	Genarate the following html code, e.g., chicken drumstck is the predicted food and food id is "1679" in the database, and prompt out a window to user which includes food's nuitrition information.

```html
<!DOCTYPE html >
<html>
	<head>
		<title>Sample Code</title>
		<style>
			body
			{
				font-family: Arial;
				font-size: 12px;
			}
			.title{
				font-size: 200%;
				font-weight:bold;
				margin-bottom:20px;
			}
			.holder{
				width:300px;
				margin:0 auto;
				padding: 10px;
			}
		</style>
		<script src="http://platform.fatsecret.com/js?key=8575eae8dc11485090730817b5c67c94&amp;auto_template=false&amp;theme=none"></script>
		<script>
			function doLoad(){
				fatsecret.setContainer('container');
				fatsecret.setCanvas("food.get", {food_id: 1679});
			}
		</script>
	</head>
	<body onload="doLoad()">
		<div class="holder">
			<div class="title"><script>fatsecret.writeHolder("foodtitle");</script></div>
			<script>fatsecret.writeHolder("nutritionpanel");</script>
			<div id="container"></div>
		</div>
	</body>
</html>
```
Users are able to see the following picture:

  <p align="center">
  <img src="https://github.com/MSBA-StellaSun/APM-project/blob/master/Data/nutritionSample.png">
  </p>

More features could be realized by app developers, discussed in session 8 "future work".

To watch a short demo, please click [here](https://youtu.be/sPo7R0-x4xI)
## 8. Conclusion
### Summarize everything above

We aggregated image data from Food 101 dataset and made image pre-processing to convert the images into proper size. After splitting train and test sets, we conducted AlexNet models to perform image recognition. The accuracy of the AlexNet model is 52.6%, which is better than the result of original paper, 50.67%. After predicting the food class from image recognition, we got the nutrition and calorie facts of the food from FatScret Platform API and showed the information to people.

### Google Could Platform 
As mentioned above, we used NVIDIA 1070TI GPU to run the model. In order to go "deeper", however, we need to obtain a faster environment. Google Cloud Platform, in this case, is a useful tool to proceed further studies and designs of our model. Some tips for using GCP: 
 * Google SDK is a useful shell to upload your dataset from your local computer to the instances used to run the model.
 * Some errors shows, for example, "You've reached your limit of 0 GPUs NVIDIA K80". You need to edit your quotas to increase the GPU limit and wait for it to be approved.
 * Don't forget to stop your instances!

### Future work - continuations or improvements

Ensemble learning algorithm can be used for future work. Ensemble method combines multiple models to obtain better predictive performance. 

Other architectures of convolutional networks, such as ResNet and VGG, can be applied in the future. ResNet makes use of Residual module and make it easier for network layers to represent the identity mapping. So ResNet have more layers and is able to go deeper but takes much more time. Compared to AlexNet, VGG uses multiple stacked smaller size kernel. These non-linear layers help increase the depth of the network, which enables the VGG to learn more complex features with a lower cost. Thus, VGG performs well on image feature extraction.   

In addition, FatSecret API could not only provide nutrition information, but also could allow users to create their own profiles and keep their food diary. Ideally, users do not need to type in all the foods they consumed for each meal, they could simply take a picture and upload to an app which utilizes our model and collaborates with FatSecret API to provide users the total nutrition they take, including calories, fat, cholesterol, and etc. 

## 9. References

[1] Pouladzadeh, P., Kuhad, P., Peddi, S. V. B., Yassine, A., & Shirmohammadi, S. (2016, May). Food calorie measurement using deep learning neural network. In IEEE International Instrumentation and Measurement Technology Conference Proceedings (I2MTC) (pp. 1-6).

[2] Bossard, L., Guillaumin, M., & Van Gool, L. (2014, September). Food-101–mining discriminative components with random forests. In European Conference on Computer Vision (pp. 446-461). Springer, Cham. Retrieved from https://www.vision.ee.ethz.ch/datasets_extra/food-101/ 

[3] Simayijiang, Z., & Grimm, S. Segmentation with Graph Cuts. Matematikcentrum Lunds Universitet.[Online]. Available: http://www.maths.lth.se/matematiklth/personal/petter/rapporter/graph. pdf.[Diakses 8 Mei 2017].  

The Github pages we used as references are as followed:

https://github.com/stratospark/food-101-keras 

https://github.com/JorgeCeja/food101-tensorflow 

https://github.com/vidhur2k/FoodNet 

https://www.kaggle.com/josephmiguel/vgg16-re-trained-model-78-accuracy/notebook 

https://github.com/jubins/DeepLearning-Food-Image-Recognition-And-Calorie-Estimation

https://github.com/NELSONZHAO/zhihu/blob/master/cifar_cnn/CIFAR-CNN.ipynb 




