
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
- [Data Preprocessing and Exploration](#data-preprocessing-and-exploration)
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

We obtained our data from the original paper using random forest (2014) [here](https://www.vision.ee.ethz.ch/datasets_extra/food-101/ "data source"). 
  
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

Since the dataset is publicly available from the website, we simply downloaded the dataset and then unzipped it into our laptops.  
The unzipped dataset contains two folders:  
```
images
meta
```
Images can be found in the "images" folder and are organized per class.
There are also two txt files providing some instructions on the use of this dataset:  
```
license_agreement.txt
README.txt
```
 
All image IDs are unique and correspond to the foodspotting.com review IDs. Thus the original articles can retrieved trough
  http://www.foodspotting.com/reviews/<image_id> or through the foodspotting api (http://www.foodspotting.com/api).

The test/train splitting used in the experiment can be found in
the "meta" directory.


### Relevant Characteristics

## 4. Data Preprocessing and Exploration

### Feature engineering/selection
### Relevant Plots

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
