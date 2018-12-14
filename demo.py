import numpy as np
import cv2
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.normalization import batch_normalization
import csv
import sys
import PyWriteHtml
    
W = 128
NUM_CLASSES = 101
TOP = 5
    
def read_image(fn):
    image = cv2.imread(fn,cv2.IMREAD_COLOR)
    #print(image)
    x, y, _ = image.shape
    padded_image = np.pad(image, ((0, 512 - x), (0, 512 - y), (0, 0)), 'edge')
    return cv2.resize(padded_image,(W, W))

#read label
def read_label():
    labels = []
    with open('label.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            labels.append(row['label'])
    return labels
    
def load_model():
    tf.reset_default_graph()
    
    image_aug = ImageAugmentation()
    image_aug.add_random_blur(1)
    image_aug.add_random_flip_leftright()
    
    net = input_data(shape=[None] + [W, W, 3], data_augmentation=image_aug)
    net = conv_2d(net, 96, 5, strides=2, activation='relu')
    net = batch_normalization(net)
    net = max_pool_2d(net, 2)
    net = dropout(net, 0.8)
            
    net = conv_2d(net, 256, 5, strides=2, activation='relu')
    net = batch_normalization(net)
            
    net = max_pool_2d(net, 2)
    net = dropout(net, 0.8)
            
    net = conv_2d(net, 384, 3, activation='relu')
    net = conv_2d(net, 384, 3, activation='relu')
    net = conv_2d(net, 256, 3, activation='relu')
    net = batch_normalization(net)
    net = max_pool_2d(net, 2)
    net = dropout(net, 0.8)
            
    net = fully_connected(net, 1024, activation='tanh')
    net = dropout(net, 0.5)
    net = fully_connected(net, 1024, activation='tanh')
    net = dropout(net, 0.5)
    net = fully_connected(net, NUM_CLASSES, activation='softmax')
    net = regression(net, optimizer='adam',loss='categorical_crossentropy', learning_rate=0.0001)
    
    clf = tflearn.DNN(net)
    clf.load('../results/checkpoint_1543800554.7789965/tmp.model')
    return clf

def classify_image(path, clf):
    #images = np.zeros((1, W, W, 3), dtype=np.uint8)
    images = np.zeros((W, W, 3), dtype=np.uint8)
    images= read_image(path)
    np.save("/Users/wd4446/Box Sync/Adv_Predictive_Modeling/Image_Recognition/model2.0/demo_test/read_x.npy", images)
    read_x = np.load("/Users/wd4446/Box Sync/Adv_Predictive_Modeling/Image_Recognition/model2.0/demo_test/read_x.npy")
    value = np.squeeze(np.array(clf.predict(np.reshape(read_x, [1, 128, 128, 3]))))
    return value.argsort()[-TOP:][::-1]


def main(argv):
    labels = read_label()
    #path = choose_image(argv)
    #path = "/Users/wd4446/Box Sync/Adv_Predictive_Modeling/Image_Recognition/model2.0/demo_test/2432.jpg"
    top5 = classify_image(argv, load_model()).astype(int)
    for i in range(TOP):
        print(labels[top5[i]])
    PyWriteHtml.show(labels[top5[0]])
    
        
if __name__ == "__main__":
    main(sys.argv[1])
    
    