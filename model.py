import numpy as np
import tensorflow as tf
import tflearn
import time
import random
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.normalization import batch_normalization

def shuffle_data(train_x, train_y):
    data = []
    for i in range(len(train_x)):
        data.append((train_x[i], train_y[i]))
    random.Random(10086).shuffle(data)
    return np.array([i[0] for i in data]), np.array([i[1] for i in data])

# models.
def cnn_model(x_shape, y_shape, archi="AlexNet"):
    image_aug = ImageAugmentation()
    image_aug.add_random_blur(1)
    image_aug.add_random_flip_leftright()
    image_aug.add_random_flip_updown()
    image_aug.add_random_rotation()
    image_aug.add_random_90degrees_rotation()
    
    # AlexNet, replacing local normalization with batch normalization.
    if archi == "AlexNet":
        net = input_data(shape=[None] + list(x_shape[1:]), data_augmentation=image_aug)
        net = conv_2d(net, 96, 7, strides=2, activation='relu')
        
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
        
        net = fully_connected(net, 4096, activation='tanh')
        net = dropout(net, 0.5)
        net = fully_connected(net, 4096, activation='tanh')
        net = dropout(net, 0.5)
        net = fully_connected(net, y_shape[1], activation='softmax')
        net = regression(net, optimizer='adam',
                         loss='categorical_crossentropy', learning_rate=0.0001)

    # ResNet, with dropout.
    if archi == "ResNet":
        n = 5
        net = tflearn.input_data(shape=[None] + list(x_shape[1:]),
                                 data_augmentation=image_aug)
        net = tflearn.conv_2d(net, 16, 5, strides=2, regularizer='L2', weight_decay=0.0001)
        net = tflearn.residual_block(net, n, 16)
        net = tflearn.residual_block(net, 1, 32, downsample=True)
        net = tflearn.dropout(net, 0.8)
        net = tflearn.residual_block(net, n-1, 32)
        net = tflearn.residual_block(net, 1, 64, downsample=True)
        net = tflearn.dropout(net, 0.8)
        net = tflearn.residual_block(net, n-1, 64)
        net = tflearn.batch_normalization(net)
        net = tflearn.activation(net, 'relu')
        net = tflearn.global_avg_pool(net)
        net = tflearn.fully_connected(net, y_shape[1], activation='softmax')
        net = tflearn.regression(net, optimizer='adam',
                                 loss='categorical_crossentropy', learning_rate=0.0001)
        
    return net

# training loop.
def cnn_train(train_x, train_y, test_x, test_y, num_epoch=100):
    tf.reset_default_graph()
    net = cnn_model(train_x.shape, train_y.shape)
    clf = tflearn.DNN(net)
    
    max_acc = 0.0
    ts = time.time()
    with open("./results/accuracy_{}.txt".format(ts), "w") as f:
        for i in range(num_epoch):
            # training.
            train_x, train_y = shuffle_data(train_x, train_y)
            clf.fit(train_x, train_y, shuffle=True,
                    n_epoch=1, batch_size=64, show_metric=True)
            
            # evaluate after every 5 epoch.
            if (i + 1) % 5 == 0:
                value = np.squeeze(np.array([clf.predict(np.reshape(
                        i, [1]+list(test_x.shape[1:]))) for i in test_x]))
                correct = 0.0    
                for i in range(test_x.shape[0]):
                    if np.argmax(value[i]) == np.argmax(test_y[i]):
                        correct += 1
                acc = correct / test_x.shape[0]
                f.write("Accuracy: {}\n".format(acc))
                print("Accuracy: ", acc)
                if acc > max_acc:
                    max_acc = acc
                    clf.save("./results/checkpoint_{}/tmp.model".format(ts))
        

def main():
    train_x = np.load("./../data/train_x.npy")
    train_y = np.load("./../data/train_y.npy")
    test_x = np.load("./../data/test_x.npy")
    test_y = np.load("./../data/test_y.npy")
    cnn_train(train_x, train_y, test_x, test_y)

if __name__ == "__main__":
   main()
