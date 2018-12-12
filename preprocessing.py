import numpy as np
import cv2
import gc
import random

# size of images to use.
W = 128

# return the data in the file.
def read_file(fn):
    data = []
    with open(fn, "r") as fd:
        for line in fd:
            data.append(line.strip())
    return data

# return the image padded to (512, 512, 3) and downsize to (128, 128, 3).
def read_image(fn):
    image = cv2.imread(fn,cv2.IMREAD_COLOR)
    x, y, _ = image.shape
    padded_image = np.pad(image, ((0, 512 - x), (0, 512 - y), (0, 0)), 'edge')
    return cv2.resize(padded_image,(W, W))

# prepare a map of labels to their result index.
def prepare_labels(dataset):
    labels = set()
    for i in dataset:
        labels.add(i.split("/")[0])
    
    labels_map = {}
    for p, l in enumerate(sorted(list(labels))):
        labels_map[l] = p
    return labels_map

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
    
if __name__ == "__main__":
    main()