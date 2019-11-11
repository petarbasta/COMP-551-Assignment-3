import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from sklearn.externals import joblib



# #Load data
# train_images = pd.read_pickle('train_max_x')
# test_images = pd.read_pickle('test_max_x')
#
# #Display first image
# plt.imshow(train_images[0], cmap='Greys')
# plt.show()

with open('test_max_x', 'rb') as file:
    arr = pickle.load(file)

#Clean data
def cleanData(dataset):
    dataset[dataset > 200] = 255
    dataset[dataset < 255] = 0
    for i, binimage in enumerate(dataset):
        binimage = np.array(binimage, dtype=np.uint8)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binimage)
        arr = stats[1:, -1]
        image = np.zeros((labels.shape))
        for j in range(0, nlabels-1):
            if arr[j] >= 25:
                image[labels == j + 1] = 255
        dataset[i] = image
    return dataset


arr = cleanData(arr)
plt.imshow(arr[3456], cmap='Greys')
plt.show()


