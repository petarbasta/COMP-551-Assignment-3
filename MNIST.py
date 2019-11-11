import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2




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
print(arr.shape)
plt.imshow(arr[22], cmap = 'Greys')
print(arr[1].shape)
plt.show()

def getBoundingBoxes(img):
    # smooth the image to avoid noises
    img = cv2.medianBlur(img,5)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(img,255,1,1,11,2)
    thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    thresh = cv2.dilate(thresh,None,iterations = 15)
    thresh = cv2.erode(thresh,None,iterations = 15)

    # Find the contours
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, find the bounding rectangle and draw it

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        ROI = img[y:y+h, x:x+w]

        cv2.imshow('img',ROI)
        cv2.waitKey(0)

img = cv2.imread('1.png', 0) 
print(img.shape)
getBoundingBoxes(img)