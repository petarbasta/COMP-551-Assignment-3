import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from skimage.feature import hog
import tensorflow as tf


class DataProcessing:

    def __init__(self):
        self.training_input_mod = None
        self.training_output_mod = None
        self.test_images_mod = None
        self.conv_layers = None

        self.training_input_original = None
        self.training_output_original = None
        self.hog_features = None

    def load_data(self):
        self.training_input = pd.read_pickle('train_max_x')  # Load data
        self.training_input = self.training_input[0:3, :, :]  # Subsetting first 1000 entries
        self.test_images = pd.read_pickle('test_max_x')

        with open('train_max_y.csv', newline='') as file:
            df = pd.read_csv(file)
            self.training_output = np.array(df)[:, 1]

        self.training_output = self.training_output[0:3]  # Subsetting first 1000 entries

        # printing summary
        print("# of training examples: " + str(self.training_input.shape[0]))
        print("# of test examples: " + str(self.training_output.shape[0]))
        print("X_train shape: " + str(self.training_input.shape))
        print("y_train shape: " + str(self.training_output.shape))
        print("X_test shape: " + str(self.test_images.shape))

        # show example of image
        index = 1
        plt.imshow(self.training_input[index], cmap='Greys')
        plt.show()
        print('y = ' + str(np.squeeze(self.training_output[index])))

        self.conv_layers = {}

    def hog(self):
        dataset = datasets.fetch_mldata("MNIST Original")

        self.training_input_original = np.array(dataset.data, 'int16')
        self.training_output_original = np.array(dataset.target, 'int')

        list_hog_fd = []
        for feat in self.training_output_original:
            fd = hog(feat.reshape(28, 28), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                     visualize=False)
            list_hog_fd.append(fd)

        self.hog_features = np.array(list_hog_fd, 'float64')
