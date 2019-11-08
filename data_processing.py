import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class DataProcessing:

    def __init__(self):
        self.train_images = None
        self.test_images = None

        self.training_input = None
        self.training_output = None

        self.conv_layers = None

    def load_data(self):
        self.training_input = pd.read_pickle('train_max_x')   # Load data
        self.training_input = self.training_input[0:1000, :, :]   # Subsetting first 1000 entries
        self.test_images = pd.read_pickle('test_max_x')

        with open('train_max_y.csv', newline='') as file:
            df = pd.read_csv(file)
            self.training_output = np.array(df)[:, 1]

        self.training_output = self.training_output[0:1000]   # Subsetting first 1000 entries


        # printing summary
        print("# of training examples: " + str(self.training_input.shape[0]))
        print("# of test examples: " + str(self.training_output.shape[0]))
        print("X_train shape: " + str(self.training_input.shape))
        print("y_train shape: " + str(self.training_output.shape))
        print("X_test shape: " + str(self.test_images.shape))


        # show example of image
        index = 374
        plt.imshow(self.training_input[index], cmap='Greys')
        plt.show()
        print('y = ' + str(np.squeeze(self.training_output[index])))

        self.conv_layers = {}

