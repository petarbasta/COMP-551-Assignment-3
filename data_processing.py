import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataProcessing:

    def __init__(self):
        self.train_images = None
        self.test_images = None

        self.training_input = None
        self.training_output = None

    def load_data(self):
        self.training_input = pd.read_pickle('train_max_x')   # Load data
        self.test_images = pd.read_pickle('test_max_x')

        with open('train_max_y.csv', newline='') as file:
            df = pd.read_csv(file)
            self.training_output = np.array(df)[:, 1]


        plt.imshow(self.train_images[0], cmap='Greys')   #Display first image
        plt.show()

