import pandas as pd 
import matplotlib.pyplot as plt

#Load data
train_images = pd.read_pickle('train_max_x') 
test_images = pd.read_pickle('test_max_x')

#Display first image
plt.imshow(train_images[0], cmap='Greys')
plt.show()