import numpy as np
import math
import h5py
import scipy
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import tensorflow as tf

from tensorflow.python.framework import ops
from cnn_utils import *

np.random.seed(1)

""" THIS IS A SAMPLE CNN IMPLEMENTED USING TENSORFLOW 


Found on: 
https://github.com/marcopeix/Deep_Learning_AI/blob/master/4.Convolutional%20Neural%20Networks/1.Foundations%20of%20Convolutional%20Neural%20Networks/ConvNet%20in%20TensorFlow.ipynb

"""


class CNN:

    def __init__(self):
        self.learning_rate = 0.009
        self.num_epochs = 400
        self.minibatch_size = 64
        self.print_cost = True

        self.Z3 = None

    # This creates placeholder variables for the features (3 dims of image, and target)
    def create_placeholders(self, n_H0, n_W0, n_C0, n_y):
        X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
        Y = tf.placeholder(tf.float32, [None, n_y])

        return X, Y

    def initialize_parameters(self):
        tf.set_random_seed(1)

        W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

        params = {"W1": W1, "W2": W2}

        return params

    # This defines the 3-layer network with 2 convolutional layers + final fully-connected layer
    def forward_propogation(self, X, params):

        # Retrieve parameter values from params dictionary (created in initialize_parameters)
        W1 = params['W1']
        W2 = params['W2']

        # CONV2D - stride=1, padding=SAME
        Z1 = tf.nn.conv2d(X, strides=[1, 1, 1], padding='SAME')

        # RELU - activation function
        A1 = tf.nn.relu(Z1)

        # MAXPOOL - window=8x8, stride=8, padding=SAME
        P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

        # CONV2D - filters=W2, stride=1, padding=SAME
        Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')

        # RELU - activation function
        A2 = tf.nn.relu(Z2)

        # MAXPOOL - window=4x4, stride=4, padding=SAME
        P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

        # FLATTEN
        P2 = tf.contrib.layers.flatten(P2)

        # FULLY CONNECTED - without non-linear activation function, 6 neurons in output layer
        Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)

        return Z3

    # Computes cost at each layer
    def compute_cost(self, Z3, Y):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))

        return cost

    # Using mini-batch gradient descent for training
    def fit_model(self, X_train, Y_train, X_test, Y_test):
        #ops.reset_default_graph()  # To rerun model without overwriting tf variables
        tf.random.set_seed(1)  # To keep results consistent (tf seed)
        seed = 3  # To keep results consistent (np seed)

        m = X_train.shape[0]    # Number of training examples
        (n_H0, n_W0, n_C0) = X_train.shape
        n_y = Y_train.shape[0]  # Double check this - should be rows?
        costs = []

        X, Y = self.create_placeholders(n_H0, n_W0, n_C0, n_y)  # Create placeholders of the correct shape

        params = self.initialize_parameters()  # Initialize parameters for model

        self.Z3 = self.forward_propogation(X, params)  # Build the forward propogation in the TF graph

        cost = self.compute_cost(self.Z3, Y)  # Add cost function to TF graph

        # Backpropogation
        #     (1) Define Tensorflow optimizer
        #     (2) Use Adam Optimizer that minimizes cost
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        init = tf.global_variables_initializer()  # Initialize all varriables globally

        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:
            sess.run(init)  # Run the initialization

            # Do the training loop
            for epoch in range(self.num_epochs):
                minibatch_cost = 0
                num_minibatches = int(
                    m / self.minibatch_size)  # number of minibatches of size 'minibatch_size' in the training set
                seed = seed + 1
                minibatches = random_mini_batches(X_train, Y_train, self.minibatch_size, seed)

                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch  # Select a minibatch

                    # Run the session to execute the optimizer and the cost
                    # The feed_dict should contain a minibatch for {X, Y}
                    _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                    minibatch_cost += temp_cost / num_minibatches

                # Print the cost for every epoch
                if self.print_cost == True and epoch % 100 == 0:
                    print("Cost after epoch %i: %f" % (epoch, minibatch_cost))

                if self.print_cost == True and epoch % 1 == 0:
                    costs.append(minibatch_cost)

                # Plot the cost
                plt.plot(np.squeeze(costs))
                plt.ylabel("Cost")
                plt.xlabel("Iterations (per tens)")
                plt.title("Learning rate = " + str(self.learning_rate))
                plt.show()

                # Calculate the correct predictions
                predict_op = tf.argmax(self.Z3, 1)
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

                # Calaculate accuracy on test set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print(accuracy)
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
                test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

                print("Training Accuracy: " + str(train_accuracy))
                print("Testing Accuracy: " + str(test_accuracy))

                return train_accuracy, test_accuracy, params

        """
        
        from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
    def fit_model(self, train_X, train_y):
        self.model.add(Convolution2D(32, 3, 3, activation='relu'))      # Convolution
        self.model.add(MaxPooling2D(pool_size=(2,2)))                   # Pooling
        self.model.add(Flatten())                                       # Flattening

        # Full connection
        self.model.add(Dense(output_dim=128, activation='relu'))
        self.model.add(Dense(output_dim=1, activation='sigmoid'))

        # Compiling the CNN
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit(train_X, train_y, validation_split=0.33, epochs=150, batch_size=10)

    def predict(self, test_X, test_y):
        scores = self.model.evaluate(test_X, test_y, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
"""
