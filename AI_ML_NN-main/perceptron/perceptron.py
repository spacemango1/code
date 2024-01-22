# requirements
# pip install numpy
# pip install matplotlib

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import numpy as np


# y = mx + c
# y = Wi * x + Wb * 1.0
# Define a Perceptron class, which models a basic neural network unit.
class Perceptron:
    def __init__(self, n, lr, activation=None):
        # Initialize the Perceptron with random weights, learning rate, and an activation function.
        # The np.random.uniform(0, 1, n) returns an array length n with the random numbers that are uniformly distributed (all values are equally likely to occur) between 0 (inclusive) and 1 (exclusive).
        self.weights = np.random.uniform(0, 1, n)  # Initialize random weights.

        # The learning rate determines how much the model's weights should be adjusted during each iteration of the training process.
        # A higher learning rate means larger weight updates, while a lower learning rate results in smaller updates.
        # The learning rate represents a trade-off between training speed and accuracy.
        self.lr = lr  # Set the learning rate.

        # The activation function in a neural network, including a Perceptron, is a mathematical /Users/lauradowell/bsc-coding-five-main/AI_ML_NN-main/perceptron/linear_relationship.csvfunction that determines whether the neuron (or Perceptron in this case) should be activated (i.e., produce an output) based on the weighted sum of its inputs.
        # The activation function introduces non-linearity to the network, allowing it to learn and represent complex patterns in data.
        # Set the activation function (if provided).
        self.activation = activation

    def train(self, inputs, desired):
        # Train the Perceptron by adjusting its weights based on the error between the desired and predicted output.
        guess = self.feed_forward(inputs)  # Make a prediction.
        print(f'The guess value: {guess}.')
        error = desired - guess  # Calculate the prediction error.
        # Update the weights using the learning rate, error, and input values.
        print(f'The error value: {error}.')

        new_weights = []  # Create a new list for updated weights.

        for i in range(len(self.weights)):
            weight = self.weights[i]  # Get the current weight.
            input_value = inputs[i]  # Get the corresponding input value.

            # Calculate the updated weight using the learning rate and error.
            updated_weight = weight + (self.lr * error * input_value)

            # Append the updated weight to the new list.
            new_weights.append(updated_weight)

        # Replace the old weights with the updated weights.
        self.weights = new_weights

    def feed_forward(self, inputs):
        # Perform the feed-forward computation to get the predicted output.

        total = 0  # Initialize the total to zero.

        # a for loop to iterate from 0 to one less than the length of the self.weights array.
        for i in range(len(self.weights)):
            weight = self.weights[i]  # Get the weight at the current index.
            # Get the input value at the current index.
            input_value = inputs[i]

            # Calculate the product of weight and input.
            weighted_input = weight * input_value
            # Add the weighted input to the running total.
            total += weighted_input

        if self.activation is None:
            # If no activation function is specified, return the total directly.
            return total
        else:
            # Apply the specified activation function.
            return self.activate(total)

    def get_weights(self):
        # Get the current weights of the Perceptron.
        return self.weights  # Return the weights.


# read the data from a csv file
linear_relationship = np.loadtxt(
    "./perceptron/linear_relationship.csv", delimiter=",", dtype=str)
# convert the data array to float
linear_relationship = np.array(linear_relationship, dtype=float)

linear_relationship_column1_input = linear_relationship[:, 0]
linear_relationship_column2_output = linear_relationship[:, 1]

plt.plot(linear_relationship_column1_input, linear_relationship_column2_output)
# should be a liner graph y = 2x
# plt.show()


# in the current perceptron we are going to train it to output linear relationship y = Wi*x + 1.0*Wb
# where 1.0 is the bias thats why we need two values in the perceptron
# you can change the learning rate and see how it affects the output
# lower learning rate will not be enough to train the perceptron as we only have 5 pairs of values to train on
# grater learning rate values will cause the weights to be unbalanced, unable to come to an average value
perceptron = Perceptron(2, 0.000001)

for i in range(len(linear_relationship_column1_input)):
    print(f'Actual input: {linear_relationship_column1_input[i]}')
    print(f'Actual output: {linear_relationship_column2_output[i]}')
    perceptron.train([linear_relationship_column1_input[i],
                     1.0], linear_relationship_column2_output[i])
    print(f'Current weights: {perceptron.get_weights()}')
    print('--------------------------------------------------------')

# predict your values here
inputValue = -1234567890
print('The input value: ' + str(inputValue))
getPrediction = perceptron.feed_forward([inputValue, 1.0])
print('The predicted value: ' + str(getPrediction))
correctPercentage = inputValue * 2 / getPrediction * 100
print('The prediction is: ' + str(correctPercentage) + '% correct.')
