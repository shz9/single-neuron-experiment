"""
Author: Shadi Zabad
Date: October 2017

This file implements a class called Neuron, which is supposed to
represent an artificial neuron. We'll explore the learning
capacities of this neuron in the accompanying scripts.
"""

import numpy as np


class Neuron:
    """
    This is a simple class definition for an artificial neuron.
    The class defines 2 attributes: The weights and the bias.
    The class also exposes getter and setter methods to manipulate those attributes.

    It also implements a method that defines the activation
    function (in this case, I implemented softmax but you
    may override that if you wish).

    Finally, I defined a get_output method,
    which returns the most probable class given the input
    and the weights.

    Acknowledgement: The implementation below borrows from
    Arthur Juliani's great tutorial on softmax classifiers:
    https://gist.github.com/awjuliani/5ce098b4b76244b7a9e3#file-softmax-ipynb
    """

    _weights = np.matrix([])

    def __init__(self, w):
        """
        Constructor for the Neuron class
        :param w: An m x n matrix where m is number of inputs to the
                  neuron and n is the number of output classes.
        """
        self._weights = w

    def set_weights(self, w):
        self._weights = w

    def get_weights(self):
        return self._weights

    def get_output(self, input_vals):
        """
        Get the output class, given the inputs and the weights.
        :param input_vals: The input vector
        :return: An integer representation of the output class
        """
        class_scores = self.activation_function(input_vals)
        return np.argmax(class_scores, axis=1)

    def activation_function(self, input_vals):
        """
        This method implements the softmax function
        :param input_vals: Input values that the neuron received
        :return: A vector representing the probability of each class
        """

        # The first step is to do the dot product between
        # the input vector and the weight matrix. This will
        # result in a weighted inputs matrix.
        #
        # In our example, the input_vals vector has
        # dimensions 1x6 and the _weights matrix has
        # dimensions 6x21. This means that the weighted
        # inputs matrix will have dimensions 1x21.
        w_inputs = np.dot(input_vals, self._weights)

        # Then we subtract the maximum value from all elements
        # in the weighted inputs matrix so that the highest value
        # is zero. This is done for numerical stability.
        # You can read more about it here:
        # http://cs231n.github.io/linear-classify/#softmax
        w_inputs -= np.max(w_inputs)

        # Finally, this is where we calculate the probability
        # for each class, given the weighted inputs.
        class_probs = (np.exp(w_inputs).T / np.sum(np.exp(w_inputs), axis=1)).T

        return class_probs
