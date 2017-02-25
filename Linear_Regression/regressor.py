# This piece of software is bound by The MIT License (MIT)
# Copyright (c) 2017 Abhishek Kumar
# Code written by : Abhishek Kumar
# Email ID : akuma151@asu.edu

import numpy as np


def feature_normalize(x):
    """
        By normalizing, the mean of each feature becomes 0 and
        the Standard deviation becomes 1. A preprocessing step
        to be done while learning.
    """
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - np.amin(x[:, i])) / (np.amax(x[:, i]) - np.amin(x[:, i]))
    return x


def cost_function(x, y, theta, lamb):
    new_theta = theta[1:, ]
    m = y.size
    predictions = np.dot(x, theta)
    sqError = (predictions - y) ** 2
    # L2: Ridge regularization
    J = (1.0 / (2 * m)) * sqError.sum() + (float(lamb)/2*m)*np.sum(new_theta**2)
    return J


class regressor(object):
    """
    This is a sample class for miniproject 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.

    """

    def __init__(self, data):
        self.x, self.y = data
        # Here is where your training and all the other magic should happen.
        # Once trained you should have these parameters with ready.
        # self.w is the weight and self.b is the weight corresponding to bias
        # basically its the interception data
        # y = b + w.x = b.1 + w.x
        self.w = np.random.rand(self.x.shape[1], 1)
        self.b = np.random.rand(1)
        # lamda the regularization factor and eta the learning rate
        self.lamb = 0.015
        self.eta = 0.1

        if np.amax(self.x) - np.amin(self.x) > 1:
            x_norm = feature_normalize(self.x)
        else:
            x_norm = self.x
        # Adding horizontally a column of 1's to x_train
        x_train = np.c_[np.ones([x_norm.shape[0], 1]), x_norm]

        # Compute Weight via 2 methods.
        # 1. Analytical Solution: np.dot( np.dot( np.inv(np.dot(x.T, x)) , x.T) , y))
        # 2. If 1st fails, Use Gradient descent

        try:
            xtx = np.dot(x_train.T, x_train)
            xtx_inverse = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            # Gradient Descent Method
            # Init random weights for the parameters
            self.theta = np.random.rand(x_train.shape[1], 1)
            J_history = np.zeros((self.y.shape[0], 1))

            # check the dimensionality
            # print x_train.shape, theta.shape

            # 1000 iterations
            for i in range(1000):
                grad = self.gradient(x_train.shape[0], x_train, self.y)
                self.theta[0, :] = self.theta[0, :] - self.eta * grad[0]
                self.theta[1:, :] = self.theta[1:, :] - self.eta * grad[1:]

                J_history[i, 0] = cost_function(x_train, self.y, self.theta, self.lamb)
                if i > 0 and J_history[i-1][0] - J_history[i][0] < 0.0001:
                    # print 'i =', i
                    break
        else:
            xdot = np.dot(xtx_inverse, x_train.T)
            self.theta = np.dot(xdot, self.y)

        self.w = self.theta[1:, :]
        self.b = self.theta[0, :]

    def gradient(self, m, x_train, y_train):
        dtheta = (1.0 / m) * np.dot(x_train.T, (np.dot(x_train, self.theta) - y_train))
        # Escape regularizing the 1st dtheta
        # Means all columns of 0th row
        dtheta[0, :] = dtheta[0, :]
        # all columns from 1st row to last row
        dtheta[1:, :] = dtheta[1:, :] + (float(self.lamb) / m) * dtheta[1:, :]
        return dtheta

    def get_params(self):
        """
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b).

        Notes:
            This code will return a random numpy array for demonstration purposes.

        """
        return self.w, self.b

    def get_predictions(self, x):
        """
        Method should return the outputs given unseen data

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of
                            ``x``
        Notes:
            Temporarily returns random numpy array for demonstration purposes.
        """
        # Here is where you write a code to evaluate the data and produce predictions.
        if np.amax(x) - np.amin(x) > 1:
            x_norm = feature_normalize(x)
        else:
            x_norm = x
        # combine horizontally: this means adding a column of 1's
        x_test = np.c_[np.ones([x_norm.shape[0], 1]), x_norm]
        # combine vertically
        t_theta = np.r_[self.b.reshape(1, 1), self.w]
        prediction = np.dot(x_test, t_theta)

        return prediction

if __name__ == '__main__':
    pass
