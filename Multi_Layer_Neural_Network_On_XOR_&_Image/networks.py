# coding=utf-8
# sample_submission.py
import numpy as np

"""
We assume that the first layer of neurons is an input layer, and
omits to set any biases for those neurons, since biases are only
ever used in computing the outputs from later layers.
"""


class Network(object):
    def __init__(self, sizes, iters=1000, eta=0.01, mbs=30, lamda=0.01, activation_fnc='sigmoid'):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.max_iters = iters
        self.eta = eta
        self.mini_batch_size = mbs
        self.lamda = lamda

        if activation_fnc == 'sigmoid':
            self.activation_fn = self.sigmoid
        elif activation_fnc == 'tanh':
            self.activation_fn = self.tanh

    def sigmoid(self, x, deriv=False):
        if deriv:
            s = self.sigmoid(x, deriv=False)
            return s * (1 - s)
        else:
            return .5 * (1 + np.tanh(.5 * x))
            # Overflow error so changed to above
            # return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def tanh(x, deriv=False):
        if deriv:
            return 1.0 - x ** 2
        else:
            return np.tanh(x)

    def feedforward(self, x, biases, weights):
        # imp line otherwise unknown errors comes
        x = np.reshape(x, (-1, 1))
        for b, w in zip(biases, weights):
            x = self.activation_fn(np.dot(w, x) + b)
        if x >= 0.5:
            return 1
        else:
            return 0

    @staticmethod
    def dcost(predicted, target, cost_function='mse'):
        if cost_function == 'mse':
            return predicted - target

    # stochastic mini batch gradient descent
    # The "training_data" is a list of tuples "(x, y)" representing
    # the training inputs and the desired outputs.

    def fit(self, train_data):
        l = len(train_data)
        for i in range(self.max_iters):
            # print 'Epoch: ',i
            np.random.shuffle(train_data)
            mini_batches = [train_data[j:j + self.mini_batch_size]
                            for j in range(0, l, self.mini_batch_size)]

            for mini_batch in mini_batches:
                self.batch_gradient_descent(mini_batch, l)

    def batch_gradient_descent(self, mini_batch, l):
        lbl_b = [np.zeros(b.shape) for b in self.biases]
        lbl_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            dlbl_b, dlbl_w = self.backpropagation(x, y)
            lbl_b = [nb + dnb for nb, dnb in zip(lbl_b, dlbl_b)]
            lbl_w = [nw + dnw for nw, dnw in zip(lbl_w, dlbl_w)]

        # regularization term added
        self.weights = [(1 - self.eta * (self.lamda / l)) * w - (self.eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, lbl_w)]
        self.biases = [b - (self.eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, lbl_b)]

    """
    The goal of backpropagation is to compute the partial derivatives ∂C/∂w
    and ∂C/∂b of the cost function C with respect to any weight w or bias b
    in the network. Backpropagation is about understanding how changing the
    weights and biases in a network changes the cost function.

    δ(l) denotes the vector of errors associated with all neurons of layer l

    Backpropagation will give us a way of computing δ(l) for every layer &
    then relating those errors to ∂C/∂w and ∂C/∂b.

    ∂C/∂b = δ, i.e. the error is exactly equal to rate of change of cost C
    w.r.t bias b

    ∂C/∂w = a(in)δ(out) where it's understood that a(in) is the activation
    of the neuron input to the weight w, and δ(out) is the error of the
    neuron output from the weight w.
    """

    def backpropagation(self, x, y):
        # layer by layer numpy arrays for biases & weights
        lbl_b = [np.zeros(b.shape) for b in self.biases]
        lbl_w = [np.zeros(w.shape) for w in self.weights]

        # forward pass
        x = np.reshape(x, (-1, 1))
        activation = x
        lbl_activations = [activation]

        lbl_weighted_inp = []
        for b, w in zip(self.biases, self.weights):
            weighted_inp = np.dot(w, activation) + b
            lbl_weighted_inp.append(weighted_inp)
            activation = self.activation_fn(weighted_inp)
            lbl_activations.append(activation)

        # backward pass
        dCost = self.dcost(lbl_activations[-1], y) * self.activation_fn(lbl_weighted_inp[-1], deriv=True)
        lbl_b[-1] = dCost
        lbl_w[-1] = np.dot(dCost, lbl_activations[-2].T)

        # Moving backward from (last - 2)th layer up to 1st layer
        for l in range(self.num_layers - 2, 0, -1):
            weighted_inp = lbl_weighted_inp[l - 1]
            dCost = np.dot(self.weights[l].T, dCost) * self.activation_fn(weighted_inp, deriv=True)
            lbl_b[l - 1] = dCost
            lbl_w[l - 1] = np.dot(dCost, lbl_activations[l - 1].T)

        return lbl_b, lbl_w


class xor_net(Network):
    """
    This is a sample class for miniproject 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.

    """

    def __init__(self, data, labels):
        self.x = data
        self.y = labels
        self.net = Network([self.x.shape[1], 8, 1], 1500, 0.1, 30, 0.001)
        train_data = zip(self.x, self.y)

        self.net.fit(train_data)

        self.b = self.net.biases
        self.w = self.net.weights
        self.params = [(w, b) for w, b in zip(self.w, self.b)]

    def get_params(self):
        """
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b).

        Notes:
            This code will return an empty list for demonstration purposes. A list of tuples of
            weights and bias for each layer. Ordering should from input to output

        """
        return self.params

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
        # return np.random.randint(low=0, high=2, size=x.shape[0])
        prediction = np.empty(x.shape[0], dtype=int)
        i = 0
        for xi in x:
            po = self.net.feedforward(xi, self.b, self.w)
            prediction[i] = po
            i += 1

        return np.array(prediction)
        # return self.net.feedforward(x, zip(*self.params)[0], zip(*self.params)[1])


class mlnn(xor_net):
    """
    At the moment just inheriting the network above.
    """

    def __init__(self, data, labels):
        # super(mlnn, self).__init__(data, labels)
        self.x = data
        self.y = labels
        self.net = Network([self.x.shape[1], 16, 8, 1], 2000, 0.01, 30, 0.001)  # 62
        train_data = zip(self.x, self.y)

        self.net.fit(train_data)

        self.b = self.net.biases
        self.w = self.net.weights
        self.params = [(w, b) for w, b in zip(self.w, self.b)]

    def get_params(self):
        """
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b).

        Notes:
            This code will return an empty list for demonstration purposes. A list of tuples of
            weights and bias for each layer. Ordering should from input to output

        """
        return self.params

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
        # return np.random.randint(low=0, high=2, size=x.shape[0])
        prediction = np.empty(x.shape[0], dtype=int)
        i = 0
        for xi in x:
            po = self.net.feedforward(xi, self.b, self.w)
            prediction[i] = po
            i += 1

        return np.array(prediction)

if __name__ == '__main__':
    pass
