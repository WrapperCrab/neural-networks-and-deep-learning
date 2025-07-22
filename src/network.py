"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import time

# Third-party libraries
import numpy as np
from PIL import Image


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch_matrix(mini_batch, eta)
            time2 = time.time()
            if test_data:
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(
                    j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(j, time2-time1))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def update_mini_batch_matrix(self, mini_batch, eta): # After trying and failing, I found some working code online
        m = len(mini_batch)

        # transform to (input x batch_size) matrix
        X = np.asarray([_x.ravel() for _x, _y in mini_batch]).transpose() # Very efficient way of combining input into one matrix ig
        # transform to (output x batch_size) matrix
        Y = np.asarray([_y.ravel() for _x, _y in mini_batch]).transpose()

        nabla_b, nabla_w = self.backprop_matrix(X, Y)
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop_matrix(self, X, Y):
        nabla_b = [0 for i in self.biases] # Add the actual matrices later
        nabla_w = [0 for i in self.weights]

        # feedforward
        activation = X
        activations = [X] # list to store all the activations, layer by layer
        Zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            Z = np.dot(w, activation) + np.kron(b, np.ones([1, Y.shape[1]])) # I had found a different, 3 line way to make the B matrix
            Zs.append(Z)
            activation = sigmoid(Z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], Y) * sigmoid_prime(Zs[-1])
        nabla_b[-1] = np.sum(delta, axis=1).reshape([len(delta), 1]) # reshape to (n x 1) matrix # This makes sense when you remember we find sum over deltas to get new b values
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # This is  the part I couldn't figure out

        for l in range(2, self.num_layers):
            Z = Zs[-l]
            sp = sigmoid_prime(Z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1).reshape([len(delta), 1]) # reshape to (n x 1) matrix
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose()) # This is  the part I couldn't figure out

        # I'm guessing the nabla_w part is the biggest time saver. 
        # The dot product has sum baked into it. It has a property that the product of the combined vector matrices
        # is equal to the sum of the matrices multiplied apart
        # [d_1 d_2][a_1 a_2]^T = d_1 a_1^T + d_2 a_2^T
        # Why does this property feel so unfamiliar to me?

        return (nabla_b, nabla_w)


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \ partial C_x /
        \ partial a for the output activations."""
        return (output_activations-y)
        # return -(output_activations-y) # TEST negating cost function

    def evaluate_image(self, path):
        """Runs the network on the passed image at path after converting it to useable data
        Prints the resulting activations in the output layer"""
        image = Image.open(path).convert("L")
        data = np.array(image)
        data = (255-data) / 255.0

        data = np.reshape(data, (784,1))
        a = self.feedforward(data)

        print(a)
        print(np.argmax(a))

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
    # return (1.0/np.pi)*(np.atan(z)) + 0.5

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
    # return (1.0/np.pi)*(1/(1 + np.square(z)))
