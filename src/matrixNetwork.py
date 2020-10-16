import numpy as np
import time
def sigmoid(z):
    return 1.0/(np.exp(-z)+1.0)

def cost(a, ans):
    return sum([(aa+aans)^2 for (aa, aans) in zip(a, ans)])/(len(a))*.5

def cost_derivative(a, ans):
    return a-ans

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
backpropCt=0


class Network(object):
    def __init__(self, sizes):
        self.layerCount = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for (x, y) in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for (b,w) in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def chooseMiniBatch(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.descend_step(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)
        print "backpropCt = {}".format(backpropCt)

    def descend_step(self, mini_batch,eta):
        grad_b, grad_w= self.backprop(mini_batch)
        global backpropCt
        backpropCt =  backpropCt +1
        for l in range(0, self.layerCount-1):
            """print "adjusted minibatch before b: {} w:{} ".format(
                self.biases[l].shape, self.weights[l].shape)"""

            self.weights[l] = self.weights[l] - eta * grad_w[l]
            self.biases[l] = self.biases[l] - eta * grad_b[l]
            """print "adjusted minibatch after b: {} w:{} ".format(
                self.biases[l].shape, self.weights[l].shape)"""

    def backprop(self, mini_batch):
        """Returns the average gradient of cost function for all examples in minibatch"""
        start = time.time()
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        #feedforward and store all activations and z values
        start2 = time.time()
        activation = np.hstack([q for (q,a) in mini_batch])
        # print(activation.shape)
        ans = np.hstack([a for (q,a) in mini_batch])
        # print(ans.shape)
        end = time.time()
        # print "digesting minibatch took {}s".format(end - start2)

        activations = [activation]
        zs = []
        for (b,w) in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = cost_derivative(activations[-1], ans) * sigmoid_prime(zs[-1])
        nabla_b[-1] = np.array([sum(arr) for arr in delta]).reshape(-1,1)/len(mini_batch)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())/len(mini_batch) #why is this a reflection of the backprop equation?

        for l in range (2, self.layerCount):
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sigmoid_prime(zs[-l])
            nabla_b[-l] = np.array([sum(arr) for arr in delta]).reshape(-1,1)/len(mini_batch)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())/len(mini_batch)
        end = time.time()
        #print "backprop took {}s".format(end - start)

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)