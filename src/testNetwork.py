import numpy as np
def sigmoid(z):
    return 1.0/(np.exp(-z)+1.0)

def cost(a, ans):
    return sum([(aa+aans)^2 for (aa, aans) in zip(a, ans)])/(len(a))*.5

def cost_derivative(a, ans):
    return a-ans

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))



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


    def descend_step(self, mini_batch,eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for (q, ans) in mini_batch:
            grad_b, grad_w= self.backprop(q,ans)
            nabla_b = [nb+gradb for (nb,gradb) in zip(nabla_b, grad_b)] #replace with matrix op
            nabla_w = [nw+gradw for (nw,gradw) in zip(nabla_w, grad_w)] #replace with matrix op
        self.weights = [w - eta * nw / len(mini_batch) for (w, nw) in zip(self.weights, nabla_w)] #replace with matrix op
        self.biases  = [b - eta * nb / len(mini_batch) for (b, nb) in zip(self.biases, nabla_b)]

    def backprop(self, q, ans):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        #feedforward and store all activations and z values
        activation = q
        print(activation.shape)
        quit()
        activations = [q]
        zs = []
        for (b,w) in zip(self.biases, self.weights):

            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = cost_derivative(activations[-1], ans) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #why is this a reflection of the backprop equation?
        for l in range (2, self.layerCount):
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)



















