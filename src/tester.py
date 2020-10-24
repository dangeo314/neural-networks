import mnist_loader
import matrixNetwork
import time


n = matrixNetwork.Network((784, 13, 10))
(training_data, test_data) = mnist_loader.load_data()

start = time.time()
a, f = training_data[0]
n.chooseMiniBatch(training_data, 40, 1000, 2.0, test_data = test_data)
end = time.time()

