
import mnist_loader
import  matrixNetwork as testNetwork
import network
import time

n = testNetwork.Network((784, 13, 10))
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

start = time.time()
n.chooseMiniBatch(training_data, 6, 100, 1.0, test_data= test_data)
end = time.time()
print "Matrix took {}s".format(end-start)

"""start = time.time()
m = network.Network((784, 13, 10))
m.SGD(training_data, 3, 100, 1.0, test_data)
end = time.time()
print "standard took {}s".format(end-start)"""