import mnist_loader
import matrixNetwork as testNetwork
import network2
import time

n = network2.Network((784, 13, 10))
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
start = time.time()
n.SGD(training_data[:1000], 30, 100, 4, 1, validation_data[:100], monitor_evaluation_accuracy=True, monitor_training_cost=True)
end = time.time()
print "exec took {}s".format(end-start)
