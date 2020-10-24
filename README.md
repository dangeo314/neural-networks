# neural-networks
This program creates and trains a neural network to classify the MNIST handwritten digits dataset.
Right now, I am using a simple feedforward neural network.

To train the network on your machine, run tester.py. When the program runs, it will output accuracy based on the MNIST test dataset.  
Libraries needed:
1. numpy 1.18.1
2. Python 3.7.6

I made this program referring to Michael Nielsen's book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com). 


#Future Additions to the Code
1. Comment the code to provide more details on network architecture.
2. Add a save feature to save the network
3. Experiment with different network architectures. Convolutional networks have worked well for this dataset, so that
may be something I look into. However, I think I might make the project use TensorFlow or pyTorch if I make a different
architecture. 