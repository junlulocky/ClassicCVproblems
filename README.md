# Hand Written Recognition
This repository contains my test of classic handwritten recognition problem on MNIST. 

The dataset contains the MNIST dataset, which is from [http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz](http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz). 

The start code is from theano tutorial, but my implementation goes beyond it's scope.


## Code Structure
/MNIST/theano
```
|-layers
    |-hiddenLayer.py    # hidden layer
    |-convPoolLayer.py  # convolutional + max pooling layer
    |-logisticRegLayer.py  # usually used for the output
```

```
|-update
    |-activationFunctions.py  # non-linear functions used in NN
    |-updateMethods.py        # implementation of update methods
```

```
|-loader.py   # load data
|-mainLogisticRegression.py  # classify by logistic regression
|-mainMLP.py  # classify by MLP
|-mainCNN.py  # classify by CNN - classic LeNet architecture  
```

/MNIST

The code relies on an unpublished library, so it cannot run. It is just a NN or CNN structure.
```
|-train_mnist_mlp.py   # use MLP to solve MNIST dataset
|-train_mnist_lenet.py # use lenet structure to solve MNIST dataset
```

/CIFAR10

links for CIFAR10:
- [From lasagne](https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py)
- [From binary net](https://github.com/MatthieuCourbariaux/BinaryConnect/blob/lasagne/cifar10.py)
```
train_cifar10_fpwider.py  # Floating Point structure, with flipping, less than 15% error rate
```

The code uses torch, 
```
|-cifar10_lenet.lua   # use LeNet structure to solve CIFAR10 dataset
```

/SVHN

The code relies on an unpublished library, so it cannot run. It is just a NN or CNN structure.
```
|-train_svhn_fpnet.py # Floating point structure, less than 3.3% error rate.
```