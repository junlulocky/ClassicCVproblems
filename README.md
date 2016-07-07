# PyML
This repository contains my test of classic handwritten recognition problem on MNIST. 

The dataset contains the MNIST dataset, which is from [http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz](http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz). 

The start code is from theano tutorial, but my implementation goes beyond it's scope.


## Code Structure
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
