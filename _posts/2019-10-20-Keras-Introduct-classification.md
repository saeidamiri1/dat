---
layout: post
title: An introduction to Keras-Classification.  
description: Keras-Classification.
date: 2019-10-20
author: Saeid Amiri
published: true
tags: Classification keras Python
categories: Deep-learning Fitting-Model
comments: false
---
[Keras](https://keras.io/) is a deep learning library written in Python which is running on top of TensorFlow, CNTK, or Theano.  
### Object
- Learn a basic knowledge about the Keras
- Learn classification

## Contents
- [Preparing data](#preparing-data)
- [Keras Model](#keras-model)
- [Building Model](#building-model)
- [Fit model ](#fit-model)
- [Run of codes](#run-of-codes)



## Preparing data
Here we consider the iris data that is well-known data set which often used for the classification, the following codes import data from from the UC Irvine Machine Learning Repository,

```
>>> import pandas as pd
>>> dataset = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None,names=['SepalLength','SepalWidth','
PetalLength','PetalWidth','Species'])
>>> dataset.head()
   SepalLength  SepalWidth  PetalLength  PetalWidth      Species
0          5.1         3.5          1.4         0.2  Iris-setosa
1          4.9         3.0          1.4         0.2  Iris-setosa
2          4.7         3.2          1.3         0.2  Iris-setosa
3          4.6         3.1          1.5         0.2  Iris-setosa

4          5.0         3.6          1.4         0.2  Iris-setosa
>>> dataset.dtypes
SepalLength    float64
SepalWidth     float64
PetalLength    float64
PetalWidth     float64
Species         object
dtype: object
```

Define the dependent variable and covariates. The dependent variable (Species) is categorical and should change to the dummy variables, it can be done using simple codes:

```
X = dataset.values[:,0:4]
Y = pd.get_dummies(dataset['Species'])
```

## Keras Model
The model part of neural networks includes the sequential layers so, the following
code imports the sequential module  which is a linear stack of layers and add it to model

```
from keras.models import Sequential
model = Sequential()
```

## Building Model
You can also simply add layers using `.add()` , one needs to
define the structure of layer. The simplest layer is the  densely-connected NN layer,

```
from keras.layers import Dense
```

The following codes show a  baseline neural network model for our data; first data line contains 10 neurons, since we have four variables, we assign `input_dim=4`. The second line creates output and emphasizes it has three values, one for each class.

```
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
```

## Fitting Model
Once the model is built, it should be compiled, the following code define a loss function, optimizer, and a metric parameter.

```
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
The following functions use X (covariates) and Y (dependent variable), and select subsamples, batch_size, to train the model  

```
model.fit(X, Y, nb_epoch=200, batch_size=10)
accuracy = model.evaluate(X,Y)
print("%s: %.2f%%" % (model.metrics_names[1], accuracy[1]*100))
```

## References
There are very interesting websites that might be useful to <br>
[Web1] https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
