---
layout: post
title: Introduction to TensorFlow
description: Introduction to TensorFlow
date: 2020-06-01
author: Saeid Amiri
published: true
tags: Python Tensorflow
categories: Tensorflow Machine_learning Python_learn
comments: false
---


# Introduction 
TensorFlow is a module developed to achieve the machine learning models. It is develped based on manipulating tensors, which are actually multidimensional array.  It supports
the hardware acceleration (GPU), which makes it suitable for machine learning model that need alot of computation. 

## Contents
- [basic](#basic)
- [Regression](#preprocessing-data)
- [Logistics regression](#logistic-regression)
- [MNI](#mni)
- [ANN](#ann)

## basic
```
from __future__ import print_function
import tensorflow as tf
```

Assign constant variable
```
x0 = tf.constant('welcome')
x1 = tf.constant(3)
print(x0)
print(x0.numpy())
print(x1)
print(x1.numpy())
```

Four main operations 
```
x2 = tf.constant(2)
x3 = tf.constant(1)

add = tf.add(x1, x2)
sub = tf.subtract(x1, x2)
mul = tf.multiply(x1, x2)
div = tf.divide(x1, x2)

print(add.numpy())
print(sub.numpy())
print(mul.numpy())
print(div.numpy())
```

More  useful operations.
```
mean = tf.reduce_mean([x1, x2, x3])
sum = tf.reduce_sum([x1, x2, x3])
# Access tensors value.
print("mean =", mean.numpy())
print("sum =", sum.numpy())
```

Matrix multiplications.
```
matrix1 = tf.constant([[1, 3], [2, 4]])
matrix2 = tf.constant([[2, 3], [1, 4]])

add = tf.add(matrix1, matrix2)
mul = tf.matmul(matrix1, matrix2)
```

You can easily use Python list and matrix to tensor object: 
```
vector = tf.convert_to_tensor([1, 3, 2, 4])
matrix = tf.convert_to_tensor([[1, 1, 1],
                               [2, 2, 2],
                               [3, 3, 3]])
```

In the case, you want to modify the tensor object, consider a variable's object
```
variable = tf.Variable([[1, 1, 1],
                        [2, 2, 2],
                        [3, 3, 3]])
```

## Regression
Here, we are going to show how to fit the regression model, we consider car data. The below shows how to select the explanatory and dependent variables.
```
import pandas as pd
car_data = pd.read_csv("https://raw.githubusercontent.com/saeidamiri1/dat/main/public/RVP/cardata.csv", delimiter=";", decimal=",")
car_data.columns
car_data.head(10)
car_data=car_data.select_dtypes(exclude=['object'])
# Check there is any missing values 
car_data.isnull().any().any()
car_data.apply(lambda x: x.isnull().any().any(), axis=0)
# Run Imputation procedure
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(missing_values=np.NaN)
imp.fit(car_data)
m_car_data=pd.DataFrame(imp.transform(car_data))
m_car_data.columns=car_data.columns
from sklearn.linear_model import LinearRegression
X=np.array(m_car_data.loc[:,['Rev','RPM']],dtype='float32')
y=np.array(m_car_data.HwyMPG)

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.scatter(X[:,0], y)
plt.subplot(2, 1, 2)
plt.scatter(X[:,1], y)
plt.show(block=False)
```
 <img src="https://saeidamiri1.github.io/dat/public/tensorflow/figure_1.png" width="500" height="300" />


### Define you model
We consider a linear model
```
def Model(x):
        return tf.matmul(W,x.T) + b
```

### Define Hyper Parameters
```
learning_rate = 0.1
training_steps = 1000
display_step = 100
num_features = X.shape[1]
```

### Assign initial values to parameters
```
W = tf.Variable(tf.random.normal([1, num_features ]))
b = tf.Variable(tf.random.normal([1]))
```

### Define loss function
We need define a loos function, the goal of modeling is to  for the regression model, the loss function is mean squared error (MSE)
```
def loss(y_pred, y_true):
     return(tf.reduce_mean(tf.square(y_pred-y_true)))
```

### Define optimizer
```
optim = tf.keras.optimizers.Nadam(learning_rate)
def run_train(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        current_loss = loss(Model(x), y)
    # Compute gradients.
    gradients = g.gradient(current_loss, [W, b]) 
    # Update W and b following gradients.
    optim.apply_gradients(zip(gradients, [W, b]))
```

### Run Optimization process
```
for step in range(1, training_steps + 1):
    run_train(X, y)
    if step % display_step == 0:
        pred = Model(X)   
        loss_value=loss(pred, y)
        print(f'step:{step}, Loss:{loss_value:.2f},b0:{b.numpy()[0]:.3f},b1:{W.numpy()[0,1]:.3f}, b2:{W.numpy()[0,0]:.2f}')
```

### Graphic display
To see the change of optimazation, you can add the plot
```
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X[:,0],X[:,1], y)
for step in range(1, training_steps + 1):
  run_train(X, y)
  if step % display_step == 0:
    pred = Model(X)   
    loss_value=loss(pred, y)
    print(f'step:{step}, Loss:{loss_value:.2f},b0:{b.numpy()[0]:.3f},b1:{W.numpy()[0,1]:.3f}, b2:{W.numpy()[0,0]:.2f}')
    ax.plot(X[:,0],X[:,1], Model(X).numpy()[0,:])
    plt.pause(0.001)
```

 <img src="https://saeidamiri1.github.io/dat/public/tensorflow/figure_2.png" width="500" height="300" />

## Logistic regression 
Here we consider the logistic regression,
### Import data
```
import pandas as pd
frog_data = pd.read_csv("https://raw.githubusercontent.com/saeidamiri1/dat/main/public/logistic_frog/frogs.csv", delimiter=";", decimal=",")
frog_data.columns
frog_data.head(10)
```

### Check data
Check whether the data has missing value or not?
```
# Check there is any missing values 
frog_data.isnull().any().any()
```

### Select variables 
Select the variables and depict the scatter plot to see the relation between them
```
X=np.array(frog_data.loc[:,['avrain','meanmin']],dtype='float32')
y=np.array(frog_data['pres.abs'],dtype='float32')
#X=X[:,np.newaxis]
import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.scatter(X[:,0], y)
plt.subplot(2, 1, 2)
plt.scatter(X[:,1], y)
plt.pause(0.001)
```
### Define you model
```
def Model(x):
        return tf.nn.softmax(tf.matmul(x, W) + b)
```

### Define Hyper Parameters
```
learning_rate = 0.01
training_steps = 1000
display_step = 2
num_classes = 2 
num_features = 2
```

### Assign initial values to parameters
```
W = tf.Variable(tf.random.normal([num_features, num_classes]))
b = tf.Variable(tf.random.normal([num_classes]))
```

### Define loss function 
```
def loss(y_pred, y_true):
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    return -tf.reduce_sum(y_true * tf.math.log(y_pred))
```
### Define optimizer
```
optim = tf.keras.optimizers.Nadam(learning_rate)
def run_train(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        current_loss = loss(Model(x), y)
    # Compute gradients.
    gradients = g.gradient(current_loss, [W, b]) 
    # Update W and b following gradients.
    optim.apply_gradients(zip(gradients, [W, b]))
```

### Run Optimization process
```
def compute_accuracy(y_pred, y_true):
    correct_pred = tf.equal(tf.cast(tf.argmax(pred, 1),tf.int32), tf.cast(y, tf.int32))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

for step in range(1, training_steps + 1):
    run_train(X, y)
    if step % display_step == 0:
        pred = Model(X)   
        loss_value=loss(pred, y)
        acc_value=compute_accuracy(pred, y)
        print(f'step:{step}, Loss:{loss_value:.2f}, Accuray:{acc_value:.2f},')
```

## MNI 
MNI is a data set of digitilized hand writting that used in many papers to machine learning papers, which includes 60000 images. 

### Import data
```
(x, y), _ = tf.keras.datasets.mnist.load_data()
x = tf.cast(x, tf.float32)
x.shape, y.shape # 28*28 images, 60000 images
for i in np.unique(y):
  print(f'Size of {i} : {tf.reduce_sum(tf.cast(y == i, tf.int32)).numpy()}')

import matplotlib.pyplot as plt
plt.imshow(x[10])
y[10]
plt.show(block=False)

x, y = tf.convert_to_tensor(x), tf.convert_to_tensor(y)
X = tf.reshape(x, (-1, 28*28))
```


### Define you model
```
def Model(x):
        return tf.nn.softmax(tf.matmul(x, W) + b)
```

### Define Hyper Parameters
```
learning_rate = 0.01
training_steps = 1000
display_step = 2
num_classes = 10 
num_features = 784
```

### Assign initial values to parameters
```
W = tf.Variable(tf.random.normal([num_features, num_classes]))
b = tf.Variable(tf.random.normal([num_classes]))
```

### Define loss function 
```
def loss(y_pred, y_true):
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    return -tf.reduce_sum(y_true * tf.math.log(y_pred))
```

### Define optimizer
```
optim = tf.keras.optimizers.Nadam(learning_rate)
def run_train(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        current_loss = loss(Model(x), y)
    # Compute gradients.
    gradients = g.gradient(current_loss, [W, b]) 
    # Update W and b following gradients.
    optim.apply_gradients(zip(gradients, [W, b]))
```

### Run Optimization process
```
def compute_accuracy(y_pred, y_true):
    correct_pred = tf.equal(tf.cast(tf.argmax(y_pred, 1),tf.int32), tf.cast(y_true, tf.int32))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

for step in range(1, training_steps + 1):
    run_train(X, y)
    if step % display_step == 0:
        pred = Model(X)   
        loss_value=loss(pred, y)
        acc_value=compute_accuracy(pred, y)
        print(f'step:{step}, Loss:{loss_value:.2f}, Accuray:{acc_value:.2f},')
```



## ANN
The Artificial Neural network considers the hidden-layer neural networks to generate the information in data which creates connectionist systems. ANN is inspired by biological neural networks that constitute brains, the idea od such systems “learn” to perform tasks by considering examples, generally without being programmed with task-specific rules.  To explain ANN, let consider a vector of input X=[x1, ,,, X_n] and Y=[y_1,.., y_n]. Here we consider  the feedforward neural network (FNN), the goal is Approximate some unknown ideal function ℱ :𝒳⟶𝒴. Let define parametric mapping  𝑦=𝑓(𝑥,𝜃). We want to learn parameters (𝜃) to get a good approximation to ℱ from available sample
Training: Optimize θ to drive 𝑦=𝑓(𝑥,𝜃) closer to ℱ. First we consider  a Single hidden layer neural network with 10 nodes, node is the basic unit of computation (represented by a single circle), it present in the following figure. 


 <img src="/Volumes/F/for_my_website/dat0/practice/tensor/onelayer.jpg" width="500" height="300" />

In this model, 
![](https://latex.codecogs.com/svg.latex?\Large&space;w^1):  the weight of a first connection node
![](https://latex.codecogs.com/svg.latex?\Large&space;w):  the weight of a connection to the output nod
![](https://latex.codecogs.com/svg.latex?\Large&space;b^1):  bias node for the first layer
![](https://latex.codecogs.com/svg.latex?\Large&space;b^{out}):  bias node for the output layer
![](https://latex.codecogs.com/svg.latex?\Large&space;\widehat{y}): output node (a weight sum of the last hidden layer)
![](https://latex.codecogs.com/svg.latex?\Large&space;\phi): Output node activation function




###### two layers
 <img src="/Volumes/F/for_my_website/dat0/practice/tensor/onelayer.jpg" width="500" height="300" />

The model for is 

![](https://latex.codecogs.com/svg.latex?\Large&space;\widehat{y}=\phi(h_1^{(2)}(x)w_1^{(out)}+h_2^{(2)}(x)w_2^{(out)}+...+h_{10}^{(2)}(x)w_{10}^{(out)}+b^{out}))


In general, we have an input layer, H hidden layers, and output layer. The reason of calling it feedforward neural network are: 
- In feed-forward networks the output of units in layer j become input to the units in layers j+1.
- No cross-connection between units in the same layer.
- No backward (“recurrent”) connections from layers downstream
- In fully-connected networks, all units in layer j provide input to all units in layer j+1.

The goal is finding a good set of weights to minimize the error at the output of the network; we can use gradient descent,because the form of the hypothesis formed by the network is: 

- No-closed form solution. 
- Differentiable! Because of the choice of sigmoid units.
- Very complex! Hence direct computation of the optimal weights is not possible.

The following shows the scratch of Gradient-descent and Stochastic Gradient-descent to estimate the parameters. 


<embed src="/Volumes/F/for_my_website/dat0/practice/tensor/gradient.pdf">

In order to put the ANN in code (credit goes to [ref2]), let just consider a two layers with linear activation 

### Code scratch 
#### prepare the data 
```
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
def preprocess(x, y):
  x, y = tf.convert_to_tensor(x), tf.convert_to_tensor(y)
  x = tf.reshape(x, (-1, 28*28))
  x = tf.cast(x, tf.float32)
  y = tf.one_hot(y, depth=10)
  return x, y
x_test, y_test = preprocess(x_test, y_test)
x_train, y_train = preprocess(x_train, y_train)
```
#### 
def accuracy(y_true, y_pred):
  equal_preds = tf.math.argmax(y_pred, axis=1) == tf.math.argmax(y_true, axis=1)
  return tf.reduce_mean(tf.cast(equal_preds, tf.float32))

####
 Define the activated function, here we consider the linear 
```
class Linear(object):
  def __init__(self, input_length, output_length):
    self.W = tf.Variable(tf.random.normal([input_length, output_length]))
    self.b = tf.Variable(tf.zeros(output_length))
    self.trainable_variables = [self.W, self.b]  
  def __call__(self, x): 
    return x @ self.W + self.b 
```

####
Define two layers, the last since we are looking for the multiclass, we use the softmax  
```
class DesignedSystem(object):
  def __init__(self, input_length, output_length):
    self.l1 = Linear(input_length, 50)
    self.l2 = Linear(50, output_length)
    self.trainable_variables = (self.l1.trainable_variables + self.l2.trainable_variables)
  def __call__(self, x):
    return tf.nn.softmax(self.l2(tf.nn.relu(self.l1(x))))
```

####
```
optim = tf.keras.optimizers.SGD(learning_rate)
nn = DesignedSystem(28*28, 10)
```

####
```
for i in range(1000):
  with tf.GradientTape() as tape:
    y_pred = nn(x_train)
    loss_per_example = tf.keras.losses.categorical_crossentropy(y_true=y_train, y_pred=y_pred)
    loss = tf.reduce_mean(loss_per_example)
  gradient = tape.gradient(loss, nn.trainable_variables)
  optim.apply_gradients(zip(gradient, nn.trainable_variables))
  print(f'Loss: {loss.numpy()}  Accuracy: {accuracy(y_pred, y_train)}')
```


### USE API
The tensor API has Artificial Neural Networks (ANN) functions which work very well:   

```
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255
x_test = x_test/255
```

```
Model = tf.keras.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(20),
  tf.keras.layers.ReLU(),
  tf.keras.layers.Dense(10, activation='softmax')
])
```

```
Model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Model.fit(x=x_train, y=y_train, batch_size=200, epochs=20, validation_data=(x_test, y_test))
```

**[⬆ back to top](#contents)**

## References
[car] Consumer Reports: The 1993 Cars - Annual Auto Issue (April 1993), Yonkers, NY: Consumers Union.
[ref1] https://github.com/aymericdamien/TensorFlow-Examples/tree/master/tensorflow_v2
[ref2]https://nicholasvadivelu.com/2019/11/04/intro-to-neural-nets/

### License
Copyright (c) 2020 Saeid Amiri
