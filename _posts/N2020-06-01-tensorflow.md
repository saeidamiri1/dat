---
layout: post
title: .
description: .
date: 2020-06-01
author: Saeid Amiri
published: false
tags: Python Tensorflow
categories: Tensorflow
comments: false
---


# Data 


## Contents
- [basic](#basic)
- [Regression](#preprocessing-data)
- [Logistics regression](#select-variables)

# basic
```
from __future__ import print_function
import tensorflow as tf
# Assign constant variable
x0 = tf.constant('welcome')
x1 = tf.constant(3)
print(x0)
print(x0.numpy())
print(x1)
print(x1.numpy())


# Four main operations 
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

# More  useful operations.
mean = tf.reduce_mean([x1, x2, x3])
sum = tf.reduce_sum([x1, x2, x3])

# Matrix multiplications.
matrix1 = tf.constant([[1, 3], [2, 4]])
matrix2 = tf.constant([[2, 3], [1, 4]])

add = tf.add(matrix1, matrix2)
mul = tf.matmul(matrix1, matrix2)
```

# Regression
```
import pandas as pd
car_data = pd.read_csv("https://raw.githubusercontent.com/saeidamiri1/saeidamiri1.github.io/master/public/data/cardata.csv", delimiter=";", decimal=",")
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
```

## Define you model
```
def Model(x):
        return tf.matmul(W,x.T) + b
```

## Define Hyper Parameters.
```
learning_rate = 0.1
training_steps = 1000
display_step = 100
num_features = X.shape[1]
```

## Assign initial values to parameters
```
W = tf.Variable(tf.random.normal([1, num_features ]))
b = tf.Variable(tf.random.normal([1]))
```

## Define loss function
```
def loss(y_pred, y_true):
     return(tf.reduce_sum(tf.square(y_pred-y_true)))
```
## Define optimizer
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

## Run Optimization process

for step in range(1, training_steps + 1):
    run_train(X, y)
    if step % display_step == 0:
        pred = Model(X)   
        loss_value=loss(pred, y)
        print(f'step:{step}, Loss:{loss_value:.2f},b0:{b.numpy()[0]:.3f},b1:{W.numpy()[0,1]:.3f}, b2:{W.numpy()[0,0]:.2f}')


## Graphic display
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

# Logistic regression 
### Import data
```
import pandas as pd
frog_data = pd.read_csv("/Volumes/F/for_my_website/saeidamiri1.github.io0/saeidamiri1.github.io/public/general_data/logistic_frog/frogs.csv", delimiter=";", decimal=",")
frog_data.columns
frog_data.head(10)
```

### Check data
Check whether the data has missing value or not?
```
# Check there is any missing values 
frog_data.isnull().any().any()
```

## Select variables
Select the variables and depict the scatter plot to see the relation between them
```
X=np.array(frog_data.loc[:,['avrain','meanmin']],dtype='float32')
y=np.array(frog_data['pres.abs'],dtype='float32')
#X=X[:,np.newaxis]
import matplotlib.pyplot as plt
ax = plt.axes()
ax.scatter(X, y)
plt.pause(0.001)
```



## Define you model
```
def Model(x):
        return tf.nn.softmax(tf.matmul(x, W) + b)
```

## Define Hyper Parameters
```
learning_rate = 0.01
training_steps = 1000
display_step = 2
num_classes = 2 
num_features = 2
```

## Assign initial values to parameters
```
W = tf.Variable(tf.random.normal([num_features, num_classes]))
b = tf.Variable(tf.random.normal([num_classes]))
```

## Define loss function 
```
def loss(y_pred, y_true):
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    return -tf.reduce_sum(y_true * tf.math.log(y_pred)
```
## Define optimizer
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

## Run Optimization process

def compute_accuracy(y_pred, y_true):
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int62))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

for step in range(1, training_steps + 1):
    run_train(X, y)
    if step % display_step == 0:
        pred = Model(X)   
        loss_value=loss(pred, y)
        acc_value=compute_accuracy(pred, y)
        print(f'step:{step}, Loss:{loss_value:.2f}, Accuray:{acc_value:.2f},')


**[⬆ back to top](#contents)**

## References

[car] Consumer Reports: The 1993 Cars - Annual Auto Issue (April 1993), Yonkers, NY: Consumers Union.
[https://github.com/aymericdamien/TensorFlow-Examples/tree/master/tensorflow_v2]

### License
Copyright (c) 2020 Saeid Amiri
