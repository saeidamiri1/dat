---
layout: post
title: Regression via Python
description: How to fit regression using Python
date: 2020-06-07
author: Saeid Amiri
published: false
tags: Python R Regression
categories: Fitting-Model
comments: false
---


# Data 
To show how fit the multiple regression using R and Python, we consider the car data [car] which has the car specifications; HwyMPG, Model, etc,. We fit different regression models to predict Highway MPG using the car specifications.

## Contents
- [Import data](#import-data)
- [Preprocessing data](#preprocessing-data)
- [Select variable](#select-variables)
- [Fit model ](#fit-model)
- [Run of codes](#run-of-codes)

## Regression
#### Import data
```
import pandas as pd
car_data = pd.read_csv("https://saeidamiri1.github.io/dat/public/cardata.csv", delimiter=";", decimal=",")
car_data.columns
car_data.head(10)
```
#### Generate a random data
## Preprocessing data
Select the numerical variables to fit the simple regression model. The data have the missing values, so we run an imputuion procedure to fill the missiong values.
```
# Select numerical variables
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
```

## Select variables
To select variable, calculate the correlation
#### Using Python

```
m_car_data.corr().HwyMPG
m_car_data.loc[:,['HwyMPG','Length']].corr()
m_car_data.loc[:,['HwyMPG','Width']].corr()

import seaborn as sns
import matplotlib.pyplot as plt
aa=pd.plotting.scatter_matrix(m_car_data.loc[:,['HwyMPG','GasTank','Rev']])
plt.show()
```

## Fit model
```
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
model_r=lr.fit(m_car_data.loc[:,['Rev','RPM']], m_car_data.HwyMPG)
model_r_pred = model_r.predict(m_car_data.loc[:,['Rev','RPM']])
from sklearn.metrics import r2_score
r2_score(m_car_data.HwyMPG, model_r_pred)

model_g=lr.fit(m_car_data.loc[:,['GasTank']], m_car_data.HwyMPG)
model_g_pred= model_g.predict(m_car_data.loc[:,['GasTank']])
r2_score(m_car_data.HwyMPG,model_g_pred)

model_rg=lr.fit(m_car_data.loc[:,['Rev','GasTank']], m_car_data.HwyMPG)
model_rg_pred = model_rg.predict(m_car_data.loc[:,['Rev','GasTank']])
r2_score(m_car_data.HwyMPG, model_rg_pred )
```


## Polynomial Regression
```
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

x=np.random.randn(100)
y=x**3+.3*np.random.randn(100)
data={'x':x,'y':y}
x_train, x_test, y_train, y_test = train_test_split(data['x'], data['y'], test_size=0.2)

train_data={'x':x_train,'y':y_train}
test_data={'x':x_test,'y':y_test}

def poly_regression(train_data, test_data, degree=0):
    reg_model=LinearRegression()
    x_train = train_data['x'][:, np.newaxis]
    y_train = train_data['y'][:, np.newaxis]
    x_test = test_data['x'][:, np.newaxis]
    y_test = test_data['y'][:, np.newaxis]
    poly_fts =PolynomialFeatures(degree=degree)
    x_train_poly =  poly_fts.fit_transform(x_train)
    x_test_poly =  poly_fts.fit_transform(x_test)
    reg_model.fit(x_train_poly,y_train)
    y_train_pred = reg_model.predict(x_train_poly)
    y_test_pred = reg_model.predict(x_test_poly)
    pred = {'train':y_train_pred, 'test':y_test_pred}
    return (reg_model, pred)

degree=[0,1,2,3,5,7]
preds={}
for d in degree:
    model, pred = poly_regression(train_data, test_data, degree=d)
    preds[d] = {'train':{'x':train_data['x'], 'y':pred['train']}, 'test':{'x':test_data['x'], 'y':pred['test']}}

i=1
for d in degree:
    plt.subplot(2,3,i)
    plt.scatter(preds[d]['test']['x'],preds[d]['test']['y'],s=10,label='Fit with d= {}'.format(d))
    plt.scatter(x_train, y_train,color='m', s=10, label='train data')
    plt.legend()
    i+=1

plt.show(block=False)
```

## Ridge regression
```
from sklearn.linear_model import Ridge

def poly_regression_ridge(train_data, test_data, degree=0,alpha=0):
    reg_model=Ridge(alpha=alpha)
    x_train = train_data['x'][:, np.newaxis]
    y_train = train_data['y'][:, np.newaxis]
    x_test = test_data['x'][:, np.newaxis]
    y_test = test_data['y'][:, np.newaxis]
    poly_fts =PolynomialFeatures(degree=degree)
    x_train_poly =  poly_fts.fit_transform(x_train)
    x_test_poly =  poly_fts.fit_transform(x_test)
    reg_model.fit(x_train_poly,y_train)
    y_train_pred = reg_model.predict(x_train_poly)
    y_test_pred = reg_model.predict(x_test_poly)
    pred = {'train':y_train_pred, 'test':y_test_pred}
    return (reg_model, pred)

degree=[0,1,2,3,5,7]
preds={}
for d in degree:
    model, pred = poly_regression_ridge(train_data, test_data, degree=d,alpha=.5)
    preds[d] = {'train':{'x':train_data['x'], 'y':pred['train']}, \
                'test':{'x':test_data['x'], 'y':pred['test']}}

i=1
for d in degree:
    plt.subplot(2,3,i)
    plt.scatter(preds[d]['test']['x'],preds[d]['test']['y'],s=10,label='Fit with d= {}'.format(d))
    plt.scatter(x_train, y_train,color='m', s=10, label='train data')
    plt.legend()
    i+=1

plt.show(block=False)
```

## Run of codes

<iframe src="https://saeidamiri1.github.io/dat/public/2019-10-14-Regression-via-R-and-Python.html" height="600" width="100%">
 </iframe>


**[⬆ back to top](#contents)**

## References

[car] Consumer Reports: The 1993 Cars - Annual Auto Issue (April 1993), Yonkers, NY: Consumers Union.

### License
Copyright (c) 2020 Saeid Amiri
