---
layout: post
title: Multiple Regression via R and Python
description: How to fit Multiple regression using R and Python
date: 2019-10-14
author: Saeid Amiri
published: true
tags: Python R Regression
categories: Python_learn R_learn Fitting-Model
comments: false
---


# Data 
To show how fit the multiple regression using R and Python, we consider the car data [car] which has the car specifications; HwyMPG, Model, etc,. We fit different regreesion models to predict Highway MPG using the car specifications.

## Contents
- [Import data](#import-data)
- [Preprocessing data](#preprocessing-data)
- [Select variable](#select-variables)
- [Fit model ](#fit-model)
- [Run of codes](#run-of-codes)

## Import data
#### Using R

```
car_data=read.csv2("https://raw.githubusercontent.com/saeidamiri1/saeidamiri1.github.io/master/public/data/cardata.csv")
names(car_data)
head(car_data)
```

#### Using Python

```
import pandas as pd
car_data = pd.read_csv("https://raw.githubusercontent.com/saeidamiri1/saeidamiri1.github.io/master/public/data/cardata.csv", delimiter=";", decimal=",")
car_data.columns
car_data.head(10)
```

## Preprocessing data
Select the numerical variables to fit the simple regression model. The data have the missing values, so we run an imputuion procedure to fill the missiong values.

#### Using R

```
# Select numerical variables
v0=as.numeric(which(unlist(lapply(car_data, is.numeric))))
# Check there is any missing values 
car_data=car_data[,v0]
anyNA(car_data)
apply(is.na(car_data),2,sum)
# Run Imputation procedure
library(mice)
m_car_data<-mice(car_data,print=FALSE)
m_car_data<-complete(m_car_data)
anyNA(m_car_data$imp)
```

#### Using Python

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

#### Using R

```
cor(m_car_data,m_car_data$HwyMPG,)
cor(m_car_data$HwyMPG,m_car_data$GasTank)
cor(m_car_data$HwyMPG,m_car_data$Rev)

plot(car_data[,c('HwyMPG','GasTank','Rev')])
```

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
### Using R

```
model_r=lm(HwyMPG ~ Rev, data=m_car_data)
summary(model_r)$r.squared

model_g=lm(HwyMPG ~ GasTank, data=m_car_data)
summary(model_g)$r.squared

model_rg=lm(HwyMPG ~ Rev+GasTank, data=m_car_data)
summary(model_rg)$r.squared
```

### Using Python

```
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
model_r=lr.fit(m_car_data.loc[:,['Rev']], m_car_data.HwyMPG)
model_r_pred = model_r.predict(m_car_data.loc[:,['Rev']])
from sklearn.metrics import r2_score
r2_score(m_car_data.HwyMPG, model_r_pred)

model_g=lr.fit(m_car_data.loc[:,['GasTank']], m_car_data.HwyMPG)
model_g_pred= model_g.predict(m_car_data.loc[:,['GasTank']])
r2_score(m_car_data.HwyMPG,model_g_pred)

model_rg=lr.fit(m_car_data.loc[:,['Rev','GasTank']], m_car_data.HwyMPG)
model_rg_pred = model_rg.predict(m_car_data.loc[:,['Rev','GasTank']])
r2_score(m_car_data.HwyMPG, model_rg_pred )
```

## Run of codes

<iframe src="https://saeidamiri1.github.io/source/post/2019-10-14-Regression-via-R-and-Python.html" height="600" width="100%">
 </iframe>


**[⬆ back to top](#contents)**

## References

[car] Consumer Reports: The 1993 Cars - Annual Auto Issue (April 1993), Yonkers, NY: Consumers Union.

### License
Copyright (c) 2019 Saeid Amiri
