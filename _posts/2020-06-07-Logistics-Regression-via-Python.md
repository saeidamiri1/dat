---
layout: post
title: Logistic Regression via Python
description: How to fit Logistics regression using Python
date: 2020-06-07
author: Saeid Amiri
published: true
tags: Python logisticRegression
categories: Python_learn Fitting-Model Machine_Learning
comments: false
---

# Logistic Regression
The logistics regression is used when the response variable is a binary variable, such as (0 or 1), (live or die), and (fail or succeed). 

## Contents
- [Mathematical presentation](#Mathematical-presentation)
- [data](#data)
    - Necessary packages
    - Import data
    - Check data
    - Select variables
- [Fit model ](#fit-model)
    - Fit directrly 
    - Fit using Cross-validation
- [Model evaluation](#Model-evaluation)

## Mathematical presentation
The model consideres the probability of event instead the event in model: 
 ![](https://latex.codecogs.com/svg.latex?\frac{P(Y_{i}=1)}{1-P(Y_{i}=1)}=\beta_{0}+\beta_{1}X_{1i}+\cdots+\beta_{k}X_{ki})

which can be converted to 
 ![](https://latex.codecogs.com/svg.latex?P(Y_{i}=1)=\frac{exp(\beta_{0}+\beta_{1}X_{1i}+\cdots+\beta_{k}X_{ki})}{1+exp(\beta_{0}+\beta_{1}X_{1i}+\cdots+\beta_{k}X_{ki})})
or 
 ![](https://latex.codecogs.com/svg.latex?P(Y_{i}=1)=\frac{1}{1+exp[-(\beta_{0}+\beta_{1}X_{1i}+\cdots+\beta_{k}X_{ki})]})

Unlike the regular regression, we can not apply mean square error to find the estimates of coefficients, they can be found using Maximum likelihood: 

![](https://latex.codecogs.com/svg.latex?\prod_{i=1}^{n}p(Y_{i}=y_{i}|X_{i})=\prod_{i=1}^{n}\pi_{i}^{y_{i}}(1-\pi_{i})^{1-y_{i}}=\prod_{i=1}^{n}\left(\frac{\pi_{i}}{1-\pi_{i}}\right)^{y_{i}}(1-\pi_{i}))

where ![](https://latex.codecogs.com/svg.latex?\frac{\pi_{i}}{1-\pi_{i}})
 is the odds for ![](https://latex.codecogs.com/svg.latex?P(Y_{i}=1|X_{i})). The log-likelihood can be obtained as
![](https://latex.codecogs.com/svg.latex?l({\beta};y)=\sum_iy_{i}X_{i}'{\beta}-\sum_i\ln(1+e^{X_{i}'{\beta}}))


## Data 
To show how fit the logistics regression using Python, we consider the [frogs] data, which is about the distribution of the Southern Corroboree frog, which occurs in the Snowy Mountains area of New South Wales, Australia.

### Necessary packages
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
```

### Import data
```
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

### Select variables
Select the variables and depict the scatter plot to see the relation between them
```
colors = ["" for x in range(frog_data.shape[0])]
colors=pd.array(colors)
colors[frog_data['pres.abs']==1]=['red']
colors[frog_data['pres.abs']==0]=['blue']
df=frog_data[['meanmin','pres.abs']]
ax=sns.regplot(x='meanmin', y="pres.abs", data=df, fit_reg=True, x_jitter=0.1,y_jitter=0.1, scatter_kws={'alpha': 0.5,'s':20})  
ax.legend().set(ylabel='Present of frogs', xlabel='mean minimum Spring temperature')
plt.show()
```

## Fit model
### Fit directrly
Logistic regression can be fit directly 
```
x_train, x_test, y_train, y_test = train_test_split(df['meanmin'], df['pres.abs'], test_size=0.3)
x_train=x_train[:,np.newaxis]
x_test=x_test[:,np.newaxis]
logistic_model = LogisticRegression(random_state=0)
logistic_model.fit(x_train, y_train)
y_train_pred=logistic_model.predict(x_train)
y_test_pred=logistic_model.predict(x_test)
logistic_model.coef_
```

### Fit using Cross-validation 
We can run cross-validation to find the best estimate
```
cross_val_model = LogisticRegression(random_state=0)
scores = cross_val_score(cross_val_model, x_train, y_train, cv=10)
print(scores)
print(np.mean(scores))

penalty = ['l2']
C = np.logspace(0, 4, 10)
random_state=[0]

# creating a dictionary of hyperparameters
hyperparameters = dict(C=C, penalty=penalty, random_state=random_state)
```

Now define the Stochastic Gradient Descent

```
clf = GridSearchCV(estimator = logistic_model, param_grid = hyperparameters, cv=5)
best_model = clf.fit(x_train, y_train)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
best_predicted_values = best_model.predict(x_test)
print(best_predicted_values)

accuracy_score(best_predicted_values, y_test.values)
```

## Model evaluation
```
cfm_train = confusion_matrix(y_train,y_train_pred)
cfm_test = confusion_matrix(y_test,y_test_pred)
print(cfm_train)
print(cfm_test)
print(classification_report(y_train,y_train_pred))
print(classification_report(y_test,y_test_pred))
```

**[⬆ back to top](#contents)**

## References

[frogs] Hunter, D. (2000) The conservation and demography of the southern corroboree frog (Pseudophryne corroboree). M.Sc. thesis, University of Canberra, Canberra.
[ref] https://github.com/animesh-agarwal/Machine-Learning-Datasets/blob/master/census-data/census%20income%20logistic%20regression.ipynb
### License
Copyright (c) 2020 Saeid Amiri
