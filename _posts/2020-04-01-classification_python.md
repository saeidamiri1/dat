---
layout: post
title: Different classification techniques
description: Different classification techniques
date: 2020-05-25
author: Saeid Amiri
published: true
tags: Classification Python
categories:  Python_learn Deep-learning   Machine_Learning 
comments: false
---

# Classification 
In this short, we show how run several classification methods;  XGBoost, Ada Boost, Discriminant Analysis, KNN, random forest, decision theory, Gaussian Process, 
Logistics regression, Gaussian Mixture Classification, SVM, and LSTM. 

## Contents
- [Dependencies](#dependencies)
- [Data](#data)
- [XGBoost](#xgboost)
- [Ada Boost](#ada-boost)
- [Discriminant Analysis](#discriminant-dnalysis)
- [KNN](#KNN)
- [Naive Bayes](#naive-bayes)
- [Random Forest](#random-forest)
- [Decision Tree](#decision-tree)
- [Gaussian Process](#gaussian-process)
- [Logistic regression](#logistic-regression)
- [Gaussian Mixture Classification](#gaussian-mixture-classification)
- [SVM](#SVM)
- [Neural regression](#neural-regression)
- [LSTM](#lstm)

### Dependencies pachages

```{python}
import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from keras.models import Sequential
from keras.layers import Dense
from xgboost import XGBClassifier
```

### Data 
We considered the DNA Microarray Data of Lung Tumors, as presented and analyzed in Garber et al. (2001). The data was obtained from the pvclust library; this dataset includes n =73 case with p =916 variables, and there are six classes, M = 6. 

```{python}
data_cancer=pd.read_csv('https://raw.githubusercontent.com/saeidamiri1/saeidamiri1.github.io/master/public/general_data/class_lung_cancer/lung_cancer.csv',sep=';')
data_cancer.head(5)
data_cancer.isnull().values.any()
data_cancer=data_cancer.fillna(data_cancer.mean())
data_cancer.isnull().values.any()

data_cancer['cancer_class'].value_counts()
data_cancer.dtypes
X = data_cancer.drop(['cancer_class'],1)
X = np.array(X)
y = np.array(data_cancer['cancer_class'])
scale = MinMaxScaler(feature_range=(0,1))
X = scale.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
np.unique(y_test, return_counts=True)
n_classes = len(np.unique(y_train))
np.unique(y_train, return_counts=True)
```



### Ada Boost
Adaptive Boosting is adaptive  in the sense that  fit classification and  use the missclassification to improve the classification.
```{python}
adab_model =  AdaBoostClassifier()
adab_model.fit(x_train,y_train)
y_test_pred_adab = adab_model.predict(x_test)
print( classification_report(y_test,y_test_pred_adab) )
```

### XGBoost 
Here we present the classification using XGBoost, which stands for Extreme Gradient Boosting and is based on decision trees. 
```{python}
xgb_model = XGBClassifier()
xgb_model.fit(x_train, aa)
y_test_pred_xgb = xgb_model.predict(x_test)
print( classification_report(y_test,y_test_pred_xgb) )
```

### Discriminant Analysis
The idea of discrimination analysis is to find a linear combination of explanatory variables  that can characterize the classes given in dependent variables, it can be don 
using the linear or quadratic form. 
```{python}
lda_model =  LinearDiscriminantAnalysis()
lda_model.fit(x_train,y_train)
y_test_pred_lda = lda_model.predict(x_test)
print( classification_report(y_test,y_test_pred_lda))

qda_model =  QuadraticDiscriminantAnalysis()
qda_model.fit(x_train,y_train)
y_test_pred_qda = qda_model.predict(x_test)
print( classification_report(y_test,y_test_pred_qda))
```

### KNN
K Nearest Neighbor is a well-known classification which consider the K closed observations to predict the class. 
```{python}
knn_model =  KNeighborsClassifier(3)
knn_model.fit(x_train,y_train)
y_test_pred_knn = knn_model.predict(x_test)
print( classification_report(y_test,y_test_pred_knn))
```


### Naive Bayes
The naive Bayes is pure bayesian classification which model using a conditional probability, 
```{python}
nb_model =  GaussianNB()
nb_model.fit(x_train,y_train)
y_test_pred_np = nb_model.predict(x_test)
print( classification_report(y_test,y_test_pred_np))
```
 
### Random Forest
The random forests, which is an extension of bagging; it adds an additional layer of randomness to bagging and this idea brings more randomness on decision tree predictors in order to obtain more diverse classifiers, see [Breiman (2001)]. 
```{python}
rf_model =  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
rf_model.fit(x_train,y_train)
y_test_pred_rf = rf_model.predict(x_test)
print( classification_report(y_test,y_test_pred_rf))
```

### Decision Tree
Decision tree is a recursive partitioning to find the final classification, 
```{python}
dt_model = DecisionTreeClassifier(max_depth=10)
dt_model.fit(x_train,y_train)
y_test_pred_dt = dt_model.predict(x_test)
print( classification_report(y_test,y_test_pred_dt))
```

### Gaussian Process
The Gaussian process generates many samples given holding the Gaussian distribution, then use average (expected value) to find the final classification. 
```{python}
gp_model = GaussianProcessClassifier(1.0 * RBF(1.0))
gp_model.fit(x_train,y_train)
y_test_pred_gp = gp_model.predict(x_test)
print( classification_report(y_test,y_test_pred_gp))
```


### Logistic regression 
The logistic regression is a model-based classification. 
```{python}
lgr_model = LogisticRegression(C=10, penalty='l2', solver='liblinear')
lgr_model.fit(x_train,y_train)
y_test_pred_lgr = lgr_model.predict(x_test)
print( classification_report(y_test,y_test_pred_lgr))
```

### Gaussian Mixture Classification 
Gaussian Mixture Model proposes an overall model for the observed data; it assumes a distribution and the underlying distribution of data is a mixture distribution, such formulation allow to define the membership of observations which can be done by incorporating a measure of probability of to the class.

```{python}
y_test_s=np.zeros(len(y_test))
y_train_s=np.zeros(len(y_train))
for i, x in enumerate(np.unique(y)):
    y_test_s[y_test==x]=i
    y_train_s[y_train==x]=i

gmm_model = GaussianMixture(n_components=n_classes,random_state=6)
gmm_model.fit(x_train,y_train_s)
y_test_pred_gmm = gmm_model.predict(x_test)
y_test_pred_gmm=np.unique(y)[y_test_pred_gmm]
print(classification_report(y_test,y_test_pred_gmm))
```

### SVM
Support-vector machines (SVM) is a machine learning method that perform a non-linear classification using the kernel function.
```{python}
svm_model = LinearSVC()
svm_model.fit(x_train, y_train)
y_test_pred_svm = svm_model.predict(x_test)
print( classification_report(y_test,y_test_pred_svm))
```

### Artificial Neural network
The Artificial Neural network considers the hidden-layer neural networks to generate the information in data.

```{python}
nl_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1)
nl_model.fit(x_train, y_train)
y_test_pred_nl = nl_model.predict(x_test)
print( classification_report(y_test,y_test_pred_nl))
```

### LSTM
Long Short Term Memory networks (LSTM) is a deep learning method that is capable of learning long-term dependencies.
```{python}
y_train_d = pd.get_dummies(y_train)
lstm_model = Sequential()
lstm_model.add(Dense(1000, input_dim=916, activation='relu'))
lstm_model.add(Dense(916, activation='relu'))
lstm_model.add(Dense(6, activation='softmax'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(x_train, y_train_d, epochs=1000, batch_size=100)
y_pred = lstm_model.predict(x_test)
y_test_pred_lstm = np.round(y_pred).astype(int)
y_test_pred_lstm=y_train_d.columns[np.array([np.argmax(item) for item in y_test_pred_lstm])]
print(classification_report(y_test,y_test_pred_lstm))
```


**[⬆ back to top](#contents)**

### References
- [1] Garber, M. E., Troyanskaya, O. G., Schluens, K., Petersen, S., Thaesler, Z., Pacyna-Gengelbach, M., ... & Altman, R. B. (2001). Diversity of gene expres- sion in adenocarcinoma of the lung. Proceedings of the National Academy of Sciences, 98(24), 13784-13789.

### License
Copyright (c) 2020 Saeid Amiri

**[⬆ back to top](#contents)**




