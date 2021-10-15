---
layout: post
title: Manipulating data-frame
description: Manipulating data-frame
date: 2019-10-21
author: Saeid Amiri
published: true
tags: Pandas columns DataFrame iloc loc np.where isin drop replace
categories: Python_learn Pandas
comments: false
---
## Manipulating data-frame
To select the data use the name of variable, or specify the indices via `.iloc` and `.loc` (link)[http://pandas.pydata.org/pandas-docs/version/0.22/indexing.html].  `.iloc` is an integer-based select and should be used with integer indies. On contrary, `.loc`   is primarily label based, and may also be used with a boolean array.


```
source ="https://storage.googleapis.com/mledu-datasets/california_housing_train.csv"
CHT = pd.read_csv(source, sep=",")
CHT.head()
CHT.longitude
CHT['longitude']
CHT.iloc[:,1]
CHT.iloc[:,[1,3]]
```

To select part of row, you can also use iloc[idenx of row,:], also rows can be selected using the logical values

```
CHT.iloc[2:10]
CHT.iloc[2:10,:]
CHT[CHT.iloc[:,1]<34]
```

To retieve part of row, should pass boolean variable, ```.iloc``` does not work with the boolean variable, and ```.loc``` should be used.  Consider the median_income in our data, by using quartile divide it into three categories.
```
CHT['famlev'] = ''
C1=CHT.median_income<=CHT.median_income.quantile(.3)
C2=CHT.median_income>=CHT.median_income.quantile(.7)
CHT.loc[C1,'famlev']='L'
CHT.loc[~C1&~C2,'famlev']='M'
CHT.loc[C2,'famlev']='H'
```

In this case we used `.loc`, obviously we specify column labels to retrieve columns instead of by position. Note: You can also using [][] apply different conditions on data.  

```
CHT['median_house_value'][CHT['famlev'] == 'M'].mean()
```

Selecting or searching can also be done using ```np.where```,  which evaluates the conditions and return the data that satisfy the conditions.
```
CHT_R=CHT[['total_rooms','total_bedrooms']]
CHT_R.where(CHT.total_rooms<1000)
CHT_R.where(CHT.total_rooms<1000,0)
con= CHT_R<1000
CHT_R.where(con, -999)
```

If you want to select specific elements in data-frame, use `.isin()`,  the following select element where 'famlev=M',
```np.where(CHT.loc[:,'famlev'].isin(['M']))```

The ```np.where``` can be used to create a new column:

```
CHT['size']=np.where(CHT.total_rooms<1000, 'small', 'big')
CHT_R=CHT[['total_rooms','total_bedrooms']]
CHT_R.where(CHT.total_rooms<1000)
CHT_R.where(CHT.total_rooms<1000,0)
con= CHT_R<1000
CHT_R.where(con, -999)
```
Opposite of `np.where` is `np.mask`, replace it with `np.where` and rerun the codes. 
To drop row and columns use ```.drop```.
```
CHT.drop([0,5], axis=0)
CHT.drop('longitude',axis=1, inplace=True)
```

To replace values, use `df.replace()`

```
CHT['famlev'].replace('L','Low').replace('M','Middle').replace('H','High')
CHT.drop('longitude',axis=1, inplace=True)
```

Note: the argument ```inplace=True``` apply the change on the original data.

Simple operation using the list comprehension can be done on data-frame as well.
```
CHT['NN']=[0 for x in CHT['total_rooms'] if x<100]
CHT['size']=['small' if x<100  else 'big'  for x in CHT['total_rooms']]
