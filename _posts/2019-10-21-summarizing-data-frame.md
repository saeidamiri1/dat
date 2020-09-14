---
layout: post
title: Summarizing data-frame
description: Summarizing data-frame
date: 2019-10-21
author: Saeid Amiri
published: true
tags: Pandas columns DataFrame iloc loc head  describe groupby reset_index pivot_table
categories: Python_learn Pandas
comments: false
---
## Summarizing data-frame
To see the type, the information and summary of variables in the data-frame, use ```.dtypes```,  ```.describe()```, and   ```.info()```. 

```
source ="https://storage.googleapis.com/mledu-datasets/california_housing_train.csv"
CHT = pd.read_csv(source, sep=",")
# show the type of variables
CHT.dtypes
# generate summary
CHT.describe()
CHT.info()
```


Beside the function print, pandas can show the first and the last part of data, using ```.head()``` and `.tail()`. By passing a number in the parenthesis, one can specify the output.

```
CHT.head(10)
CHT.tail(10)
CHT.sort_values(by='housing_median_age', ascending=False).head(3)
CHT.columns
```

It is easy to find the duplicates in data-frame  and  drop them, see below.
```
CHT.duplicated()
CHT.drop_duplicates()
```
To check the duplication in variables, specify their names as well, 
```
CHT.duplicated(['longitude'])
CHT.drop_duplicates(['longitude'], keep='last')
CHT.index.duplicated()
```

Although `.describe` can give a summary of variables,  more specific summery of variables (columns) can be extracted, see below.

```
CHT.count()
CHT[CHT.iloc[:,1]<34].nunique()
```

The following table includes the useful functions.

|Function|Description|
|---|---|
`count`| Number of non-null observations
`sum` | Sum of values
`mean` | Mean of value
`mad` | Mean absolute deviation
`median` | median of values
`min` |Minimum
`max` |Maximum
`mode` |Mode
`abs` | Absolute Value
`prod` | Product of values
`std` | Unbiased standard deviation
`var` | Unbiased variance
`sem` | Unbiased standard error of the mean
`skew` | Unbiased skewness (3rd moment)
`kurt` | Unbiased kurtosis (4th moment)
`quantile` | Sample quantile (value at %)
`cumsum` | Cumulative sum
`cumprod`| Cumulative product
`cummax` | Cumulative maximum
`cummin` | Cumulative minimum
`nunique`| number of unique elements
`value_counts`| Counts of unique values
`cov`| Calculate the covariance between columns
`corr`| Calculate the correlation between columns

The summaries can be obtained using any grouping variables in the data set:

```
CHT.groupby(['famlev']).groups.keys()
CHT.groupby(['famlev']).groups['H']
CHT.groupby(['famlev']).first()

CHT.groupby(['famlev']).sum()

CHT.groupby(['famlev'])['median_house_value'].sum()
# better output
CHT.groupby(['famlev'])[['median_house_value']].sum()
```

The grouped variables would be assigned as indices, to bring them back as variables use `df.reset_index()`
```
CHT.reset_index()
```

It is possible to apply even complex function, the following scripts calculate the coefficient of data.

```
def cv(x):
 return (np.mean(x)/np.var(x))

aggr = {
    'total_rooms':'sum',
    'population': lambda x: cv(x)
}
CHT.groupby('famlev').agg(aggr)
```

The output can be tidied up,

```
aggr = {
    'total_rooms':['mean','std']
}
grouped = CHT.groupby('famlev').agg(aggr)
grouped.columns = grouped.columns.droplevel(level=0)
grouped.rename(columns={"mean": "total_rooms", "std": "total_rooms"})
grouped.head()
```

The summarizations can be done using pivot table, 
```
pd.pivot_table(CHT, index=['famlev'], aggfunc=['mean'])
```