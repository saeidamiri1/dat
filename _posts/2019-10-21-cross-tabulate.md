---
layout: post
title: Crosstab
description: Generate cross tabulates
date: 2019-10-21
author: Saeid Amiri
published: true
tags: Pandas columns DataFrame Crosstab quantile
categories: Python_learn Pandas
comments: false
---
## Crosstab
To show how generate the cross tabulate, let us categorize the columns; consider two continuous variables ( e.g., housing_median_age and total_rooms), categorize them according their .3 and .7 quantiles, and label the elements as L, M, and H. Then find the cross tabulate of them,

```
CHT['houlev'] = ''
C1=CHT.housing_median_age<=CHT.median_income.quantile(.3)
C2=CHT.housing_median_age>=CHT.median_income.quantile(.7)
CHT.loc[C1,'houlev']='L'
CHT.loc[~C1&~C2,'houlev']='M'
CHT.loc[C2,'houlev']='H'

CHT['roomlev'] = ''
C1=CHT.total_rooms<=CHT.total_rooms.quantile(.3)
C2=CHT.total_rooms>=CHT.total_rooms.quantile(.7)
CHT.loc[C1,'roomlev']='L'
CHT.loc[~C1&~C2,'roomlev']='M'
CHT.loc[C2,'roomlev']='H'

pd.crosstab(CHT.roomlev, CHT.houlev, margins=True)
```

You can add more than two columns.