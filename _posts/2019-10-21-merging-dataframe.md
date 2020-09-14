---
layout: post
title: Merging data-frame 
description: Merging data-frame
date: 2019-10-21
author: Saeid Amiri
published: true
tags: Pandas columns concat DataFrame iloc loc merge 
categories: Python_learn Pandas
comments: false
---

## Merging data-frames
Panada is very useful for merging dataset; to merging data consider the following data sets, where 'id1' and 'id2' include the ids of data.

```
raw_data = {'id1': range(4),'income': [10,12,14,16]}
dat1 =pd.DataFrame(raw_data, columns = ['id1', 'income'])

raw_data = {'id2': range(6),'pay': [9,11,13,15,17,19]}
dat2 =pd.DataFrame(raw_data, columns = ['id2', 'pay'])
```

Obviously the id variables are not the same, they can be
compared using

```
dat1['id1'].isin(dat2['id2']).value_counts()
dat2['id2'].isin(dat1['id1']).value_counts()
```

`pd.merge` can merge different data-frames, the merging can be done based on the identities of left dataset, if there is no match in the right file, Python adds `NaN`.

```
result = pd.merge(dat1, dat2, left_on='id1', right_on='id2',how='left')
```
On contrary, one can the right dataset as matching,

```
result = pd.merge(dat1, dat2, left_on='id1', right_on='id2',how='right')
```

Since the ids are not the same, one can do merging based on the
intersection of the ids,

```
result = pd.merge(dat1, dat2, left_on='id1', right_on='id2',how='inner')
```

Merging can also be done based on the union of the ids,

```
result = pd.merge(dat1, dat2, left_on='id1', right_on='id2',how='outer')
```

Note: If the names of id variables are the same in the both datasets, you can use ```on=id_name``` instead of ```left_on=``` and ```right_on=```.

Note: if you want to identify where the elements in rows are from, add  argument ```indicator=True```, then new column named `_merge` would be added to the merged data which shows its originate.

```
result = pd.merge(dat1, dat2, left_on='id1', right_on='id2',how='outer', indicator=True)
```

To combine datasets row-wisely, use `concat`,
```
result = pd.concat([dat1, dat2],axis=1)
```