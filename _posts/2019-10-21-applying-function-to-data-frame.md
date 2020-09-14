---
layout: post
title: Applying function on data-frame
description: How to apply a function on row or column
author: Saeid Amiri
published: true
tags: Pandas 
categories: Python_learn Pandas
comments: false
---
## Applying a function on row or column
Using ```df.apply(fun)``` can apply a function on columns or row:
```
df.apply(np.sum, axis=0)
df.apply(np.sum, axis=1)
```

Even can write a new function and run on columns or rows
```
def prod(col):
    return col['A'] * col['B']

df.apply(prod, axis=1)
df['productcolmn']=df.apply(prod, axis=1)
```