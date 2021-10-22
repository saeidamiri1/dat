---
layout: post
title: How run apply on array
description: How run apply on array
date: 2021-10-22
author: Saeid Amiri
published: true
tags: Python Python_learn
categories:  Python_learn
comments: false
---

# How run apply on array
```
import numpy as np
x = np.array([[5,2,1,3], [2,1,5]])
fun = lambda t: np.argmax(t)
np.array([fun(xi) for xi in x])
```




