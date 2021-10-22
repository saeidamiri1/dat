---
layout: post
title: How generate variable name using loop item
description: How generate variable name using loop item
date: 2021-10-22
author: Saeid Amiri
published: true
tags: Python_learn
categories:  Python_learn
comments: false
---

# How generate variable name using loop item 
If you need to create the variable name using the loop object, use `exec`:
````
for i in range(4):
    exec(f'var_{i} = [range(i)]')
````

I personally prefer using the dictionary object instead the variable. 
````
var={}
for i in range(4):
    var[f'var_{i}']=range(i)
````







