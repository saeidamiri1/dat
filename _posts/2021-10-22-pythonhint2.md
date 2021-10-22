---
layout: post
title: How to save all objects
description: How to save all objects
date: 2021-10-22
author: Saeid Amiri
published: true
tags: Python_learn
categories:  Python_learn
comments: false
---

# How to save all objects
```
# to save session
dill.dump_session('backup_2021_10_22.db')
# to load 
backup_restore = dill.load_session('backup_2021_10_22.db')
```




