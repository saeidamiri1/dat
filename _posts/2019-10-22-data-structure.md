---
layout: post
title: Data Structure
description: Data Structure
date: 2019-10-22
author: Saeid Amiri
published: true
tags: Numpy List Tuple Dictionary Arrays Class set
categories: Python_learn Data Structures
comments: false
---

## Data Structures
Python provides a variety of useful data structures, such as lists, sets, and dictionaries, and a new structure define by programmer which called class.

### list
A list is a sequence of values that is assigned to the variable. The values in a list are called elements or sometimes items. The value of list can be accessed using the square brackets.

```
weights=[20,15,19,21,16] 
type(weights)
colors=['red','blue','green','black','white']
colors
```

Use the square brackets ([]) to index it.

```
colors[1:3]
colors[:3]
colors[3:]
colors[-1]
colors[1]
colors[:-1]
colors[::-1]
```

The following scripts show how reverse, adds new object, and sort the elements.

```
colors.reverse()
colors

colors.extend('blue')
colors

colors.extend(['blue'])
colors

sorted(colors)
colors.sort()
colors.sort(key=len)
colors
```

Finding the index of elements and counting them are vey simple, see below.  

```
colors.index('blue')
colors.count('blue')
```

To change the element of list use `replace`:

```
a = 'hello, world!'
a[2]='z'
a.replace('l', 'z', 1)
a.replace('l', 'z')
```

### Tuple
Tuple is a sequence of objects like list, but it is immutable. To define tuple, Python uses the parenthesis:

```
>>> colors=('red','blue','green','black','white')
>>> type(colors)
<class 'tuple'>
>>> colors[1:3]
('blue', 'green')
```

To access the content in a tuple use brackets like list. Obviously the list and tuple look the same, the objects of tuple is immutable, i.e., when it is created, it can not modified.  

```
>>> colors[1]
'blue'
>>> colors[1]='yellow'
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
```
A brief comparison of mutable and immutable and application can be found [here](https://www.afternerd.com/blog/difference-between-list-tuple/).


### Set
Set is a collection of elements without order and index, the same as defined in Algebra. Duplication of elements in set does not make sense, so Python drops the duplication automatically. Use the curly bracket ({) to create the set:
```
colors={'red','blue','green','black','white'}
type(colors)
```

A new object can be added using the `add`, to add more than one objects, use `update`. To drop elements use the `remove` or `discard`, both remove elements but the  `discard` does not raise any error notification, if the element does not exist in set.
``` 
colors.add('pink')
colors.update(['purple','orange'])
colors.remove('pink')
colors.remove('pink')
colors.discard'(pink')
```

To delete elements and remove the set completely, use `clear` and `del`, respectively. To see whether the element exists in set, use `in`. The functions `len` and `sorted` can be applied on set:  
```
'blue' in colors
len(colors)
sorted(colors)
```

The functions `len`, `sorted`, `intersection`, `union`, and `difference` can be applied on sets:  
```
'blue' in colors
len(colors)
sorted(colors)
{'red','blue'}.intersection({'red','white'})
{'red','blue'}.union({'red','white'})
{'red','blue'}.difference({'red','white'})
```

### Dictionary
Dictionary is a generalized form of list, unlike the list its indices can be any type of values. A dictionary maps a set of objects (keys) to another set of objects (values).

Dictionary includes key and items, the key is actually indices and item is the values. A Python dictionary is a mapping of unique keys to values. Use the curly brackets to construct the dictionary, separate the key and value with colons (:) and with commas (,) between each pair. Keys must be quoted. We can print out the dictionary by printing the reference to it.

```
prices = {
   'BMW': 50,
   'BENZ': 55,
   'Ford': 25,
   'Chevy': 30,  
   'GM': 28
}

prices.values()
prices.keys()
prices.items()
```

The dictionary is very simple to manipulate,  
```
student={'A':10, 'B':20, 'AB':100 }
student.values()
student.keys()
student['C']=45
student[45]=34
del student['C']
```

Note, dictionaries are unordered, so the order that the keys are added doesn't necessarily reflect what order they may be reported back.

```
dic1={
'x1' : 1,
'x2' : 2,
'x3' : 3 }

dic2={
'y1' : 10,
'x1' : 11,
'x2' : 2 }

dic1.keys() & dic2.keys()
dic1.keys() - dic2.keys()
dic1.items() & dic2.items()
```

Example: Create a list which its elements are dictionary. 

```
team = [
    {
        'name': 'Saeid',
        'city': 'Torronto',
    },
    {
        'name': 'Leila',
        'city': 'Torronto',
    },
    {
        'name': 'Ryan',
        'city': 'Montreal',
    },
```

### Array
Numpy's array is a generalization of list, it is more appropriate for the computation.  

```
import numpy as np
weight=[58,89,68,74,62,77,65,65]
weight
weight_arr=np.array(weight)
weight_arr
```

All elements should have the same type, therefore if you add a strict element, it saves all element as strict. Array also accepts multi lists.
```
arr1=np.array([range(i) for i in [1, 2, 3]])
arr1[1]
arr1[1][0]
arr2=np.array([range(i, j+i) for i in [1, 2, 3] for j in [1, 2, 3]])
arr2[1]
arr2[1][0]
```

To generate a constant array, use ```np.full(,) ```
```
np.full(2, 2.2)
np.full((2,1), 2.2)
np.full((2,2), 2.2)
```

```
np.arange(1,14,4)
np.arange(21,30,3)
np.arange(2,1,-0.1)
```

To create an array of k values between two values
```
np.linspace(0, 1, 10)
```

To refer elements of array should use ```[]```
```
weight[1]# first element
weight[2:]# second elements to the rest
weight[:3]# elements before the third and including the third
```

To refer elements of multi array should use ```[,]```
```
weight2=np.array([weight_arr,2.20*weight_arr,35.27*weight_arr])
weight2[1,1]
weight2[1:,1:]
weight2[1:,2:]
```

To change the shape, use `reshape`:
```
weight2.reshape((8, 3))
```


### Class
Python is an object-oriented programming language, and has strong tools working with different objects. If the structure is not defined, one can create own object; the class can be used to present new structure for your data or change the existing one. It is very useful to tie a certain data and functions together.

```
class body(object):
    """Define class"""
    """ we can use weigt and height"""
    def __init__(bd, weight,height):
     bd.weight=weight
     bd.height=height
def bmi_body(b):
    print (' BMI is (%g)' %((b.weight/b.height)))

SAM = body(90,79)
SAM.weight
SAM.height
print (SAM)
bmi_body(SAM)
```    

Or it can be simplified to

```
class body(object):
    """Define class"""
    """ we can use weight and height"""
def bmi_body(b):
    print (' BMI is (%g)' %((b.weight/b.height)))

SAM = body()
SAM.weight = 90
SAM.height = 79
print (SAM)
bmi_body(SAM)
```    

Example: American present the data as
Month/Day/Year. Write a function represent it with Canadian style.


```
class dateg(object):
    """Define class"""
    """Month, Day, Year """
def ca_date(d):
    print (' Canadian date is (%g/%g/%g)' %((d.day,d.month,d.year)))

mybirt = dateg()
mybirt.day = 11
mybirt.month = 11
mybirt.year = 11
print (mybirt)
ca_date(mybirt)
```