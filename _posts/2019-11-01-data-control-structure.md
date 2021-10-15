---
layout: post
title: Data control structure
description: Data control structure
date: 2019-11-01
author: Saeid Amiri
published: true
tags: True False Boolean-value ~ f-string Try-except
categories: Data Structures Python_learn
comments: false
---


## Data control structure


## Contents
- [Boolean value](#boolean-value)
- [Control Structure](#control-structure)
- [Try Except](#try-except)

### Boolean value
The value True (T) and False (F) are referred to as logical values and used the same in Python; their corresponding values are 1 and 0. Run the following codes and explains what the codes do.


```
>>> 8<9
True
>>> 9<8
False
>>> x=3
>>> y=9
>>> x<y
True
>>> x>y
False
>>>
>>> X=range(-3,3)
>>> [X[i]<2 for i in range(6)]
[True, True, True, True, True, False]
>>> sum([X[i]<2 for i in range(6)])
5
>>> sum(X)
-3
```

One of the main application of logical operator is to extract specific elements, see the following codes,

```
>>> weight=[58,89,68,74,62,77,65,65]
>>> [weight[i]<74 for i in range(len(weight))]
[True, False, True, False, True, False, True, True]
>>> weight<74
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: '<' not supported between instances of 'list' and 'int'
```

Obviously ``weight<74`` does not work for the list, To run it change the data to the [array](https://saeidamiri1.github.io/pythonseum/numpy/2019/10/22/data-structure) provided in Numpy:  

```
>>> weight=np.array(weight)
>>> weight<74
array([ True, False,  True, False,  True, False,  True,True], dtype=bool)
>>> (weight<74) & (weight==89)
array([False, False, False, False, False, False, False, False], dtype=bool)
>>> weight[(weight<74) & (weight==89)]
array([], dtype=int64)
>>> weight[(weight<74) & (weight==62)]
array([62])
>>> weight[(weight<74) | (weight==62)]
array([58, 68, 62, 65, 65])
>>> weight[~(weight<74) & (weight==62)]
array([], dtype=int64)
>>> weight[~((weight<74) | (weight==62))]
array([89, 74, 77])
```

### Control Structure
Commands with control structure often include conditional command that use comparisons operators (>, <, =>, <=, ==, !=, ~, is)

```
>>> 3<4
True
>>> 3!=4
True
>>> 3==4
False
>>> 3 is 4
False
>>> 'hi' == 'h' + 'i'
True
>>> 'HI' != 'hi'
True
>>> [1, 2] != [2, 1]
True
>>> ~True
-2
>>> ~False
-1

```

The structure command of if is as below.
```
If cond satisfies the cons.expr run otherwise alt.expr run.
if(cond) cons.expr elif (condition) alt.expr else alt.expr
```

```
x=4
y=4

if x<y: 
  print('x is less than y')
elif x>y:
 print('x greater than y')
else: 
 print(' x and y are equal')
```

To pass the value inside the quote, use the f-string format 

```
if x<y: 
  print( f'{x} is less than {y}')
elif x>y:
 print(f'{x}greater than {y}')
else: 
 print(f'x={x} and y={y} are equal')
```

### Try except
When there is any possibility for error, it is better to use `try except`, which tests the statement infront try, if there is an error, it goes to except, otherwise passes and goes to else.

```{Python, echo = FALSE, message = FALSE}  
x='Just test'
try:
  print(x)
except:
  print("Something went wrong")
else:
  print("Nothing went wrong")

try:
  print(y)
except:
  print("Something went wrong")
else:
  print("Nothing went wrong")
```

**[⬆ back to top](#contents)**
### License
Copyright (c) 2019 Saeid Amiri