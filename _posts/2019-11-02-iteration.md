---
layout: post
title: Iteration
description: Iteration
date: 2019-11-02
author: Saeid Amiri
published: true
tags: Range For While Comprehension-structure 
categories: Python_learn
comments: false
---


## Iteration
Python is equipped with strong tools for the repeat of some commands or produce sequence number.  

## Contents
- [Range](#range)
- [For](#for)
- [While](#while)
- [Comprehension structure](#comprehension-structure)

### Range
The function `range` can be used to produce series of number between two numbers.
```
range(3,15)
```

The advance function of `range` is `arange` in `numpy` which can be used to generate a series of number from a number to another number with specific increment:

```
>>> np.arange(8, 20,1)
array([ 8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
>>> np.arange(2,1,-0.1)
array([ 2. ,  1.9,  1.8,  1.7,  1.6,  1.5,  1.4,  1.3,  1.2,  1.1])
```

A specified element can be repeated for specific number.

```
>>> [2,3]*2
[2, 3, 2, 3]
>>> np.repeat([2, 3],[2,3])
array([2, 2, 3, 3, 3])
>>> np.repeat(["A", "B"],[2,3])
array(['A', 'A', 'B', 'B', 'B'],dtype='<U1')
```

### For
The most useful function for the iteration is `for`  that repeats the specified commands for specified times, run the following codes to see how does it work

```
>>> for r in range(1,5):
...  print(r^3)
...
2
1
0
7

 >>> for i in [2,3,1,7]:
...  print(i^3)
...
1
0
2
4
```  

```
>>> score=[10, 15, 7, 20]
>>> for i in (range(0,4)):
...  if (score[i]<10):
...       print("fail")
...  else:
...        print("pass")
...
pass
pass
fail
pass
```

```
>>> for i in range(0,4):
...   if score[i]<10:
...       print("fail")
...   elif(score[i]>=10&score[i]<14):
...       print("middle")
...   elif(score[i]>=14&score[i]<17):
...        print("good")
...   elif(score[i]>=17):
...       print("BEST")
...
middle
middle
fail
middle
```

### While
The command `while` runs iteration until the condition be attained,

```
>>> x=8
>>> i=0
>>> while(x<12):
...   i=i+1
...   x=x+x/8
...   print(i,x)
...
1 9.0
2 10.125
3 11.390625
4 12.814453125
```

Inversely, the command of repeat continue until the condition situated inside commands be attained, in the following codes, the loop continues until the condition ```(x>12)```  is violated.


```    
>>> x=8
>>> i=0
>>> while True:
...  i=i+1
...  x=x+x/8
...  print(i,x)
...  if (x>12):
...    break
...
1 9.0
2 10.125
3 11.390625
4 12.814453125
```

### Comprehension structure
Comprehension structure in Python helps to combine several iteration in one line, to practice let write a simple function to select pair of unequal numbers between (1, 100).

```
combs=[]
for x in range(3):
 for y in range(3):
  if x!= y:
    combs.append((x,y))
```

The code can be simplify as list comprehension
```
[(x,y) for x in range(3) for y in range(3) if x!=y]
```

Comprehension structure can be used for different Python structures, see the following script that generates numbers between (1,10) and put them in different Python structures.  

```
# A generator expression

print ((x for x in range(10)))

# A list comprehension
print ([x for x in range(10)])

# A set comprehension
print ({x for x in range(10)})

# A dictionary comprehension
print ({x: x for x in range(10)})
```


**[⬆ back to top](#contents)**
### License
Copyright (c) 2019 Saeid Amiri