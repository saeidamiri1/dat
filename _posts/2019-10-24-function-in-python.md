---
layout: post
title: Function in Python
description: Function in Python
date: 2019-10-24
author: Saeid Amiri
published: true
tags: Numpy List Tuple Dictionary Arrays Class set
categories: Python_learn
comments: false
---


## Function
In the context of programming, a function is a sequence of statements that performs a computation. Functions has three parts; argument, script, and output. Python has two kinds of function: built-in function that is in the core of Python or are collected as package. User-defined function that is written by user.


## Contents
- [Built-in function](#built-in-function)
- [User function](#user-function)
- [In-line function](#in-line-function)
- [Map and Filter](#map-and-filter)
- [Decorators](#decorators)


### Built-in function
Python has a number of functions in its core that are always available, [see](https://docs.python.org/3/library/functions.html)

```
x=[1,2,3]
type(x)
len(x)
min(x)
```
To round the value, use the `round(value,size)` function
```
round(0.12345,2)
round(0.12345,3)
```

### User function
Functions has three parts; argument, script, and output. It has simple structure

```
def name (argument):  
  script
  return output
```

For instance write a function get two argument, add them together and return it.

```
def sum0 (x,y):  
  s0=x+y
  return s0
```

If you do not specify the arguments, use a * argument,
```
def sum0 (x,*y):  
  s0=x+mean(y)
  return s0
```

You can define a default value for argument.  
```
def sum0 (x,y=1):  
  s0=x+y
  return s0
```

You can define an optional argument.  
```
def sum0 (x,y=None):  
  if y is None:
    return x
  elif:
     return x+y
```

```
def letterGrade(score):
    if score >= 90:
        letter = 'A'
    elif score >= 80:
        letter = 'B'
    elif score >= 70:
        letter = 'C'
    elif score >= 60:
        letter = 'D'
    else:
        letter = 'F'
    return letter
```    
### In-line function
A simple function can be written in one line,
```
sum0 = lambda x, y: x + y
sum0(2,3)
```

Such function is more suitable for using inside the other operation, the follow get first and second name, then it sort according the last name. 
 ```
names = ['Sam Amiri', 'Leila Alimehr','Ryan Amiri']
sorted(names, key=lambda name: name.split()[-1].lower())
>>> sorted(names, key=lambda name: name.split()[-1].lower())
['Leila Alimehr', 'Sam Amiri', 'Ryan Amiri']
>>> sorted(names)
['Leila Alimehr', 'Ryan Amiri', 'Sam Amiri']
```


### Map and Filter
Python access to a higher order function  which allows a function operates on other functions, either by taking a function as its argument, or by returning a function. Most popular ones are ```map``` (apply function on element) and ```filter ``` (apply function, if it is true return element)
 ```
x=[-1,0,1]
list(map(abs, x))
list(filter(lambda x: x <= 0,x))
 ```

Example: Write a function to divide two number, if the denominator is zero, stop the function and give an notification. 
```
def divide(x, y):
  try:
    x / y
  except: 
   print('Can not divide by zero!')
  else:
   return x / y

divide(3,1)
divide(3,0)
```

The function is also can be rewritten using ```raise```, which raise an error and stop the function. 

def divide(x, y):
    """Divide Function"""
    if y == 0:
        raise Exception('Can not divide by zero!')
    return x / y
  

### Decorators
Decoreators in Python allows you to take a function and add additional uses without modifying its structure, the following example is from [ref](https://realpython.com/primer-on-python-decorators/#functions)
```{Python, echo = FALSE, message = FALSE}
def my_decorator(func):
   def wrapper():
       print("Something is happening before the function is called.")
       func()
       print("Something is happening after the function is called.")
   return wrapper

def say_whee():
   print("Whee!")

say_whee()

say_whee = my_decorator(say_whee)
say_whee()   
```

The decorator often simplify using
``@name of decorator``
```
def say_whee():
   print("Whee!")
```

**[⬆ back to top](#contents)**
### License
Copyright (c) 2019 Saeid Amiri