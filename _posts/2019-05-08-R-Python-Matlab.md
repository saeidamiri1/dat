---
layout: post
title: Cheatsheet for R, Python, and Matlab 
description: A brief cheatsheet for R, Python, and Matlab
date: 2019-05-08
author: Saeid Amiri
published: true
tags: R Python Matlab Cheat-sheet
categories: Cheat-sheet R_learn Python_learn
comments: false
---

#  R-Python-Matlab (Basics)
A brief cheatsheet for those who use R daily, and use other script programing languages, Python and Matlab, occasionally.  

## Contents
- [Arithmetic Operators](#arithmetic-operators)
- [Basic function](#basic-function)
- [Function](#function)
- [Round](#round)
- [Trigonometry](#trigonometry)
- [Logical Operators](#logical-operators)
- [Data structure](#data-structure)
- [Data frame](#data-frame)
- [Finding](#finding)
- [Missing](#missing)
- [Set operator](#Set-operator)
- [Conditional operator](#conditional-operator)
- [Loop](#loop)
- [Summary statistics](#summary-statistics)
- [Random Number](#random-number)
- [Sequences](#sequences)
- [Repeating](#repeating)
- [Working directory](#working-directory)



## Arithmetic Operators

 R  | Python | Matlab | Description
-------  | -------| -------| -------
x<-c <br> x=c| x=c| x=c| define x and assign c to it
x+y| x+y| x+y| addition
x-y|x-y |x-y | subtraction
x*y|x*y |x*y | multiplication
x/y|x/y |x/y | division
x^y|x**y|x.^y | power
x%/%y| x%y |rem(x,y)| Reminder


## Basic function

 R  | Python | Matlab | Description
-------   | -------| -------| -------
sqrt(x)| math.sqrt(x) |sqrt(x)| square root
log(x)|math.log(x) |log(x) | logarithm base e
log10(x)|math.log10(x) | log10() |logarithm base 10
exp(x)|math.exp(x) | exp(x)|




## Function

 R  | Python | Matlab | Description
-------   | -------| -------| -------
name<-function(arguments){<br>  script <br>return(output)<br>}|def name (arguments):<br>&nbsp; &nbsp; script<br> &nbsp; &nbsp; return output|function [returns]=name(argument):<br> &nbsp; &nbsp;script<br> &nbsp; &nbsp;returns=output|


## Round

 R  | Python | Matlab|Description
-------   | -------| -------| -------
round()|round() |  round() |  round
ceil() | math.ceil()|  ceil() | round up
floor()| math.floor()|  floor()| round down

## Trigonometry

 R  | Python | Matlab|Description
-------   | -------| -------| -------
sin(x)| math.sin(x)| sin(x)|
asin(x)| math.asin(x)| asin(x)


## Logical Operators

 R  | Python | Matlab|Description
-------   | -------| -------| -------
FALSE|False | false|
TRUE|True |true |
a & b| a & b| a & b|AND  
a\|b | a\|b | a\|b| OR
!a | not a|~a|Not
xor(a,b)| a!=b | xor(a,b)|Logical exclusive OR
x&&y| and |x&&y|Short-circuit FALSE
\|\| |  or | \|\| |Short-circuit TRUE
any(x,y)| any([x,y])|any(x,y)|TRUE if any element is TRUE
all(a,b)| all([x,y])|all(x,y)|TRUE if all element is TRUE

## Data structure

 R  | Python | Matlab|Description
-------   | -------| -------| -------
vec=c(x,y,z)| vec=np.array([[x, y, z]]) | vec=[x y z]| vector
vec[i]| vec[i] |cev(i) | call the ith  element
t(c(x,y,z))| np.array([[x, y, z]]).T | [x y z]'| transpose of vector
mat=matrix(c(x,y,z,w),ncol=2)| mat=np.array([[x, y], [z,w]]) | mat=[x y, z w]| matrix
matrix(0,x,y)| zeros(x,y) | zeros(x,y) | Zero x*y matrix
dim(mat)| mat.shape|size(mat)  | size of matrix
diag(x)| np.diag(x)  | diag(x)|  diagonal matrix  
mat[i,]| mat[i-1,] |mat(i,:) | call the ith  row
mat[,j]| mat[,j-1] |mat(:,j) | call the ith  column
rbind(mat1,mat2)|np.concatenate(mat1,mat2,axis=0)|[mat1;mat2]| combine by row
cbind(mat1,mat2)|np.concatenate(mat1,mat2,axis=1)|[mat1,mat2]| combine by column


## Data frame

 R  | Python | Matlab|Description
-------   | -------| -------| -------
df=data.frame()| df=pd.DataFrame()|ds = mat2dataset(X)|
apply(df, 2, FUN)| df.apply(FUN, axis=1)|varfun(FUN,ds)| Run function on columns
apply(df, 1, FUN)| df.apply(FUN, axis=0)|varfun(FUN,ds)| Run function on rows

```
df = pd.DataFrame([[4, 4], [5, 5]])
df.apply(lambda x: x.mean(), axis=0)
df.apply(lambda x: x.mean(), axis=1)
```

## Finding

 R  | Python | Matlab|Description
-------  | -------| -------| -------
which(x==a)|np.where(x == a)|find(x == a)|Check the condition
which.max(x)|np.where(x==np.max(x))|find(x == max(x))| Find maximum
which.min(x)|np.where(x==np.min(x))|find(x == min(x))| Find minimum

## Missing

 R  | Python | Matlab|Description
-------   | -------| -------| -------
NA| np.nan |NaN| Missing
NaN| np.nan |NaN| Not a number
Inf | 'inf'|Inf | infinity
is.na(x)| np.isnan(x) |  isnan(x)|  Test whether it na
anyNA(X)| x.isnull().any().any()| | sum(isnan(x)>1|Test whether there is any na

## Set operator

| R  | Python | Matlab|Description
-------   | -------| -------| -------
unique(x)| np.unique(x) | unique(x)| find unique value
intersect(x,y)| x.intersectionf(y) |intersect(x,y) | find itersect value
setdiff(x,y)|x-y | setdiff(x,y) | difference of set
 see below|x.symmetric_difference(y)|setxor(x,y)|| set exclusion
  is.element(c,x)<br> c%in%x|np.isin(c,x)|ismemner(c,x)| True if x includes c

``setxor <- function(x,y) setdiff(union(x,y), intersect(x,y))``

## Conditional operator

R  | Python | Matlab|Description
-------   | -------| -------| -------
if (condition){<br>dd<br>} |if condition : <br>  &nbsp; codes|if condition <br> &nbsp; code <br> end |
ifelse|elif | elseif|
else| else | else|

## loop

| R  | Python | Matlab|Description
-------   | -------| -------| -------
for (i in indices){ <br> codes <br>}| for i in indices: <br> &nbsp; codes | for i=indices <br> &nbsp; codes <br>  <br> end|


## Summary statistics

 R  | Python | Matlab|Description
-------   | -------| -------| -------
 mean(x)| np.mean(x)|mean(x)|
median(x) | np.median(x)|median(x)|calculate sample mdeian
 var(x)| np.var(x)|var(x)|
 sd(x)| np.std(x) |std(x)|
cov(x,y) |np.cov(x,y)| cov(x,y)|
 corr(x,y)|np.corrcoef(x,y)| corr(x,y)|


## Random number

R  | Python | Matlab|Description
-------   | -------| -------| -------
runif(n,a,b)| np.random.uniform(a,b,n)| unifrnd(a,b,1,n)
rnorm(n,a,b)| np.random.normal(a,b,n)| normrnd(a,b,1,n)

## Sequences

R  | Python | Matlab|Description
-------   | -------| -------| -------
seq(a,b,by=c)| np.arange(a,b,c)| a:c:b
seq(a,b,length.out=c)| np.linspace(a,b,c)|  linspace(a,b,c)

## Repeating  

R  | Python | Matlab|Description
-------   | -------| -------| -------
rep(a,n)| np.repeat(a,n)|repmat(a,1,n)|
a:b|a:b|a:b|

## Working directory

 R  | Python | Matlab|Description
-------   | -------| -------| -------
dir()| os.dir(".") |dir |  list files and folders
getwd|os.getwcd() | pwd | Display the current working directory
setwd|os.chdir("folder") | cd folder | Change the current working directory
ls()| dir() | who | list of loaded objects
rm(x)| del x| clear x | clear object x from memory
rm(list = ls())| see below|clear all |  clear all objects from memory
```
for obj in dir(): 
    if obj[0] == '_': continue
    del globals()[obj]
```
## Needed Python library
``import math``<br>
`` import numpy as np``<br>
`` import pandas as pd``



More useful link:
- [Matlab / R](https://umaine.edu/mathematics/wp-content/uploads/sites/70/2018/08/matlabR.pdf)
- [MATLAB commands in numerical Python](http://mathesaurus.sourceforge.net/matlab-python-xref.pdf)

### License
Copyright (c) 2018 Saeid Amiri

**[⬆ back to top](#contents)**
