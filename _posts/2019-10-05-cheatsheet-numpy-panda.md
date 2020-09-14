---
layout: post
title: Cheat Sheet of Numpy,  Panda and python core 
description: A brief of Useful functions of Numpy and Panda
date: 2019-10-05
author: Saeid Amiri
published: true
tags: Python Numpy Pandas Cheat-sheet
categories: Python_learn Numpy Pandas Cheat-sheet 
comments: false
---


## Contents

- [Primer](#primer)
  - [Libraries](#libraries) 
  - [Starting](#starting)
  - [Mathematical operations](#mathematical-operations)
  - [Built-in Constants](#built-in-Constants)
  - [Type](#type)
  - [Lists](#lists)
  - [Dictionary](#dictionary)
  - [Built-in function](#built-in-function)
  - [User Function](#user-function)
  - [Iterates](#iterates)
  - [Boolean Comparisons](#boolean-comparisons)
  
  
- [Numpy](#numpy)
  - [Arrays](#arrays)
  - [Random](#random)
  - [O\S](#o\s) 
  - [Inspecting arrays](#inspecting-arrays)
  - [sorting](#sorting)     
  - [Reshaping](#reshaping)
  - [Concatenate](#concatenate)
  - [Copying](#copying) 
  - [Adding/Removing Elements](#adding/removing-elements)
  - [Indexing](#indexing)  
  - [Slicing](#slicing)
  - [Subsetting](#subsetting)
  - [Vector Math](#vector-Math)
  - [Aggregate Functions](#aggregate-functions) 

- [Pandas ](#pandas)
  - [Create Data frame and series](#create-data-frame-and-series)
  - [I/O](#i/o)
  - [Inspecting arrays](#inspecting-arrays)
  - [sorting](#sorting) 
  - [Adding/Removing Elements](#adding/removing-elements)
  - [Missing ](#missing)  
  - [Indexing](#indexing)
  - [Subsetting](#subsetting)
  - [Group by](#group-by)
  - [Apply ](#apply) 
  - [Join/Combine](#join/Combine)
  - [Aggregate Functions](#aggregate-functions)  
 
# Primer

## Libraries

|||
|---|---|
```pip install numpy```| Install library 
```pip install git+https://github.com/mwaskom/seaborn.git```| Install from GitHub
```import numpy```| load library

## starting 

By putting # infront line, Python ignores running the rest. 

|||
|---|---|
|```x=1 # Python ignore``` | Python ignores running after #
|```import os``` <br>
|```os.getcwd()``` | See the working directory
|```os.chdir()``` | Change  the working directory

## Mathematical operations

|||
|---|---|
|```1 + 2``` | Addition|
|```1 - 2``` | Subtraction|
|```1 / 2``` | Division|
|```1 * 2``` | Multiplication|
|```1 ** 2```| Power| 
|```x += 1```| Assign the value of x + 1 to x|
|```x -= 1```| Assign the value of x - 1 to x|

## Built-in Constants

|||
|---|---|
```None``` | Absence of a value
```False```| The bool type false
```True```| The bool type true


## Type 

|||
|---|---|
```str()```  |  Convert the string
```int()```  |  Convert the integer
```float()```|  Convert the floating
```bool()``` |  Convert the boolen


## Lists

|||
|---|---|
```l = list()```|Assign an empty list to l
```l = [3,2]```| Create list and assgn 3 and 2 to it 
```l[0]```     | Return the first value 
```l[-1]```    | Return the last value 
```l[-4:]```    | Return that last 4 items.  
```l[1:4]```| Return subset containing the second till  fourth values 
```l[1::3]```| Return every third items starting from l[1]
```l.append(1)```| Append the value 1 to the end of l
```l+[1]```| Append the value 1 to the end of l
```l.sort()```|  Sort l and replace original with it 
```l.reverse()``` | Sort reversely the items in l
```l.remove(a)``` |  Removes the first item equals to a.  
```l.pop(2)```| Restun the second item and drop it from l
```[i for i in range(1,100) if i%2==0]```| Generate list of even number between 1 and 100 
```l= [1, "", 3]```| Creat a list with missing value. 


## Dictionary

|||
|---|---|
```d={"weight":2.4, "height":15}```| Creat a dictionary with keys "weight" and "height", and their corresponding values of 2.4, 15 
```d["weight"]``` | Return values corrsponding "weight"
```d.keys()```| Return the keys from d
```d.values()```| Return the values from d
```d.items()```| Return  (key, value) pairs from d


## Built-in function

|||
|---|---|
```len(x)```| Return the number of elements in x
```min(x)```| Return the min of the values of x
```sum(x)```| Return the sum of the values of x
```type(x)```| Return type of the values of x
```range(3,10)```| Generate a series of number from a number (3) to another number (10) with specific increment


##  User Function

|||
|---|---|
|```def name (argument):``` <br> &nbsp; ```script``` <br>```return output```| Create a function 
|```name = lambda arguments: script```| Create one line function
```if state1:```<br> &nbsp;```script1``` <br> ```elif x < 0:```<br> &nbsp; ```script2``` <br> ```else:```<br> &nbsp; ```script3``` | Test state1 and state2, and run script corresponding to the correct statement, otherwise run script3


## Iterates

|||
|---|---|
|```for value in obj:```<br>&nbsp; ```script```| Iterate the code for each value in obj |
```while cond:```<br> &nbsp; ```script```|Run the code until reach to condion  

## Boolean Comparisons

|||
|---|---|
```x == 2```| Test whether x is equal to 2
```x != 2```| Test whether x is not equal to 2
```x < 2```| Test whether x is less than 2
```x <= 2```| Test whether x is less than or equal to 2
```x > 2```| Test whether x is greater than 2
```x >= 2```| Test whether x is greater than or equal to 2
```(x == 2) or (y == 1)```| Test whether x is equal to 5 or y is equal to 1
```(x == 2) | (y == 1)```| Test whether x is equal to 5 or y is equal to 1
```(x == 2) and  (y == 1)```| Test whether x is equal to 5 and y is equal to 1
```(x == 2) & (y == 1)```| Test whether x is equal to 5 and  y is equal to 1
```3 in l```| Checks whether the value 3 exists in the  l


# Numpy 

Numpy (NUMerical PYthon) provides very useful arrays structure to work with data. 

|||
|---|---|
```import numpy as np``` | 
```pips install numpy ```|
```python3 -m pip install --upgrade  numpy``` |

## Arrays

|||
|---|---|
```np.array([1,2,3])``` | One dimensional array
```np.array([(1,2,3),(4,5,6)])``` | Two dimensional array
```arr[i]``` | The ith element
```arr[i:]``` | The ith row
```arr[i][j]``` | The ith and jth element, the same as ```arr[i,j]``` 
```np.full((2,1),2.2)``` | 2x1 array with all values 2.2
```np.linspace(0, 1, 10)``` |  
```np.eye(3)``` | A diagonal array of size 3x3 (Identity matrix)
```np.zeros(3)``` | An array of length 3 with all values 0
```np.ones((4,2))``` |  An array of size 4x2 with all values 1
```np.arange(1,14,4)``` | An 1D array from 1 to 14 with step 4

## Random 

|||
|---|---|
```np.random.rand(4,2)``` | Generates a 4x2  array of random number from uniform
```np.random.rand(4,2)``` | Generates a 4x2 array  of random number from standard normal
```np.random.randint(low=1,high=20, size=(2,3))``` | Generates a 2x3 array of random ints between 1–20
```np.random.choice(arr,size=s,replace=True,p=pr)```| Resamples of size s from arr acording probability pr  

## O\S

|||
|---|---|
```np.savetxt('file.txt',arr,delimiter=' ')``` | Writes to a text file
```np.savetxt('file.csv',arr,delimiter=',')```| Writes to a CSV file
```np.loadtxt('file.txt')``` | Loads from a text file
```np.genfromtxt('file.csv',delimiter=',')``` | Loads from a CSV file
```np.save('file_of_arr.npy ', arr)``` | Saves  array into a file 
```np.savez('file_of_arr.npz', arr1, arr2)``` | Saves  array into a file 
```np.load('my_array.npy')``` | Loads arrays
 

## Inspecting arrays

|||
|---|---|
```arr.dtype``` | Returns type of elements in array
```arr.size``` | Returns number of elements in array
```len(arr)``` | Length of array
```arr.shape``` | Returns dimensions of arr
```arr.astype(dtype)``` | Convert arr elements to type dtype
```arr.tolist()``` | Convert arr to a Python list

## Sorting

|||
|---|---|
```arr.sort()``` | Sort elements of arr
```arr.sort(axis=0)``` | Sorts elements of axis=0 of arr

## Reshaping

|||
|---|---|
```arr.reshape(4,3)``` | Reshapes arr to 4x3 without changing data
```arr.resize((4,3))``` | Changes arr shape to 4x3 and fills new values with 0
```arr.T``` | Transposes arr

## Concatenate

|||
|---|---|
```np.concatenate(arr1,arr2,axis=0)``` |concatenate arr2 to arr1 along  the axis
```np.hstack((arr1,arr2))``` |Stack arrays horizontally (column wise) 
```np.vstack((arr1,arr2))``` |Stack arrays vertically (row wise) 


## Copying 

|||
|---|---|
```arr2 = arr1.view()``` | Create a view of the array with the same data
```np.copy(arr)``` |   Create a copy of aar
```arr2 = arr1.copy()``` |   Create a deep copy of the array
 
## Adding/Removing Elements

|||
|---|---|
```np.append(arr1,arr2)``` |Append arr2 to arr1 
```np.insert(arr, 1, 10)``` |Insert 10 on index 1 items 
```np.delete(a,2)``` |Delete element on index 1 from array


## Indexing

|||
|---|---|
```arr[2]``` | Returns the element at index 2
```arr[2]=3``` | Assigns 3 to the element on index 2
```arr[2,3]``` | Returns the array element on index [2,3]
```arr[2,3]=10``` | Assigns 3 to the element on index [2,3]

## Slicing

|||
|---|---|
```arr[:2]``` | Returns the elements at indices 0,1
```arr[2:4]``` | Returns the elements at indices 2,3 
```arr[0:2,3]``` | Returns the elements on rows 0,1 at column 3
```arr[:,1]``` | Returns the elements on column 2
```arr[[1,2],[2,3]]``` | Returns the elements at indices [1,3] and [2,3]

## Subsetting

|||
|---|---|
```arr<2``` | Returns a boolen array, True for arr<2 and False for the rest 
```(arr1<2) & (arr2>3)``` | Returns a boolen array, True for (arr1<3) & (arr2>5) and False for the rest 
```arr[arr<2]``` | Select array with elements smaller than 2
```arr[~(arr<2)]``` | Select array with elements not smaller than 2


## Vector Math

|||
|---|---|
```np.add(arr1,arr2)``` | Elementwise add arr2 to arr1
```np.subtract(arr1,arr2)``` | Elementwise subtract arr2 from arr1
```np.multiply(arr1,arr2)``` | Elementwise multiply arr1 by arr2
```np.divide(arr1,arr2)``` | Elementwise divide arr1 by arr2
```np.power(arr1,arr2)``` | Elementwise raise arr1 raised to the power of arr2
```np.array_equal(arr1,arr2)``` | Returns True if the arrays have the same elements and shape
```np.sqrt(arr)``` | Square root of each element in the array
```np.sin(arr)``` | Sine of each element in the array
```np.log(arr)``` | Natural log of each element in the array
```np.abs(arr)``` | Absolute value of each element in the array
```np.ceil(arr)``` | Rounds up to the nearest int
```np.floor(arr)``` | Rounds down to the nearest int
```np.round(arr)``` | Rounds to the nearest int


## Aggregate Functions

|||
|---|---|
```arr.min()``` | Returns minimum value of arr
```arr.max(axis=0)``` | Returns maximum value of specific axis
```np.mean(arr,axis=0)``` | Returns mean of specific axis
```np.median(arr,axis=0)``` | Returns median of  specific axis
```arr.sum()``` | Returns sum of arr
```np.var(arr)``` | Returns the variance of array
```np.std(arr,axis=1)``` | Returns the standard deviation of specific axis
```np.quantile(arr,q=(q1,q2,..), axis=1)``` | Returns the (q1,q2, ....) quantiles of specific axis
```arr.corrcoef()``` | Returns correlation coefficient of array


#  Pandas 

Pandas is built for working with data set.

|||
|---|---|
```import pandas as pd``` | 
```pip3 install pandas ```|
```python3 -m pip install --upgrade  pandas``` |

## Create Data frame and series

|||
|---|---|
```pd.DataFrame(matrix, column=)``` |  Create data frame from matrix.
```pd.DataFrame(dict)``` | Create data frame from a dict, keys would be used as the name of columns
```pd.Series(list)``` |  Create a series from a list
```df.index = pd.date_range('2000/1/1', periods=df.shape[0])``` |  Add a date index to the data frame

## I/O

|||
|---|---|
```pd.read_csv(filename)``` | Load CSV file
```pd.read_table(filename)``` | Load from a delimited text file 
```pd.read_excel(filename)``` | Load from an Excel file
```pd.read_sql(query, connection_object)``` | Load from a SQL table/database
```pd.read_json(json_string)``` | Load from Read JSON formatted file s
```pd.read_html(url)``` | Create a data from from an html URL
```pd.read_clipboard()``` | Create a data frame from  the contents of your clipboard
```df.to_csv(filename)``` | Save df as  a CSV file
```df.to_excel(filename)``` | Save df as an Excel file
```df.to_sql(table_name, connection_object)``` | Save df to a SQL table
```df.to_json(filename)``` | Save df as a file in JSON format



## Inspecting arrays

|||
|---|---|
```arr.dtype``` | Returns type of elements in array
```arr.size``` | Returns number of elements in array
```len(arr)``` | Length of array
```arr.shape``` | Returns dimensions of arr
```arr.astype(dtype)``` | Convert arr elements to type dtype
```arr.tolist()``` | Convert arr to a Python list
```arr.value_counts(dropna=False)``` | View unique values and counts
```df.head(l)``` | Return the first l rows of the DataFrame
```df.tail(l)``` | Return  the last l rows of the DataFrame
```df.shape``` | Return the number of rows and columns
```df.info()``` | Return index, datatype, and memory information
```df.describe()``` | Return the summary statistics of numerical columns


## Sorting

|||
|---|---|
```df.sort_values(col)``` | Sort data frame values by col in ascending order
```df.sort_values(col,ascending=False)``` | Sort data frame values by col in descending order
```df.sort_values([col1,col2],ascending=[True,False])``` | data frame values by col1 in ascending order then col2 in descending order

 
## Adding/Removing Elements

|||
|---|---|
```df.columns = ['name1','name2']``` | Add new name to  columns
```df.rename(columns={'old_name': 'new_ name'}) ``` | Rename columns
```df.set_index('colu')``` | Change the index to the given column
```s.replace([1,2],['two','one'])``` | Replace 1 and 2 with  'two' and 'one', respectively

## Missing 

|||
|---|---|
```pd.isnull()``` | Find the null values, True for the null 
```pd.notnull()``` | Opposite to  pd.isnull()
```df.dropna()``` | Drop all rows that contain null values
```df.dropna(axis=1)``` | Drop all columns that contain null values
```df.dropna(axis=1,thresh=n)```| Keep only the column with at least n non null values
```df.fillna(x)``` | Replace all null values with x
```df.fillna([‘A’:0,‘B’:0])``` | Replace all null values in column ‘A’, and ‘B’, with 0, 1 respectively.
```s.fillna(s.mean())``` | Replace all null values with the mean or any other statistics you define


## Indexing

|||
|---|---|
```df[col]``` | Returns column with label col as Series
```df[[col1, col2]]``` | Returns columns corresponding  col1, col2
```df.iloc[i,:]``` | The (i-1)th row
```df.iloc[i,j]``` | The (i-1)th and (j-1)th element
```s.iloc[i]``` | Return elements in position (i-1) by position
```s.loc['index']``` | Return elements correspondingthe index

## Subsetting

|||
|---|---|
```df[df[col] < 1]``` | Return element less than 1 in column col
```df[(df[col] > 0.5) & (df[col] < 1)]``` | Return 0.5< element < 1 in column col

## Group by

|||
|---|---|
```df.groupby(col)```           | Group data frame based on col. 
```df.groupby(col).mean()```    | Calculate mean after grouping based on col. 
```df.groupby(col1).agg(fun)``` | Group the data frame based on col and run function on it. 
```df.pivot_table(index=col1,values=[col2,col3],aggfunc=mean)``` | Create a pivot table using index of col1 and calculates the mean (or any other function) of col2 and col3

## Apply 

|||
|---|---|
```df.apply(fun)``` |  Apply the function fun across each column
```df.apply(np.max,axis=1)``` |  Apply the function fun across each row


## Join/Combine

|||
|---|---|
```df1.append(df2)``` | Add the rows in df1 to the end of df2 (columns should be identical)
```pd.concat([df1, df2],axis=1)``` | Add the columns in df1 to the end of df2 (rows index should be identical)


## Aggregate Functions

|||
|---|---|
```df.describe()``` | Summary statistics for numerical columns
```df.count()``` | Returns the number of non-null values in each DataFrame column
```df.min()``` | Returns the minimum in each column
```df.max()``` | Returns the maximum in each column
```df.mean()``` | Returns the mean in each column
```df.median()``` | Returns the median in each column
```df.std()``` | Returns the standard deviation in each column
```df.corr()``` | Returns the correlation between columns

**[⬆ back to top](#contents)**
## References

[sw] https://swcarpentry.github.io/sql-novice-survey/ <br>
[dpo] https://docs.python.org/3.7/library/sqlite3.html <br>
[sw2] https://swcarpentry.github.io/sql-novice-survey/10-prog/index.html <br>
[gc] https://github.com/CoreyMSchafer/code_snippets/tree/master/Python-SQLite <br>
[dor] https://datacarpentry.org/R-ecology-lesson/05-r-and-databases.html <br>
[wdc] https://www.datacamp.com/community/tutorials/sqlite-in-r <br>
[sw3] https://swcarpentry.github.io/sql-novice-survey/11-prog-R/index.html <br>
[w3]  https://www.w3schools.com/sql/sql_create_table.asp <br>
[wt] https://www.techonthenet.com/sql/index.php <br>
[qlt] https://www.sqlitetutorial.net <br>

### License
Copyright (c) 2019 Saeid Amiri and Leila Alimehr
