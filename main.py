#Python NumPy:

#NumPy getting sarted:
'''import numpy
arr = numpy.array([1, 2, 3, 4, 5])
print(arr)

import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)

import numpy as np
print(np.__version__)'''
import datetime

#NumPy creating arrys:
'''import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))

import numpy as np
arr = np.array((1, 2, 3, 4, 5))
print(arr)

import numpy as np# 0 Dimensions array
arr = np.array(42)
print(arr)

import numpy as np# 1 Demensions array
arr = np.array([1, 2, 3, 4, 5])
print(arr)

import numpy as pn# 2 Dimensions array
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)

import numpy as np# 3 Dimensions array
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr)

import numpy as np# check number of Dimension
a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

import numpy as np# Higher Dimesional arrays
arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print(f"number of dimesions: {arr.ndim}")'''

#NumPy indexing:
'''import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr[0])

import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr[2] + arr[3])

import numpy as np# Access 2-D Arrays
arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('2nd element on 1st row: ', arr[0, 1])

import numpy as np
arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('5th element on 2nd row: ', arr[1, 4])

import numpy as np# Access 3-D Arrays
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr[0,1,2])

import numpy as np# Negative indexing
arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('Last element from 2nd dim: ', arr[1, -1])'''

#NumPy array slicing:
'''import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:5])

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[4:])

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[:4])

import numpy as np# Negative slicing
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[-3:-1])

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:5:2])

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[::2])

import numpy as np# slicing 2-D arrays
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[1, 1:4])

import numpy as np
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[0:2, 2])

import numpy as np
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[0:2, 1:4])'''

#NumPy Data types:
'''import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr.dtype)

import numpy as np
arr = np.array(['apple', 'banana', 'cherry'])
print(arr.dtype)

# Creating Arrays With a Defined Data Type
import numpy as np
arr = np.array([1, 2, 3, 4], dtype='S')
print(arr)
print(arr.dtype)

import numpy as np # Create an array with data type 4 bytes integer:
arr = np.array([1, 2, 3, 4], dtype='i4')
print(arr)
print(arr.dtype)

#value error
import numpy as np #What if a Value Can Not Be Converted?
arr = np.array(['a', '2', '3'], dtype='i')
print(arr.dtype)

import numpy as np# converting data type on existing arrays
arr = np.array([1.1, 2.1, 3.1])
newarr = arr.astype('i') # i as a perameter value of integer
print(newarr)
print(newarr.dtype)

import numpy as np
arr = np.array([1.1, 2.1, 3.1])
newarr = arr.astype(int) # int is a perameter value of integer
print(newarr)
print(newarr.dtype)

import numpy as np # Change data type from integer to boolean:
arr = np.array([1, 0, 3])
newarr = arr.astype(bool)
print(newarr)
print(newarr.dtype)'''

#NumPy Array Copy vs View:

'''import numpy as np
var = np.array([1, 2, 3, 4])
c = var.copy()
print(f"var = {var}")
print(f"copy = {c}")

import numpy as np#Make a copy, change the original array, and display both arrays:
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42
print(arr)
print(x)

import numpy as np#Make a view, change the original array, and display both arrays:
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42
print(arr)
print(x)

import numpy as np#Make a view, change the view, and display both arrays:
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
x[0] = 31
print(arr)
print(x)

import numpy as np ##Make a copy, change the original array, and display both arrays:
arr = np.array([1,2,3,4,5])
x = arr.copy()
y = arr.view()
print(x.base)
print(y.base)'''

#NumPy Array Shape & DReshaping:

'''import numpy as np
arr = np.array([[1,2,3,4], [5,6,7,8]])
print(arr.shape)

import numpy as np
arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print('shape of array :', arr.shape)

import numpy as np#Reshape From 1-D to 2-D
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(4, 3)
print(newarr)

import numpy as np#Reshape From 1-D to 3-D
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(2, 3, 2)
print(newarr)
print(newarr.ndim)
print(newarr.dtype)

import numpy as np
arr = np.array([1,2,3,4,5,6,7,8])
newarr = arr.reshape(3, 3)
print(newarr)
print(newarr.shape)

import numpy as np#Check if the returned array is a copy or a view:
arr = np.array([1,2,3,4,5,6,7,8])
print(arr.reshape(2, 4).base)#The example above returns the original array, so it is a view.

import numpy as np
arr = np.array([1,2,3,4,5,6,7,8])
newarr = arr.reshape(2,2,-1)
print(newarr)#Convert 1D array with 8 elements to 3D array with 2x2 elements:

import numpy as np
arr = np.array([[1,2,3], [4,5,6]])
newarr = arr.reshape(-1)
print(newarr)#Convert the array into a 1D array
#exercise:
import numpy as np
a1 = np.array([1,2,3,4])
a2 = np.array([5,6,7,8])
print(np.concatenate((a1,a2)))'''

#NumPy Array Iterating:

'''import numpy as np#Iterate on the elements of the following 1-D array:
arr = np.array([1,2,3])
for x in arr:
    print(x)

import numpy as np#Iterate on the elements of the following 2-D array:
arr = np.array([[1,2,3], [4,5,6]])
for x in arr:#If we iterate on a n-D array it will go through n-1th dimension one by one
    print(x)

import numpy as np#Iterate on each scalar element of the 2-D array:
arr = np.array([[1,2,3], [4,5,6]])
for x in arr:
    for y in x:
        print(y)

import numpy as np#Iterate on the elements of the following 3-D array:
arr = np.array([[[1,2,3], [4,5,6],[7,8,9]]])
for x in arr:
    print(x)

import numpy as np#Iterate down to the scalars:
arr = np.array([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])
for x in arr:
    for y in x:
        for z in y:
            print(z)

import numpy as np#Iterate through the following 3-D array:
arr = np.array([[[1,2], [3,4]], [[5,6], [7,8]]])
for x in np.nditer(arr):
    print(x)

import numpy as np#Iterate through the array as a string:
arr = np.array([1,2,3])
for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
    print(x)

import numpy as np#Iterate through every scalar element of the 2D array skipping 1 element:
arr = np.array([[1,2,3,4], [5,6,7,8]])
for x in np.nditer(arr[:, ::2]):
    print(x)

import numpy as np#Enumerate on following 1D arrays elements:
arr = np.array([1,2,3])
for idx, x in np.ndenumerate(arr):#with index number:
    print(idx, x)

import numpy as np#Enumerate on following 2D array's elements:
arr = np.array([[1,2,3,4], [5,6,7,8]])
for idx, x in np.ndenumerate(arr):
    print(idx, x)'''

#NumPy Array Join:

'''import numpy as np
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)

import numpy as np#Join two 2-D arrays along rows (axis=1):
arr1 = np.array([[1,2], [3,4]])
arr2 = np.array([[5,6], [7,8]])
arr = np.concatenate((arr1, arr2), axis= 1)
print(arr)

import numpy as np#Joining Arrays Using Stack Functions
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr = np.stack((arr1, arr2), axis= 1)
print(arr)

import numpy as np#NumPy provides a helper function: hstack() to stack along rows.
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr = np.hstack(((arr1, arr2)))
print(arr)

import numpy as np#NumPy provides a helper function: vstack()  to stack along columns.
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr = np.vstack((arr1, arr2))
print(arr)

import numpy as np#NumPy provides a helper function: dstack() to stack along height, which is the same as depth.
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr = np.dstack((arr1,arr2))
print(arr)'''

#NUmPy Array Split:

'''import numpy as np#We use array_split() for splitting arrays, we pass it the array we want to split and the number of splits.
arr = np.array([1,2,3,4,5,6])
newarr = np.array_split(arr, 3)
print(newarr)

import numpy as np#Split the array in 4 parts:
arr = np.array([1,2,3,4,5,6])
newarr = np.array_split(arr, 4)
print(newarr)

import numpy as np#Split Into Arrays:
arr = np.array([1,2,3,4,5,6])
newarr = np.array_split(arr, 3)
print(newarr[0])#Access the splitted arrays:
print(newarr[1])
print(newarr[2])

import numpy as np#Splitting 2-D Arrays
arr = np.array([[1,2], [3,4], [5,6], [7,8], [9,10], [11, 12]])
newarr = np.array_split(arr, 3)
print(newarr)

import numpy as np#Split the 2-D array into three 2-D arrays.
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3)
print(newarr)

import numpy as np#Split the 2-D array into three 2-D arrays along rows.
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3, axis=1)
print(newarr)

import numpy as np#Use the hsplit() method to split the 2-D array into three 2-D arrays along rows.
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.hsplit(arr, 3)
print(newarr)'''

#NumPy Aarray Search:

'''import numpy as np#Find the indexes where the value is 4:
arr = np.array([1,2,3,4,5,4,4])
x = np.where(arr == 4)
print(x)

import numpy as np#Find the indexes where the values are even:
arr = np.array([1,2,3,4,5,6,7,8])
x = np.where(arr%2 == 0)
print(x)

import numpy as np#Find the indexes where the values are odd:
arr = np.array([1,2,3,4,5,6,7,8])
x = np.where(arr%2 == 1)
print(x)

import numpy as np#Find the indexes where the value 7 should be inserted:
arr = np.array([6,7,8,9])
x = np.searchsorted(arr, 7)
print(x)

import numpy as np#By default the left most index is returned, but we can give side='right' to return the right most index instead.
arr = np.array([6,7,8,9])
x = np.searchsorted(arr, 7, side='right')
print(x)

import numpy as np#To search for more than one value, use an array with the specified values.
arr = np.array([1,3,5,7])
x = np.searchsorted(arr, [2, 4, 6])
print(x)'''

#NumPy Array Sort:

'''import numpy as np#Sort the array:
arr = np.array([3,2,0,1])
print(np.sort(arr))

import numpy as np#Sort the array alphabetically:
arr = np.array(['banana', 'cherry', 'apple'])
print(np.sort(arr))

import numpy as np#Sort a boolean:
arr = np.array([True, False, True])
print(np.sort(arr))

import numpy as np#Sorting a 2-D Array
arr = np.array([[3,2,4], [5,0,1]])
print(np.sort(arr))'''

#NumPy Array Filter:

'''import numpy as np#Filtering Arrays
arr = np.array([41,42,43,44])
x = [True, False, True, False]#The example above will return [41, 43], why?Because the new array contains only the values where the filter array had the value True, in this case, index 0 and 2.
newarr = arr[x]
print(newarr)

import numpy as np#In the example above we hard-coded the True and False values, but the common use is to create a filter array based on conditions.
arr = np.array([41, 42, 43, 44])
# Create an empty list
filter_arr = []
# go through each element in arr
for element in arr:
  # if the element is higher than 42, set the value to True, otherwise False:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)#Create a filter array that will return only values higher than 42:

import numpy as np#Create a filter array that will return only even elements from the original array:
arr = np.array([1,2,3,4,5,6,7])
#create an empty list
filter_arr = []
#go through each element in arr
for element in arr:
    #if the element is completely divisble by 2, set the value to True, otherwise False
    if element%2 == 0:
        filter_arr.append(True)
    else:
        filter_arr.append(False)
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

import numpy as np#Creating Filter Directly From Array
arr = np.array([41,42,43,44])
filter_arr = arr > 42
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

import numpy as np#Create a filter array that will return only even elements from the original array:
arr = np.array([1,2,3,4,5,6,7])
filter_arr = arr%2 == 0
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)'''

#NumPy Random:


#Random Numbers in NumPy
'''from numpy import random# NumPy offers the random module to work with random numbers.
x = random.randint(100)
print(x)

from numpy import random# The random module's rand() method returns a random float between 0 and 1.
x = random.rand()
print(x)

from numpy import random# The randint() method takes a size parameter where you can specify the shape of an array
x = random.randint(100, size=(5))
print(x)#Generate a 1-D array containing 5 random integers from 0 to 100:

from numpy import random# Generate a 2-D array with 3 rows, each row containing 5 random integers from 0 to 100:
x = random.randint(100, size=(3,5))
print(x)

from numpy import random# The rand() method also allows you to specify the shape of the array.
x = random.rand(5)
print(x)# Generate a 1-D array containing 5 random floats:

from numpy import random# Generate a 2-D array with 3 rows, each row containing 5 random numbers:
x = random.rand(3,5)
print(x)

# Generate Random Number From Array
from numpy import random# Return one of the values in an array:
x = random.choice([3,5,7,9])
print(x)

from numpy import random# Generate a 2-D array that consists of the values in the array parameter (3, 5, 7, and 9):
x = random.choice([3,5,7,9], size=(3,5))
print(x)

#Example:
from numpy import random
low = 1
high = 100
guesses = 0
number = random.randint(low, high)
while True:
    guess = int(input(f"enter a number between ({low} - {high}): "))
    guesses +=1
    if guess < number:
        print(f"{guess} is too low")
    elif guess > number:
        print(f"{guess} is too high")
    else:
        print(f"{guess} is correct")
        break
print(f"this round took you {guesses} guesses")'''

# Random Data Distribution

'''from numpy import random
x = random.choice([3,5,7,9], p=[0.1,0.3,0.6,0.0], size=(100))
print(x)

from numpy import random# Same example as above, but return a 2-D array with 3 rows, each containing 5 values.
x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(3, 5))
print(x)'''

#Random Permutations

'''from numpy import random# Randomly shuffle elements of following array:
import numpy as np
arr = np.array([1,2,3,4,5])
random.shuffle(arr)
print(arr)# The shuffle() method makes changes to the original array.

from numpy import random# Generate a random permutation of elements of following array:
import numpy as np
arr = np.array([1,2,3,4,5])
print(random.permutation(arr))# The permutation() method returns a re-arranged array (and leaves the original array un-changed).'''

#Normal Distribution

'''from numpy import random# Normal Distribution
x = random.normal(size=(2,3))
print(x)

from numpy import random# Generate a random normal distribution of size 2x3 with mean at 1 and standard deviation of 2:
x = random.normal(loc=1, scale=2, size=(2, 3))
print(x)'''

#Binomial Distribution

'''from numpy import random
x = random.binomial(n=10, p=0.5, size=10)
print(x)

#Uniform Distribution

from numpy import random
x = random.uniform(size=(2, 3))
print(x)

#Logistic Distribution

from numpy import random
x = random.logistic(loc=1, scale=2, size=(2, 3))
print(x)'''

#Multinomial Distribution

'''from numpy import random
x = random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
print(x)

#Exponential Distribution

from numpy import random
x = random.exponential(scale=2, size=(2, 3))
print(x)

#Chi Square Distribution

from numpy import random
x = random.chisquare(df=2, size=(2, 3))
print(x)

#Rayleigh Distribution

from numpy import random
x = random.rayleigh(scale=2, size=(2, 3))
print(x)

#Pareto Distribution

from numpy import random
x = random.pareto(a=2, size=(2, 3))
print(x)

#Zipf Distribution

from numpy import random
x = random.zipf(a=2, size=(2, 3))
print(x)'''


#NumPy ufuncs:


'''x = [1,2,3,4]
y = [4,5,6,7]
z = []
for i, j in zip(x,y):
    z.append(i + j)
print(z)

import numpy as np
x = [1,2,3,4]
y = [4,5,6,7]
z = np.add(x,y)
print(z)'''

#Create Your Own ufunc:

'''import numpy as np#Create your own ufunc for addition:
def myadd(x,y):
    return x+y
myadd = np.frompyfunc(myadd, 2, 1)
print(myadd([1,2,3,4], [5,6,7,8]))

import numpy as np#Check if a function is a ufunc:
print(type(np.add))

import numpy as np#Check the type of another function: concatenate():
print(type(np.concatenate))

import numpy as np#Check the type of something that does not exist. This will produce an error:
print(type(np.blahblah))

import numpy as np
if type(np.add) == np.ufunc:
    print('add is ufunc')
else:
    print('add is not ufunc')'''

#Simple Arithmetic

'''import numpy as np
arr1 = np.array([10,11,12,13,14,15])
arr2 = np.array([20,21,22,23,24,25])
newarr = np.add(arr1, arr2)
print(newarr)

import numpy as np
arr1 = np.array([10,20,30,40,50,60])
arr2 = np.array([20,21,22,23,24,25])
newarr = np.subtract(arr1, arr2)
print(newarr)

import numpy as np
arr1 = np.array([10,20,30,40,50,60])
arr2 = np.array([20,21,22,23,24,25])
newarr = np.multiply(arr1, arr2)
print(newarr)

import numpy as np
arr1 = np.array([10,20,30,40,50,60])
arr2 = np.array([3,5,10,8,2,33])
newarr = np.divide(arr1, arr2)
print(newarr)

import numpy as np
arr1 = np.array([10,20,30,40,50,60])
arr2 = np.array([3,5,6,8,2,33])
newarr = np.power(arr1, arr2)
print(newarr)

import numpy as np
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 7, 9, 8, 2, 33])
newarr = np.mod(arr1, arr2)
print(newarr)

import numpy as np
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 7, 9, 8, 2, 33])
newarr = np.remainder(arr1, arr2)
print(newarr)

import numpy as np#Return the quotient and mod:
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 7, 9, 8, 2, 33])
newarr = np.divmod(arr1, arr2)
print(newarr)

import numpy as np
arr = np.array([-1,-2,1,2,3,-4])
newarr = np.absolute(arr)
print(newarr)'''

#Rounding Decimals
#Remove the decimals, and return the float number closest to zero. Use the trunc() and fix() functions.
'''import numpy as np
arr = np.trunc([-3.1666,3.6667])
print(arr)

import numpy as np
arr = np.fix([-3.1666,3.6667])
print(arr)

import numpy as np
arr = np.around(3.1666,2)
print(arr)

import numpy as np
arr = np.floor([-3.1666,3.6667])
print(arr)

import numpy as np
arr = np.ceil([-3.1666, 3.6667])
print(arr)'''

#NumPy Logs

'''import numpy as np#Find log at base 2 of all elements of following array:
arr = np.arange(1, 10)
print(np.log2(arr))

import numpy as np
arr = np.arange(1, 10)
print(np.log10(arr))

import numpy as np
arr = np.arange(1,10)
print(np.log(arr))

from math import log
import numpy as np
nplog = np.frompyfunc(log, 2, 1)
print(nplog(100, 15))'''

#NumPy Summations

'''import numpy as np
arr1 = np.array([1,2,3])
arr2 = np.array([1,2,3])
newarr = np.add(arr1, arr2)
print(newarr)

import numpy as np
arr1 = np.array([1,2,3])
arr2 = np.array([1,2,3])
newarr = np.sum([arr1, arr2])
print(newarr)

import numpy as np
arr1 = np.array([1,2,3])
arr2 = np.array([1,2,3])
newarr = np.sum([arr1, arr2], axis=1)
print(newarr)

import numpy as np#Cummulative sum means partially adding the elements in array.
arr = np.array([1,2,3,4])
newarr = np.cumsum(arr)
print(newarr)'''

#NumPy Products

'''import numpy as np
arr = np.array([1,2,3,4])
x = np.prod(arr)
print(x)

import numpy as np
arr1 = np.array([1,2,3,4])
arr2 = np.array([5,6,7,8])
x = np.prod([arr1, arr2])
print(x)

import numpy as np
arr1 = np.array([1,2,3,4])
arr2 = np.array([5,6,7,8])
newarr = np.prod([arr1, arr2], axis=1)
print(newarr)

import numpy as np
arr = np.array([5,6,7,8])
newarr = np.cumprod(arr)
print(newarr)'''

#NumPy Differences

'''import numpy as np
arr = np.array([10,15,25,5])
newarr = np.diff(arr)
print(newarr)

import numpy as np
arr = np.array([10,15,25,5])
newarr = np.diff(arr, n=2)
print(newarr)'''

#NumPy LCM Lowest Common Multiple

import numpy as np
num1 = 4
num2 = 6
x = np.lcm(num1, num2)
print(x)

import numpy as np
arr = np.array([3,6,9])
x = np.lcm.reduce(arr)
print(x)

import numpy as np
arr = np.arange(1,11)
x = np.lcm.reduce(arr)
print(x)

































































































































































































