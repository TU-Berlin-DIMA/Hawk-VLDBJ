import sys
import pandas as pd
import math
from itertools import islice

inputValues = []

f = open(sys.argv[1], 'r')

# load values abandoning first element
for value in islice(f, 1, None):
  inputValues.append(float(value))

num_elements = len(inputValues)
pdSeries = pd.Series(inputValues).sort_values().reset_index(drop=True)
#std_dev = pdSeries.std()
reference = pdSeries.median()

# remove outliers
print(reference)
while (pdSeries.iat[num_elements-1]/reference > 1.5):
  pdSeries.pop(num_elements-1)
  num_elements-=1

# caluclate measurements
std_dev = pdSeries.std()
min = pdSeries.min()
max = pdSeries.max()
variance = pdSeries.var()
mean = pdSeries.mean()

standard_error_of_the_mean = pdSeries.sem()
lower_bound_of_error_bar=mean-std_dev
upper_bound_of_error_bar=mean+std_dev

standard_error_of_the_mean=std_dev/math.sqrt(num_elements)
lower_bound_of_standard_error_of_the_mean_error_bar=mean-standard_error_of_the_mean
upper_bound_of_standard_error_of_the_mean_error_bar=mean+standard_error_of_the_mean

# print measurements
print(min, "\t", max, "\t", mean, "\t", variance, "\t", std_dev, "\t", standard_error_of_the_mean, "\t", lower_bound_of_error_bar, "\t", upper_bound_of_error_bar, "\t", lower_bound_of_standard_error_of_the_mean_error_bar, "\t", upper_bound_of_standard_error_of_the_mean_error_bar)
