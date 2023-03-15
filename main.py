import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# plt.style.use('seaborn')
# plt.style.use('fivethirtyeight')
# plt.style.use('dark_background')
plt.style.use('Solarize_Light2')
import math
import statistics as stats
import random
 #########
#FUNCTIONS#
 #########
def get_random(df, percent):
    num_perc = int((percent * len(df)) / 100)
    training_set = df.sample(num_perc)
    training_set = training_set.sort_index()
    test_set = df.drop(training_set.index)
    return training_set, test_set

def regression_equ(x_array, y_array, linear=True):
    x_fourth = x_array**4
    x_third = x_array**3
    x_squared = x_array**2
    y1x1 = y_array * x_array
    y1x2 = y_array * x_squared
    N = len(x_array)
    if linear:
        left_matrix = [[sum(x_squared), sum(x_array)], [sum(x_array), N]]
        right_matrix = [[sum(y1x1)], [sum(y_array)]]
        inv_mat_left = np.linalg.inv(left_matrix)
        solution = np.dot(inv_mat_left, right_matrix)
        slope, y_int = solution[0], solution[1]
        return (slope, y_int)
    else:
        left_matrix = [[sum(x_fourth), sum(x_third), sum(x_squared)], [sum(x_third), sum(x_squared), sum(x_array)], [sum(x_squared), sum(x_array), N]]
        right_matrix = [[sum(y1x2)], [sum(y1x1)], [sum(y_array)]]
        inv_mat_left = np.linalg.inv(left_matrix)
        solution = np.dot(inv_mat_left, right_matrix)
        x2_coeff, x_coeff, con_coeff = solution[0], solution[1], solution[2]
        return(x2_coeff, x_coeff, con_coeff)

def multi_linear_regression(x1, x2, y):
    x1_squared = x1 ** 2
    x2_squared = x2**2
    x1y = x1 * y
    x2y = y * x2
    x1x2 = x1*x2
    N = len(x1)
    ans_list = []
    left_matrix = [[sum(x1_squared), sum(x1x2), sum(x1)], [sum(x1x2), sum(x2_squared), sum(x2)], [sum(x1), sum(x2), N]]
    right_matrix = [[sum(x1y)], [sum(x2y)], [sum(y)]]
    inv_mat_left = np.linalg.inv(left_matrix)
    solution = np.dot(inv_mat_left, right_matrix)
    x1_coeff, x2_coeff, con_coeff = solution[0], solution[1], solution[2]
    for i in range(N):
        ans_list.append(sum( (y[i] - (x1_coeff*x1[i] + x2_coeff*x2[i] + con_coeff))**2))
    return(ans_list)


#CLEAN UP THE DATAFRAME
house_df = pd.read_csv('data.csv', delim_whitespace=False)
house_df['price'] = house_df['price'].round()
house_df = house_df[(house_df['price']>0) & (house_df['price']<5000000)]

training_set, test_set = get_random(house_df, 50)


x, b = regression_equ(training_set['sqft_living'], training_set['sqft_lot'])

y = []
for item in training_set['sqft_living']:
    y.append(item * int(x) + int(b))

# print(len(y), len(training_set['sqft_living']))

fig, ax = plt.subplots(nrows=1, ncols=3)
training_set.plot.scatter('sqft_living', 'price', ax=ax[0])
training_set.plot('sqft_living', y, ax=ax[0])
training_set.plot.scatter('sqft_lot', 'price', ax=ax[1])


plt.show()
print(x, b)


