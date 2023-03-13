import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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





#CLEAN UP THE DATAFRAME
house_df = pd.read_csv('data.csv', delim_whitespace=False)
house_df['price'] = house_df['price'].round()


training_set, test_set = get_random(house_df, 60)





