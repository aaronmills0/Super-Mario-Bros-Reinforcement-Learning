import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import statistics
from math import sqrt
from ast import literal_eval

fig_x = 8
fig_y = 8

def csv_to_lists(path):
    pass


def get_mean(*args):
    pass

def standard_dev(vector):
  mean = sum(vector)/len(vector)
  sd = 0
  for i in range(len(vector)):
    sd += (vector[i] - mean)**2
  sd /= len(vector)
  sd = sqrt(sd)
  return sd

df = pd.read_csv("data/dqn_rewards.csv")
print(df)
del df[df.columns[0]]

arr = [[0 for i in range(1000)] for j in range(10)]

for i in range(10):
   
   for j in range(1000):
      
      string = df.iloc[i]

      arr[i][j] = sum(literal_eval(string.iloc[j]))

averaged = [0 for i in range(1000)]

for j in range(1000):
   avg = 0
   for i in range(10):
      avg += arr[i][j]
   
   averaged[j] = avg/10 

print(averaged)

x = [i+1 for i in range(1000)]

X_Y_Spline = make_interp_spline(x, averaged)

X_ = np.linspace(min(x), max(x), 300)
Y_ = X_Y_Spline(X_)

plt.figure(figsize=(fig_x, fig_y))

plt.show()
