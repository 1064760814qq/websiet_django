import os
import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

file_dir ='../moving/'


# def get_csv_sums(file_dir):
#     sum = 0
#     for files in os.listdir(file_dir):
#         sum += 1
#     return sum


def panding(a,b):
   if a < 0:
    a_label = int(a)-1
   else:
       a_label = int(a)
   if b < 0:
       b_label = int(b)-1
   else:
       b_label = int(b)
   return a_label, b_label


def get_csv_labels(file_dir):
    path = '../moving_zuobiao/'
    dir_s = []

    for files in os.listdir(file_dir):
        # df =pd.read_csv(files)
        dir_s.append(file_dir + files)
    sum = len(dir_s)

    for i in dir_s:
        labels = []
        k=i[i.rfind('/'):]
        df = pd.read_csv(i)
        df = df.to_numpy()
        df = df[:, 2:4]
        train_x = []
        train_y = []
        print(k)
        a = df[:, 0]
        b = df[:, 1]
        train_x.append(a)
        train_y.append(b)
        plt.scatter(train_x, train_y,  marker="*", s=0.5, color="red", ls="-",)
    plt.show()


        # print(df)
        # for a, b in df:
        #     a_label, b_label = panding(a, b)
        #     labels.append((a_label, b_label))
        # data = DataFrame(labels)
        # data.to_csv(os.path.join(path + str(k) +'.txt'), sep=',', header=None, index=None)


get_csv_labels(file_dir)


