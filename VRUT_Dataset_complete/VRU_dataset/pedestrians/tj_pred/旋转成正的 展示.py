import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
from pandas import  DataFrame
def Srotate(angle,valuex,valuey,pointx,pointy):
  valuex = np.array(valuex)
  valuey = np.array(valuey)
  sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
  sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
  return sRotatex,sRotatey


file_dir ='../moving/'
angle =math.radians(50)

def panding(a,b):
    # a = np.array(a)
    # b = np.array(b)
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
    path = '../moving_zheng/'
    dir_s = []

    for files in os.listdir(file_dir):
        # df =pd.read_csv(files)
        dir_s.append(file_dir + files)
    # sum = len(dir_s)

    for i in dir_s:
        labels = []
        k=i[i.rfind('/'):]
        df = pd.read_csv(i)
        df = df.to_numpy()
        df = df[:, 2:4]
        train_x = []
        train_y = []
        # print(k)
        # a = df[:, 0]
        # b = df[:, 1]
        for a, b in df:
            a, b = Srotate(angle, a, b, 0, 0)
            a_label, b_label = panding(a, b)
            train_x.append(a)
            train_y.append(b)
            labels.append((a,b,a_label, b_label))
        data = DataFrame(labels)
        data.to_csv(os.path.join(path + str(k) + '.csv'), sep=',', header=None, index=None)
        plt.scatter(train_x, train_y,  marker="*", s=0.5, color="red", ls="-",)
    plt.show()

# sPointx ,sPointy = Srotate(math.radians(45),pointx,pointy,0,0)
get_csv_labels(file_dir)

















# def rotatecordiate(angle,rect):
#     angle=angle*math.pi/180
#     n=1600
#     m=1200
#     def onepoint(x,y):
#         # X = x*math.cos(angle) - y*math.sin(angle)-0.5*n*math.cos(angle)+0.5*m*math.sin(angle)+0.5*n
#         # Y = y*math.cos(angle) + x*math.sin(angle)-0.5*n*math.sin(angle)-0.5*m*math.cos(angle)+0.5*m
#         X = x * math.cos(angle) - y * math.sin(angle) - 0.5 * n * math.cos(angle) + 0.5 * m * math.sin(angle) + 0.5 * n
#         Y = y * math.cos(angle) + x * math.sin(angle) - 0.5 * n * math.sin(angle) - 0.5 * m * math.cos(angle) + 0.5 * m
#         return [int(X),int(Y)]
#     newrect=[]
#     for i in range(4):
#         point=onepoint(rect[i*2],rect[i*2+1])
#         newrect.extend(point)
#     newrect.extend([1])
#     print(newrect)
#     return newrect
# rotatecordiate(30,)