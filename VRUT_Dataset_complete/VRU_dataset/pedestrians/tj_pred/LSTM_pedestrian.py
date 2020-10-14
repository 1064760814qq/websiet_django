import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
learning_rate = 1e-3
#data_csv = pd.read_csv('C:/Users/10647/Desktop/VRU/VRUT_Dataset_complete/VRU_dataset/pedestrians/moving_zuobiao/7_24.csv.txt')
data_csv = pd.read_csv('C:/Users/10647/Desktop/VRU/VRUT_Dataset_complete/VRU_dataset/pedestrians/moving_zheng/7_24.csv.csv')
data_csv = data_csv.dropna()
dataset = data_csv.to_numpy()
# dataset = dataset[:, 2:4]
# train_x = []
# train_y = []
# for i in dataset:
#     train_x.append(i[0])
#     train_y.append(i[1])
# train_x = np.asarray(train_x)
# train_y = np.asarray(train_y)
#
# plt.plot(train_x,train_y,'ro',label='train data')
# plt.legend()
# plt.show()
dataset = dataset.astype('float64')

#数据预处理
# for i in range(2):
#     dataset[:, i] = (dataset[:, i]-dataset[:, i].min())/(dataset[:,i].max()-dataset[:,i].min())
# data_test = dataset[-5:]
def create_dataset(dataset,times_x,times_y):
    datax = []
    datay = []
    for i in range(len(dataset)-times_x-times_y):
        every_x_in=dataset[i:(i+times_x)]

        datax.append(every_x_in)
        datay.append(dataset[i+times_x:i+times_x+times_y])
    return np.asarray(datax),np.asarray(datay)

data_x, data_y = create_dataset(dataset, 5, 3)

data_x = data_x.astype(float)
data_x = data_x.reshape(-1,5,4)
# print(data_x)
# exit()
data_y = data_y.astype(float)
data_y = data_y.reshape(-1, 3, 4)
# print(data_y)
data_x = torch.from_numpy(data_x)
data_y = torch.from_numpy(data_y)
# print(data_y)

class Net(nn.Module):
    def __init__(self, input_size, hidden_units, output_size=3, num_layers=4):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_units, num_layers)  #
        self.cov1d = nn.Conv1d(in_channels=20, out_channels=4, kernel_size=3, stride=1)
    def forward(self, x):
        # (390,5,4)->(390,5,20)
        x, (hn, cn) = self.rnn(x)
        #(390,5,20)->(390,20,5)
        x = torch.transpose(x,2,1)
        #(390,20,5)->(390,4,3)
        x = self.cov1d(x)
        #(390,2,3)->（390，3，2）
        return torch.transpose(x,1,2)

net = Net(4, 20)
loss_fun =nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

for i  in range(5000):
    var_x = Variable(data_x)
    var_x = torch.tensor(var_x, dtype=torch.float32)
    var_y = Variable(data_y)
    var_y = torch.tensor(var_y, dtype=torch.float32)
    out = net(var_x)
    loss =loss_fun(out,var_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i + 1) % 100 == 0:  # 每 10 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(i + 1, loss.item()))
        print(out)
net = net.eval()
flag = 1
if flag==0:
    data_xx, data_yy = create_dataset(dataset,5,0)
    data_test = torch.tensor(data_xx, dtype=torch.float32)
    data_test = Variable(data_test)
    pred = net(data_test)
    pred = pred.data.numpy()
    pred = np.array(pred)
    # data_test = torch.unsqueeze(data_test,dim=0)
    pred = pred[-1]
    print(pred)
# else:
# data_csv2 = pd.read_csv('C:/Users/10647/Desktop/VRU/VRUT_Dataset_complete/VRU_dataset/pedestrians/moving_zheng/31_3.csv.csv',header=None).to_numpy()
# # print(data_csv2)
# # exit()
# dataset2 = data_csv2.astype('float64')
# data_x, data_y = create_dataset(dataset2, 5, 0)
# data_x = data_x.astype(float)
# data_x = data_x.reshape(-1, 5, 4)
# data_test = torch.tensor(data_x, dtype=torch.float32)
# data_test = Variable(data_test)
# pred = net(data_test)
# pred = pred.data.numpy()
# pred = np.array(pred)
# pred = pred[-1]
# print(pred)





# pred = pred * 158.45














