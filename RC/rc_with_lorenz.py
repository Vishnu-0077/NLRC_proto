import numpy as np
from scipy import linalg
import pandas as pd

dataa = pd.read_csv('/home/vishnu/Downloads/traindata.csv')
dataa = dataa[1000:65000]

data = dataa.to_numpy()
y_data = dataa.iloc[1: ,0].to_numpy()


insize = 3
outsize = 1
ressize = 100
init_len = 100
train_length = 2000
test_length = 100
a=0.3

np.random.seed(42)
W_in = np.random.rand(ressize,insize+1) - 0.5
W = np.random.rand(ressize,ressize) - 0.5

spectral_radius = np.max(np.abs(linalg.eigvals(W)))
target_radius = 0.95
W = W*target_radius/spectral_radius

x = np.zeros(ressize).reshape(-1,1)
x_history = np.zeros((1+insize+ressize,train_length-init_len))

for i in range(train_length):
    u = data[i].reshape(-1,1)
    x = x*(1-a)+a*np.tanh(np.dot(W,x)+np.dot(W_in,np.vstack((1,u))))
    if i>=init_len:
        x_history[:,i-init_len] = np.vstack((1,u,x.reshape(-1,1)))[:,0]

print(f'shape of x_history is {x_history.shape}')

yt = y_data[init_len:train_length].reshape(-1,1).T
reg = 1e-8
w_out = np.dot(np.dot(yt,x_history.T),linalg.inv(np.dot(x_history,x_history.T)+np.eye(ressize+insize+1)*reg))
Y = np.zeros((outsize,test_length))

print(f'shape of w_out is {w_out.shape}')

u = data[train_length].reshape(-1,1)
for i in range(test_length):
    x = x*(1-a)+a*np.tanh(np.dot(W,x)+np.dot(W_in,np.vstack((1,u))))
    y = np.dot(w_out,np.vstack((1,u,x.reshape(-1,1 ))))
    Y[:,i] = y
    u = np.vstack((y,data[train_length+i+1,1],data[train_length+i+1,2]))

sum=0
for i in range(test_length):
    sum += (y_data[train_length+i]-Y[0,i])**2
print(sum/test_length)
    


        



