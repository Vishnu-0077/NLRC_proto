import numpy as np
from scipy import linalg
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def mackey_glass(length=2000, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0):
    x = np.zeros(length)
    
    # initial condition (important!)
    x[:tau+1] = 1.2  
    
    for t in range(tau, length-1):
        x_tau = x[t - tau]
        x[t+1] = x[t] + dt * (beta * x_tau / (1 + x_tau**n) - gamma * x[t])
    
    return x

def test_data(a = 3.95, i = 0.5422,length = 2000):
    x = np.zeros(length)
    x[0] = i
    for t in range(length-1):
        x[t+1] = a*x[t]*(1-x[t])
    return x

data = mackey_glass()
y_data = data[1:]

insize = 1
outsize = 1
ressize = 300
init_len = 50
train_length = 1000
test_length = 100
a=0.3
def tentmap(x):
    b = 0.5
    return np.where(x<b,x/b,(1-x)/(1-b))
            
np.random.seed(42)
W_in = np.random.rand(ressize,insize+1) - 0.5
W = np.random.rand(ressize,ressize) - 0.5

spectral_radius = np.max(np.abs(linalg.eigvals(W)))
target_radius = 0.95
W = W*target_radius/spectral_radius

x = np.zeros(ressize).reshape(-1,1)
x_history = np.zeros((1+insize+ressize,train_length-init_len))


for i in range(train_length):
    u = data[i]
    r = np.dot(W,x)+np.dot(W_in,np.vstack((1,u)))
    r = 1/(1+np.exp(-r)) #here iam trying to make them bound between (0 and 1)
    r = tentmap(r)
    x = x*(1-a)+a*r
    if i>=init_len:
        x_history[:,i-init_len] = np.vstack((1,u,x.reshape(-1,1)))[:,0]

print(f'shape of x_history is {x_history.shape}')

yt = y_data[init_len:train_length].reshape(-1,1).T
reg = 1e-8
w_out = np.dot(np.dot(yt,x_history.T),linalg.inv(np.dot(x_history,x_history.T)+np.eye(ressize+insize+1)*reg))
Y = np.zeros((outsize,test_length))

print(f'shape of w_out is {w_out.shape}')

u = data[train_length]
for i in range(test_length):
    r = np.dot(W,x)+np.dot(W_in,np.vstack((1,u)))
    r = 1/(1+np.exp(-r))
    r = tentmap(r)
    x = x*(1-a)+a*r
    y = np.dot(w_out,np.vstack((1,u,x.reshape(-1,1 ))))
    Y[:,i] = y
    u = y

sum=0
print(f'shape of Y is {Y.shape}')
for i in range(test_length):
    sum += (y_data[train_length+i]-Y[0,i])**2
print(sum/test_length)