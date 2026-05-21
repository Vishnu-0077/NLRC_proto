import numpy as np
import random
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class nlfea:
    def __init__(self,b=0.5,n=10,test_len=10,reg=1e-8):
        self.b = b
        self.n = n
        self.test_len = test_len
        self.reg = reg

    def ss_to_binary(self,x,thres):
        return (np.array(x)>thres).astype(int)

    def firing_rate(self,x):
        b = self.b
        c = 0
        for v in x:
            if v>b:
                c+=1
        return c/len(x)
    
    def variance(self,x):
        if len(x) ==0:
            return 0
        return np.var(x)
    
    def energy(self,x):
        if len(x) ==0:
            return 0
        return np.mean(np.square(x))
    
    def entropy(self,x):
        b = self.b
        if len(x) ==0:
            return 0
        x = self.ss_to_binary(x,b)
        p = np.count_nonzero(x)/len(x)
        eps = 1e-10
        p = np.clip(p,eps,1-eps)
        return -(p*np.log2(p)) - ((1-p)*np.log2(1-p))
    
    def neuron_gen(self,x):
        b = self.b
        neigh = 1e-8
        x = np.clip(x,neigh,1-neigh)

        if x>=b:
            return (1-x)/(1-b)
        return x/b
    
    def neuron_iterator(self,u):
        n = self.n
        X = np.zeros(n)
        X[0] = u.item()
        for j in range(n-1):
            X[j+1] = self.neuron_gen(X[j])
        return X
    
    def build_features(self,data):
        n = self.n
        history_x = np.zeros((6,len(data)))
        for i in range(len(data)):
            u = data[i]
            x = self.neuron_iterator(u)
            ene = self.energy(x)
            ent = self.entropy(x)
            var = self.variance(x)
            fr = self.firing_rate(x)
            history_x[:,i] = np.vstack((1,u,ene,ent,var,fr)).flatten()
        return history_x
    
    def fit(self,data,y_data):
        yt = y_data.reshape(-1,1)
        yt = yt.T
        reg = self.reg
        history_x = self.build_features(data)
        self.w_out = np.dot(np.dot(yt,history_x.T),linalg.inv(np.dot(history_x,history_x.T)+np.eye(6)*reg))
        return self
    
    def predict(self,u):
        test_len= self.test_len
        n = self.n
        Y = np.zeros((1,test_len))
        for i in range(test_len):
            X = np.zeros(n)
            X[0] = u
            for j in range(n-1):
                X[j+1] = self.neuron_gen(X[j])
            ene = self.energy(X)
            ent = self.entropy(X)
            var = self.variance(X)
            fr = self.firing_rate(X)
            y = np.dot(self.w_out,np.vstack((1,u,ene,ent,var,fr)).flatten())
            Y[:,i] = y
            u = y.item()
        return Y



def test_data(a = 3.95, i = 0.5422,length = 1500):
    x = np.zeros(length)
    x[0] = i
    for t in range(length-1):
        x[t+1] = a*x[t]*(1-x[t])
    return x

def mackey_glass(length=2000, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0):
    x = np.zeros(length)
    
    # initial condition (important!)
    x[:tau+1] = 1.2  
    
    for t in range(tau, length-1):
        x_tau = x[t - tau]
        x[t+1] = x[t] + dt * (beta * x_tau / (1 + x_tau**n) - gamma * x[t])
    
    return x[100:]

data = mackey_glass()
train_len = 1000
test_len = 10

p = 0
push = train_len+p

X_train = data[:train_len]
X_test = data[push:push+test_len]

y_train = data[1:train_len+1]
y_test = data[1+push:test_len+1+push]

scalar = MinMaxScaler(feature_range=(0,1))
X_train = scalar.fit_transform(X_train.reshape(-1,1))
y_train = scalar.transform(y_train.reshape(-1,1))
X_test = scalar.transform(X_test.reshape(-1,1))
y_test = y_test.flatten()

model = nlfea(n=10,test_len=test_len,b=0.5)
model.fit(X_train,y_train)
y_pred = model.predict(X_test[0].item())
y_pred = scalar.inverse_transform(y_pred.reshape(-1,1)).flatten()

mse = mean_squared_error(y_test,y_pred)
print(f'mse is {mse}')

plt.scatter(y_pred[:-1],y_pred[1:])
plt.show()