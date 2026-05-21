import numpy as np
import random
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

'''
what does this code does??

1. it uses the gls neuron binary trace values as the feature extractor of the time series input of
of the previous data.
2. we do not keep a neighbour hood region instead we have a specific number of iterations that the
gls neuron can take
3. the made feature training matrix is used for the calculation of the w_out
4. next the name w_out is used for the calcualtion of the recursive value of the y,
(recusrive mean, the previous output is taken as the next input)
5. done. the remaining features are then quotedd in the code'''

class nlrc:
    def __init__(self, b=0.5,reg=1e-8,n=10,test_len = 10):
        self.b = b
        self.reg = reg
        self.n = n
        self.test_len = test_len

    def gls_neuron_gen(self,x):
        """
        the gls neuron binary trace values as the feature extractor of the time series input of
        of the previous data.
        """
        b = self.b
        neigh = 1e-8
        x = np.clip(x,neigh,1-neigh)

        if x>=b:
            return (1-x)/(1-b)
        return x/b
    
    def build_features(self,data):
        '''
        building the features of the specific time point using the gls neuron binary trace values
        '''
        n = self.n
        history_x = np.zeros((1+n,len(data)))
        for i in range(len(data)):
            u = data[i] #setting the first value
            X = np.zeros(n)
            X[0] = u.item() #item becoz, u might be ---> array[u] instead of float(u)
            for j in range(n-1):
                X[j+1] = self.gls_neuron_gen(X[j]) #iterating through the gls neuron
            history_x[:,i] = np.vstack((1,X.reshape(-1,1))).flatten()
        
        return history_x #printing the it with shape(n+1,train_len)
    
    def fit(self,data,y_data):
        yt = y_data.reshape(-1,1)
        yt = yt.T
        reg = self.reg
        history_x = self.build_features(data)
        self.w_out = np.dot(np.dot(yt,history_x.T),linalg.inv(np.dot(history_x,history_x.T)+np.eye(self.n+1)*reg))
        return self
    
    def predict(self,test_u):
        test_len = self.test_len
        n = self.n
        Y = np.zeros((1,test_len))
        u = test_u
        for i in range(test_len):
            X = np.zeros(n)
            X[0] = u
            for j in range(n-1):
                X[j+1] = self.gls_neuron_gen(X[j])
            y = np.dot(self.w_out,np.vstack((1,X.reshape(-1,1))).flatten())
            Y[:,i] = y
            u = y.item() #y is an array, so we send only the float values further
        return Y.flatten()

def test_data(a = 3.95, i = 0.5422,length = 1000):
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
    
    return x


data = mackey_glass()
train_len = 100
test_len = 100

p = 0 #distance between and of training data and start of test data
push = train_len+p #for easy indexing

train_X = data[:train_len] #getting the train len
test_X = data[push:push+test_len]


train_y = data[1:train_len+1]
test_y = data[push+1:push+test_len+1]

scalar = MinMaxScaler(feature_range=(0,1)) #scaling between the 0-1
train_X = scalar.fit_transform(train_X.reshape(-1,1))
train_y = scalar.transform(train_y.reshape(-1,1))
test_X = scalar.transform(test_X.reshape(-1,1))
test_y = test_y.flatten()


model = nlrc(test_len=test_len,n=10)
model.fit(train_X,train_y)
y_pred = model.predict(test_X[0].item())
y_pred = scalar.inverse_transform(y_pred.reshape(-1,1)).flatten()

mse = mean_squared_error(test_y,y_pred)
print(f'mse is {mse}')

plt.plot(test_y)
plt.plot(y_pred)
plt.legend(['original','predicted'])
plt.show()
