import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator,RegressorMixin
import math

class NGRC(BaseEstimator,RegressorMixin):
    def __init__(self,k=4,s=1,deg=3,reg=10e-7,test_len=10):
        self.k = k
        self.s = s
        self.deg = deg
        self.reg = reg
        self.test_len = test_len
    
    def build_features(self,data):
        k = self.k
        train_len = len(data)
        poly = PolynomialFeatures(degree=self.deg)
        poly.fit(np.zeros((1,k)))
        size_featues = math.comb(k+self.deg,self.deg)
        total_featues = np.zeros((size_featues,train_len-k))

        for t in range(k,train_len):
            O_lin = np.zeros((k,1))
            for i in range(k):
                O_lin[i,0] = (data[t-(i*self.s),0])
            
            poly_features = poly.transform(O_lin.reshape(1,-1))
            total_featues[:,t-k] = poly_features.flatten()
        return total_featues
    
    def fit(self,data,y_data):
        train_len = len(data)
        yt = y_data[self.k:train_len]
        yt = yt.T
        total_features = self.build_features(data)
        self.f_mean = np.mean(total_features, axis=1, keepdims=True)
        self.f_std = np.std(total_features, axis=1, keepdims=True) + 1e-8 #to prevent division by 0
        total_features = (total_features - self.f_mean) / self.f_std
        feature_len = total_features.shape[0]

        reg = self.reg
        self.w_out = np.dot(np.dot(yt,total_features.T),linalg.inv(np.dot(total_features,total_features.T)+np.eye(feature_len)*reg))
        return self
    
    def predict(self,test_data):
        test_len = self.test_len
        self.poly = PolynomialFeatures(degree=self.deg)
        self.poly.fit(np.zeros((1,self.k)))
        Y = np.zeros((1,test_len))

        for t in range(test_len):
            O_lin = np.zeros((self.k,1))
            for i in range(self.k):
                O_lin[i,0] = test_data[(self.k-1)-(i*self.s),0]
            
            poly_features = self.poly.transform(O_lin.reshape(1,-1))
            x = poly_features.reshape(-1,1)
            x = (x - self.f_mean) / self.f_std
            x = x.flatten()
            y = np.dot(self.w_out,x).reshape(-1,1)
            Y[:,t] = y
            test_data = np.vstack((test_data[1:],y))

        return Y.flatten()
    

dataa = pd.read_excel('/home/vishnu/Downloads/Mackey-Glass Time Series(taw17).xlsx')
dataa = dataa.drop('Unnamed: 0',axis = 1)
dataa = dataa[100:1050]

o_data = dataa['t'].to_numpy().reshape(-1,1)
o_y_data = dataa['t+1'].to_numpy()
k=10


mean = np.mean(o_data)
std = np.std(o_data)
y_mean = np.mean(o_y_data)
y_std = np.std(o_y_data)

data = (o_data - mean) / std
y_data = (o_y_data - y_mean) / y_std

model = NGRC(k=k,deg=3,reg=1e-4)
model.fit(data[:700],y_data[:700])

test_data = np.array(data[700-k:700])
y_test = np.array(y_data[700:710])

y_pred = model.predict(test_data)
print(y_pred)

y_pred = y_pred * y_std + y_mean    

y_data_orig = o_y_data[700:710]

mse = mean_squared_error(y_data_orig,y_pred)
print(mse)
