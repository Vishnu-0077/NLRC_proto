import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def gls_neuron_gen(X,b,ss): #skew binary method
    '''
    Docstring for gls_neuron_gen
    
    :param X: the single box of a 2d input, (single float value between 0 and 1)
    :param b: threshold
    :param ss: symbolic sequence where the binary values are appended
    '''
    if X>=b:
        ss.append(1)
        return (1-X)/(1-b)
    ss.append(0)
    return X/b

def iterate_gls_neuron(x,y,eps):
    '''
    Docstring for iterate_gls_neuron
    
    :param x: total 2d input array/list
    :param y: target values which will serve as reference state
    :param eps: neighbourhood region
    '''
    b = 0.5
    ttss = np.zeros((len(x),len(x[0])))
    max_iter = 1000
    for i in range(len(x)):
        for j in range(len(x[0])):
            ss = []
            val = x[i][j]
            N=0
            while ((y-eps)>val or val>(y+eps)) and N<max_iter:
                N+=1
                val = gls_neuron_gen(val,b,ss)
            h = ss.count(1)
            ttss[i][j] = h/N if N>0 else 0
    return ttss

def ttss_allocations(x,y_set,eps):
    # allocate all the ttss values by iterating through the y_targer vzlues
    final_ttss = []
    for i in range(len(y_set)):
        ttss_i = iterate_gls_neuron(x,y_set[i],eps)
        final_ttss.append(ttss_i)
    return np.array(final_ttss)

    

def mean_respresentation(final_ttss,y_lst,m,n):
    final_M = np.zeros((len(y_lst),n))
    for i in range(len(y_lst)):
        for j in range(n):
            final_M[i][j] = np.mean(final_ttss[i,:,j],axis=0)
    return final_M

def model_train(X_train,y_train): #train
    pipe = Pipeline([('svc',SVC())])
    pipe.fit(X_train, y_train)
    return pipe

X,y = load_iris(return_X_y=True)
mm = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
(a,b) = X_train.shape
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
y_target = np.linspace(0.1,0.9,10) #fixed intervals with not much differene b/w them
def main():
    
    eps = 0.1
    
    train_ttss = ttss_allocations(X_train,y_target,eps).transpose(1,0,2)
    final_M = mean_respresentation(train_ttss,y_target,len(y_target),b)
    train_ttss = train_ttss.reshape(len(X_train),len(y_target)*b)
    
    test_ttss = ttss_allocations(X_test,y_target,eps).transpose(1,0,2).reshape(len(X_test),len(y_target)*b)
    model = model_train(train_ttss,y_train)
    y_pred = model.predict(test_ttss)
    return accuracy_score(y_test,y_pred)

def main_2():
    epsilon = [0.2,0.1,0.05,0.02,0.01]
    results = {}
    for eps in epsilon:
        train_ttss = ttss_allocations(X_train,y_target,eps).transpose(1,0,2).reshape(len(X_train),len(y_target)*b)
        test_ttss = ttss_allocations(X_test,y_target,eps).transpose(1,0,2).reshape(len(X_test),len(y_target)*b)

        model = model_train(train_ttss,y_train)
        y_pred = model.predict(test_ttss)
        results[eps] = round(accuracy_score(y_test,y_pred),2)
    return results

print(main())