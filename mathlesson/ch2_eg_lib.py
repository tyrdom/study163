import numpy as np

def sigmoid (x):
    return 1/(1+np.exp(-x))

def z_score(x):
    x_mean=np.mean(x,axis=0)
    x_norm=np.linalg.norm(x-x_mean,ord=2,axis=0)**0.5
    x_z_score =(x-x_mean) /x_norm
    return x_z_score

def initparams():
    w1 = np.random.rand(3, 2)-0.5
    w2 = np.random.rand(2, 3)-0.5
    w3 = np.random.rand(3, 2)-0.5
    b1 = np.zeros((1,2))+0.1
    b2 =np.zeros((1,3))+0.1
    b3 =np.zeros((1,2))+0.1
    params = {"W1": w1, "W2": w2, "W3": w3,"B1":b1,"B2":b2,"B3":b3}
    return params

def train_loop (X,Y,params,eta):
    W1=params['W1']
    W2=params['W2']
    W3=params['W3']
    B1=params['B1']
    B2=params['B2']
    B3=params['B3']
    m=X.shape[0]
    Y1 = np.dot(X,W1)+B1 #shape(M,2)
    A1 = sigmoid(Y1) #shape(M,2)
    Y2 = np.dot(A1, W2)+B2 #shape(M,3)
    A2 = sigmoid(Y2) #shape(M,3)
    Y3 = np.dot(A2, W3)+B3 #shape(M,2)
    Out = sigmoid(Y3)  #shape(M,2)
    E= Out-Y
    loss =np.sum(E*E)

    dY3 = E * Out * (1-Out) #shape(m,2)
    dW3 = 1/m* np.dot(A2.T, dY3) #shape(3,2)
    dB3=1/m*np.sum(dY3, axis=0,keepdims=True)#shape(1,2)
    dA2 = np.dot(dY3, W3.T) #shape(m,3)
    dY2 = dA2 * A2 * (1-A2) #shape(m,3)
    dW2 = 1/m*np.dot(A1.T, dY2) #shape(2,3)
    dB2 = 1/m*np.sum(dY2,axis=0,keepdims=True)#shape(1,3)
    dA1 = np.dot(dY2, W2.T) #shape(m,2)
    dY1 = dA1 * A1 * (1-A1) #shape(m,2)
    dW1 = 1/m * np.dot(X.T,dY1) #shape (3,2)
    dB1 = 1/m*np.sum(dY1,axis=0,keepdims=True)#shape(1,2)

    W1=W1-eta*dW1
    W2=W2-eta*dW2
    W3=W3-eta*dW3
    B1=B1-eta*dB1
    B2 = B2 - eta * dB2
    B3 = B3 - eta * dB3
    params_out = {"W1":W1,"W2":W2,"W3":W3,"B1":B1,"B2":B2,"B3":B3}
    return params_out,loss




