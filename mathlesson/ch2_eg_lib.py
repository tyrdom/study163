import numpy as np

def sigmoid (x):
    return 1/(1+np.exp(-x))

def z_score(x):
    x_mean=np.mean(x,axis=0)
    x_norm=np.linalg.norm(x-x_mean,ord=2,axis=0)**0.5
    x_z_score =(x-x_mean) /x_norm
    return x_z_score

def initparams():
    w1 = np.random.rand(3, 2) - 0.5
    w2 = np.random.rand(2, 3) - 0.5
    w3 = np.random.rand(3, 2) - 0.5
    params = {"W1": w1, "W2": w2, "W3": w3}
    return  params

def train_loop (X,Y,params,eta):
    W1=params['W1']
    W2=params['W2']
    W3=params['W3']
    m=X.shape[0]
    net2 = np.dot(X,W1) #shape(M,2)
    hidden1 = sigmoid(net2) #shape(M,2)
    net3 = np.dot(hidden1, W2) #shape(M,3)
    hidden2 = sigmoid(net3) #shape(M,3)
    net4 = np.dot(hidden2, W3) #shape(M,2)
    O = sigmoid(net4)  #shape(M,2)

    loss =1/2/m * np.linalg.norm((Y-O),ord=2)

    dnet4 =1/m*(Y-O) * O * (1-O) #shape(m,2)
    dW3 =1/m*np.dot(hidden2.T, dnet4) #shape(3,2)
    dhidden2 = np.dot(dnet4, W3.T) #shape(m,3)
    dnet3 = dhidden2 * hidden2 * (1-hidden2) #shape(m,3)
    dW2 =1/m*np.dot(hidden1.T, dnet3) #shape(2,3)
    dhidden1 = np.dot(dnet3, W2.T) #shape(m,2)
    dnet2 = dhidden1 * hidden1 * (1-hidden1) #shape(m,2)
    dW1 =1/m * np.dot(X.T,dnet2) #shape (3,2)

    W1=W1-eta*dW1
    W2=W2-eta*dW2
    W3=W3-eta*dW3

    params_out = {"W1":W1,"W2":W2,"W3":W3}
    return params_out,loss




