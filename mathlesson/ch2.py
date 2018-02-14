import numpy as np
import tensorflow as tf
import ch2_eg_lib as c2l
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X_data =np.array( np.loadtxt("ch2data.csv",delimiter=',',usecols=(0,1),unpack=True)).T
Y_data =np.array( [np.loadtxt("ch2data.csv",delimiter=',',usecols=(3),unpack=True)]).T

X_zs=c2l.z_score(X_data)
Y_zs=c2l.z_score(Y_data)
print(X_zs,Y_zs)
# fig=plt.figure()
# ax= Axes3D(fig)
# ax.plot_surface(np.arange(-4, 4, 0.25), np.arange(-4, 4, 0.25), Y_data, rstride=1, cstride=1, cmap='rainbow')
learning_rate = 0.5

X =tf.placeholder(tf.float32,[None,2])
Y =tf.placeholder(tf.float32,[None,1])

batch =256
steps = X_data.shape[0]//batch-1

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

Layer1 = add_layer(X,2,1024,activation_function=tf.nn.relu)

Layer2 = add_layer(Layer1,1024,512,activation_function=tf.nn.relu)
Layer3 = add_layer(Layer2,512,256,activation_function=tf.nn.relu)
Layer4 = add_layer(Layer3,256,128,activation_function=tf.nn.relu)

Pred = add_layer(Layer4,128,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y-Pred)))

Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    for i in range(0, 1000):
        i_step =i%steps
        batch_X = X_zs[i_step * batch:((i_step + 1) * batch - 1), :]
        batch_Y = Y_zs[i_step * batch:((i_step + 1) * batch - 1), :]
        l,_=sess.run([loss,Optimizer],feed_dict= {X:batch_X,Y:batch_Y})
        print("Epoch {0}: {1}".format(i, l ))