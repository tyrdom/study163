import numpy as np
import ch2_eg_lib


data=np.load("ch2_data.npz")
x_data=data["arr_0"]
y_data=data["arr_1"]
x_zs=ch2_eg_lib.z_score(x_data) #zscore归一化
y_zs=ch2_eg_lib.z_score(y_data)
# print(x_zs,y_zs)
np.random.seed(9) #随机种子
params=ch2_eg_lib.initparams() #初始化参数
eta =5
m= x_data.shape[0]

for i in range(0,m): #循环100次
    # xi=np.array([x_zs[i]])
    # yi=np.array([y_zs[i]])
    params,loss = ch2_eg_lib.train_loop(x_zs,y_zs,params,eta) #训练参数
    print(loss)
print(params['W1'],
      params['W2'],
      params['W3'],
      params['B1'],
      params['B2'],
      params['B3'])
# result
#  [[-0.48962038  0.00175194]
#  [-0.00422788 -0.36618807]
#  [-0.35764881 -0.28145701]]
#  [[-0.08154142 -0.25174518 -0.41560398]
#  [-0.15437849 -0.33311583  0.37879221]]
#  [[ 0.4515835  -0.46074807]
#  [ 0.19954682  0.07311573]
#  [ 0.39870043  0.16729628]]
