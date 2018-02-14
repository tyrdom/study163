import numpy as np


#min (x.T A x + x.T b)
def grad_method (A,b,x0,epsilon):
    x=x0
    i=0
    grad=2*(np.matmul(A,x)+b)
    while( np.linalg.norm(grad,ord=2) > epsilon ):
        i=i+1
        t=np.linalg.norm(grad,ord=2)**2 /( 2 * np.matmul( np.matmul(grad.T,A), grad )) #最优步长值
        x= x-t * grad
        grad = 2 * (np.matmul(A , x) + b)
        fun_val = np.matmul(np.matmul(x.T , A) , x) + np.matmul(b.T,x)
        print("%r grad:%r value:%r"%(i,np.linalg.norm(grad,ord=2),fun_val))





