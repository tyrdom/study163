import numpy as np
import ch1

A=np.array([[1,0],[0,2]])
b=np.array([[0],[0]])
x0=np.array([[1],[2]])
epsilon = 1e-5

ch1.grad_method(A, b, x0, epsilon)
