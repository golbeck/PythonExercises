from numpy.core.umath_tests import matrix_multiply
import numpy as np
A=np.random.normal(size=5*10*20*30).reshape(5,10,20,30)
B=np.random.normal(size=5*10*30*40).reshape(5,10,30,40)
C = matrix_multiply(A, B)


A=np.random.normal(size=5*10).reshape(5,10)
B=np.random.normal(size=5*10*20).reshape(5,10,20)
C = matrix_multiply(A, B)


#C2=np.tensordot(A,B,axes=([0,3],[0,2]))
C4=np.tensordot(A,B,axes=([3],[2]))



alpha=np.random.normal(size=25*5).reshape(25,5)
grad_act=np.random.normal(size=7*24).reshape(7,24)
C=np.dot(grad_act,alpha[range(1,25),:])

A=np.random.normal(size=100*10).reshape(100,10)
B=np.random.normal(size=100*40).reshape(100,40)
C5=np.tensordot(A,B,axes=[])


M=np.array([100,200,10])
alpha=np.random.normal
alpha.append([np.random.normal(size=M[l]*M[l-1]).reshape(M[l],M[l-1]) for l in range(1,3)])

C=np.array([[[alpha[layer+1][q+1,s]*grad_act[:,q]*X_in[obs_index,r] for s in range(M[layer+1])] 
                for r in range(p+1)] for q in range(M[layer])]).transpose()