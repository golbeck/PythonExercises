####################################################################################
####################################################################################
import numpy as np
from scipy import special
import os

####################################################################################
####################################################################################
def softmax_fn(T):
    #T: linear combination of hidden layer outputs
    #determine how many columns (classes) for the output
    K=T.shape[1]
    #compute the denominator of the softmax function
    temp=np.exp(T).sum(axis=1)
    #iterate over each class to generate dividend matrix
    g=np.array([np.exp(T[:,j]) for j in range(K)]).transpose()
    #divide each column by the sum
    g= g/temp[:,None]
    #return a matrix of dim (n,K), where "n" is the number of observations
    return g

####################################################################################
####################################################################################
def grad_sigmoid(X):
    #X: linear combination of features
    g=special.expit(X)*(1-special.expit(X))
    return g
####################################################################################
####################################################################################
def grad_softmax(T):
    #T: linear combination of hidden layer outputs, dim (n,K)
    #number of observations in T
    n=T.shape[0]
    #determine how many columns (classes) for the output
    K=T.shape[1]
    #compute the denominator
    temp=np.exp(T).sum(axis=1)
    #iterate over each class to generate dividend matrix
    g=np.array([np.exp(T[:,j]) for j in range(K)]).transpose()
    #divide each column by the sum
    g= g/temp[:,None]
    #generate a dim (n,K,K) matrix for derivatives of the softmax function with respect to T[i,l]
    f1=np.array([[g[:,i] for i in range(K)]*g[:,j] for j in range(K)])
    A=np.eye(K)
    f2=np.array([[g[:,i]*A[i,j] for i in range(K)] for j in range(K)])
    #matrix of derivatives: for each observation i, 
    #the derivative of the kth column of the softmax function
    #with respect to the jth output
    g=f2-f1
    return g.transpose()
####################################################################################
####################################################################################


n=30
p=5
X_in=np.random.normal(size=n*p).reshape(n,p)
K=3
Y_in=np.random.normal(size=n*K).reshape(n,K)
M=np.array([10,12])
n_layers=M.shape[0]
alpha=[]
alpha.append(np.random.normal(size=(p+1)*M[0]).reshape(p+1,M[0]))
alpha.append(np.random.normal(size=(M[0]+1)*M[1]).reshape(M[0]+1,M[1]))
alpha.append(np.random.normal(size=(M[1]+1)*K).reshape(M[1]+1,K))

rng_state = np.random.get_state()  
#start at the given random state supplied by the user (for debugging)
np.random.set_state(rng_state)
#randomly permuate the features and outputs using the same shuffle for each epoch
np.random.shuffle(X_in)
np.random.set_state(rng_state)
np.random.shuffle(Y_in)      
rng_state = np.random.get_state()  

#dimension of data
#number of rows (observations)
n=X_in.shape[0]   
#number of observations per each step through an epoch
batch_number=0
batch_size=5
n_mini_batch=n/batch_size
#number of input features 
p=X_in.shape[1]
#add bias vector to inputs
X=np.column_stack((np.ones(n),np.copy(X_in)))
#number of rows in the dependent variable
n_Y=Y_in.shape[0]
#number of classes
K=Y_in.shape[1]

#randomly permuate the features and outputs using the same shuffle for each epoch
np.random.shuffle(X)
np.random.set_state(rng_state)
np.random.shuffle(Y_in)        
#update rng state after shuffling in order to apply the same permutation to both X_in and Y
rng_state = np.random.get_state()


#initialize gradient list
grad=[]
##############################################################################################
#input to first layer
layer=0
#observations used to update alpha, beta
obs_index=range(batch_number*batch_size,(batch_number+1)*batch_size)
#linear combination of inputs
T=np.dot(X[obs_index,:],alpha[layer][:,:])
#gradient of activation function with respect to T
grad_act=grad_sigmoid(T)
#hidden layer outputs
Z=special.expit(T)
#add bias vector to hidden layer; dim (batch_size,M[0]+1)
Z=np.column_stack((np.ones(batch_size),np.copy(Z)))
#dim (batch_size,M[1],p+1,M[0])
C=np.array([[[alpha[layer+1][q+1,s]*grad_act[:,q]*X[obs_index,r] for s in range(M[layer+1])] for r in range(p+1)] for q in range(M[layer])]).transpose()
#gradient update
grad.append(C)

##############################################################################################
#2nd layer
layer=n_layers-1
#linear combination of inputs
T=np.dot(Z,alpha[layer][:,:])
#gradient of activation function with respect to T
grad_act=grad_sigmoid(T)
#dim (batch_size,K,M[layer-1]+1,M[layer])
C=np.array([[[alpha[layer+1][q+1,s]*grad_act[:,q]*Z[:,r] for s in range(K)] for r in range(M[layer-1]+1)] for q in range(M[layer])]).transpose()
#gradient update for current layer
grad.append(C)
#gradient update for earlier layers
C=np.array([[[[alpha[layer+1][s+1,j]*grad_act[:,s]*grad[0][:,s,r,q] for j in range(K)] for s in range(M[layer])] for r in range(p+1)] for q in range(M[layer-1])]).transpose()
grad[0]=C.sum(2)

#hidden layer outputs
Z=special.expit(T)
#add bias vector to hidden layer 
Z=np.column_stack((np.ones(batch_size),np.copy(Z)))

##############################################################################################
#output of last hidden layer
layer=2
#linear combination of hidden layer outputs
T=np.dot(Z,alpha[layer][:,:])
#outputs
g=softmax_fn(T)

#compute 
temp1=(Y_in[obs_index,:]/g)
grad_output=grad_softmax(T)
#Y*(1/g)*dg/dT
C0=np.array([[temp1[:,k]*grad_output[:,k,j] for k in range(K)] for j in range(K)]).transpose()
C0=C0.sum(1)
#gradient update for current layer
C=np.array([[C0[:,q]*Z[:,r] for r in range(M[layer-1]+1)] for q in range(K)]).transpose()
grad.append(C)

#gradient update for previous layers
C=np.array([[[C0[:,j]*grad[0][:,j,r,q] for j in range(K)] for r in range(p+1)] for q in range(M[layer-2])]).transpose()
grad[0]=C.sum(1)

C=np.array([[[C0[:,j]*grad[1][:,j,r,q] for j in range(K)] for r in range(M[layer-2]+1)] for q in range(M[layer-1])]).transpose()
grad[1]=C.sum(1)


for i in range(len(M)+1):
    alpha[i]=alpha[i]-eps_alpha*(-grad[i].sum(0))

