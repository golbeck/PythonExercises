####################################################################################
####################################################################################
import numpy as np
from scipy import special
import os

####################################################################################
####################################################################################
def NN_classifier(eps_penalty,alpha,X_in,Y,activation_fn,output_fn):
    #negative log likelihood function for classification
    #eps_penalty: parameter for L2 regularization penalty term
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #X_in: np.array of inputs; dim (n,p+1) (each of the n rows is a separate observation, p is the number of features)
    #Y: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function
    
    #number of observations
    n=X_in.shape[0]
    #number of hidden layers
    n_layers=len(alpha)-1
    layer=0
    #add bias vector to hidden layer; dim (batch_size,M[0]+1)
    Z=np.column_stack((np.ones(n),activation_fn(np.dot(X_in,alpha[layer]))))

    for layer in range(1,n_layers):
        #add bias vector to hidden layer 
        Z=np.column_stack((np.ones(n),activation_fn(np.dot(Z,alpha[layer]))))

    ##############################################################################################
    #output of last hidden layer
    layer=n_layers
    #linear combination of hidden layer outputs
    T=np.dot()
    #outputs
    g=output_fn(Z,alpha[layer])
    return g
####################################################################################
####################################################################################
def cost_fn(eps_penalty,alpha,X_in,Y,activation_fn,output_fn):
    #negative log likelihood function for classification
    #eps_penalty: parameter for L2 regularization penalty term
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #X_in: np.array of inputs; dim (n,p+1) (each of the n rows is a separate observation, p is the number of features)
    #Y: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function
    
    #number of observations
    n=X_in.shape[0]
    #number of hidden layers
    n_layers=len(alpha)-1
    layer=0
    #add bias vector to hidden layer; dim (batch_size,M[0]+1)
    Z=np.column_stack((np.ones(n),activation_fn(np.dot(X_in,alpha[layer]))))

    for layer in range(1,n_layers):
        #add bias vector to hidden layer 
        Z=np.column_stack((np.ones(n),activation_fn(np.dot(Z,alpha[layer]))))

    ##############################################################################################
    #output of last hidden layer
    layer=n_layers
    #outputs
    g=output_fn(np.dot(Z,alpha[layer]))

    #L2 regularization penalty term
    L2=0.5*eps_penalty*np.array([(alpha[i]**2).sum() for i in range(len(alpha))]).sum()
    #generate negative log likelihood function
    f=-(np.log(g)*Y).sum()+L2
    return f
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
def confusion_matrix_multi(y_out,y,n_class):
    #compute logistic function
    m=y.shape[0]
    tempTP=0
    tempTN=0
    tempFP=0
    tempFN=0
    
    #rows: actual class label
    #cols: predicted class label
    CF=np.zeros((n_class,n_class))
    
    for i in range(m):
        if(y_out[i]==y[i]):
            CF[y[i]-1,y[i]-1]+=1
        else:
            CF[y[i]-1,y_out[i]-1]+=1
            
    return CF        
####################################################################################
####################################################################################


##################################################################################################
##################################################################################################
####################################################################################
#GET INPUTS
####################################################################################
####################################################################################
####################################################################################

#load data
pwd_temp=os.getcwd()
# dir1='/home/sgolbeck/workspace/PythonExercises/NeuralNets'
dir1='/home/golbeck/Workspace/PythonExercises/NeuralNets'
if pwd_temp!=dir1:
    os.chdir(dir1)
dir1=dir1+'/data' 
import scipy.io as sio
dat=sio.loadmat(dir1+'/ex3data1.mat')
X_in=np.array(dat['X'])
y_mat=np.array(dat['y'])
#create a matrix with 0-1 class labels
K=y_mat.max()
Y_in=np.zeros((y_mat.shape[0],K))
for i in range(y_mat.shape[0]):
    Y_in[i,y_mat[i]-1]=1
p=X_in.shape[1]


#randomly permuate the features and outputs using the same shuffle for each epoch
rng_state = np.random.get_state()  
np.random.shuffle(X_in)
np.random.set_state(rng_state)
np.random.shuffle(Y_in)      
rng_state = np.random.get_state()  

#total number of obs in data set
n=X_in.shape[0]

####################################################################################
####################################################################################
####################################################################################
#TRAINING PARAMETERS
####################################################################################
#####################################################################################
#####################################################################################
#number of observations used in each gradient update
batch_size=500
#number of complete iterations through training data set
epochs=20
#hyperparameters
eps_alpha=0.1
eps_penalty=0.01
mom_param=np.array([0.50,0.99,20.0])
gamma=0.04
#number of neurons in each hidden layer
M=np.array([5,5])
#number of hidden layers
n_layers=M.shape[0]
#append the number of output units to M
M=np.append(M,K)
#list of network parameters
alpha=[]
weight_L=-4*np.sqrt(6./(p+M[0]))
weight_H=4*np.sqrt(6./(p+M[0]))
#input parameters for first layer activation function
alpha.append(np.random.uniform(low=weight_L,high=weight_H,size=(p+1)*M[0]).reshape(p+1,M[0]))
#parameters for inputs to all other hidden layer activation functions and the output function (K units)
for layer in range(1,n_layers+1):
    alpha.append(np.random.uniform(low=weight_L,high=weight_H,size=(M[layer-1]+1)*M[layer]).reshape(M[layer-1]+1,M[layer]))


#number of observations
n=X_in.shape[0]
#number of features
p=X_in.shape[1]
#number of mini-batches
n_mini_batch=n/batch_size
#add bias vector to inputs
X_in=np.column_stack((np.ones(n),X_in))
#number of rows in the dependent variable
n_Y=Y_in.shape[0]
#number of classes
K=Y_in.shape[1]
#check if X and Y have the same number of observations

##################################################################################################
##################################################################################################
##################################################################################################
#TRAIN NETWORK
##################################################################################################
##################################################################################################
##################################################################################################
#initialize iterator
epoch_iter=0
#save rng state to apply the same permutation to both X and Y
rng_state = np.random.get_state()
#randomly permuate the features and outputs using the same shuffle for each epoch
np.random.shuffle(X_in)
np.random.set_state(rng_state)
np.random.shuffle(Y_in)        

#iterate through the entire observation set, updating the gradient via mini-batches
# for batch_number in range(n_mini_batch):
batch_number=0
#initialize gradient list using range function
grad=range(n_layers+1)
grad_act=[]
Z=[]
##############################################################################################
#input to first layer
layer=0
#observations used to update alpha, beta        
obs_index=range(batch_number*batch_size,(batch_number+1)*batch_size)
#linear combination of inputs
T=np.dot(X_in[obs_index,:],alpha[layer])
#gradient of activation function with respect to T
grad_act.append(grad_sigmoid(T))
#add bias vector to hidden layer; dim (batch_size,M[0]+1)
Z.append(np.column_stack((np.ones(batch_size),special.expit(T))))

for layer in range(1,n_layers):
    #linear combination of inputs
    T=np.dot(Z[layer-1],alpha[layer])
    #gradient of activation function with respect to T
    grad_act.append(grad_sigmoid(T))
    #add bias vector to hidden layer 
    Z.append(np.column_stack((np.ones(batch_size),special.expit(T))))


print Z[1][0,5], Z[0][0,5]
##############################################################################################
#output of last hidden layer
layer=n_layers
#linear combination of hidden layer outputs
T=np.dot(Z[layer-1],alpha[layer])
#gradient of output function
grad_output=grad_softmax(T)
#outputs
g=softmax_fn(T)

#Y/g*grad_output: dimensions (n_batch,K,K)
B_old=np.einsum('ij,ij,ijk->ik',Y_in[obs_index,:],1/g,grad_output)
#sum over observations and save gradient
grad[layer]=np.einsum('ij,ik->jk',Z[layer-1],B_old)

for layer in range(n_layers,1,-1):
    B_old=np.einsum('ij,kj,ik->ik',B_old,alpha[layer][range(1,M[layer-1]+1),:],grad_act[layer-1])
    grad[layer-1]=np.einsum('ij,ik->jk',Z[layer-2],B_old)
    # grad[layer-1]=np.einsum('ij,ik->kj',B_old,Z[layer-2])

layer=1
B_old=np.einsum('ij,kj,ik->ik',B_old,alpha[layer][range(1,M[layer-1]+1),:],grad_act[layer-1])
grad[layer-1]=np.einsum('ij,ik->jk',X_in[obs_index,:],B_old)

[grad[i].shape for i in range(n_layers+1)]
[alpha[i].shape for i in range(n_layers+1)]


grad_new=grad


##############################################################################################
#initialize gradient list
grad=[]
##############################################################################################
#input to first layer
layer=0
#observations used to update alpha, beta        
obs_index=range(batch_number*batch_size,(batch_number+1)*batch_size)
#linear combination of inputs
T=np.dot(X_in[obs_index,:],alpha[layer])
#gradient of activation function with respect to T
grad_act=grad_sigmoid(T)
#add bias vector to hidden layer; dim (batch_size,M[0]+1)
Z=np.column_stack((np.ones(batch_size),special.expit(T)))
print Z[0,5]
#dim (batch_size,M[1],p+1,M[0])
grad.append(np.array([[[alpha[layer+1][q+1,s]*grad_act[:,q]*X_in[obs_index,r] for s in range(M[layer+1])] 
    for r in range(p+1)] for q in range(M[layer])]).transpose())

for layer in range(1,n_layers):
    #linear combination of inputs
    T=np.dot(Z,alpha[layer])
    #gradient of activation function with respect to T
    grad_act=grad_sigmoid(T)
    #dim (batch_size,K,M[layer-1]+1,M[layer])
    grad.append(np.array([[[alpha[layer+1][q+1,s]*grad_act[:,q]*Z[:,r] for s in range(M[layer+1])] 
        for r in range(M[layer-1]+1)] for q in range(M[layer])]).transpose())
    #gradient update for input parameters
    grad[0]=np.array([[[[alpha[layer+1][s+1,j]*grad_act[:,s]*grad[0][:,s,r,q] for j in range(M[layer+1])] 
        for s in range(M[layer])] for r in range(p+1)] for q in range(M[0])]).transpose().sum(2)
    if layer>1:
        for h in range(1,layer):
            grad[h]=np.array([[[[alpha[layer+1][s+1,j]*grad_act[:,s]*grad[h][:,s,r,q] for j in range(M[layer+1])] 
                for s in range(M[layer])] for r in range(M[h-1]+1)] for q in range(M[h])]).transpose().sum(2)

    #add bias vector to hidden layer 
    Z=np.column_stack((np.ones(batch_size),special.expit(T)))
    print Z[0,5]

##############################################################################################
#output of last hidden layer
layer=n_layers
#linear combination of hidden layer outputs
T=np.dot(Z,alpha[layer])
#outputs
g=softmax_fn(T)

#compute 
temp1=(Y_in[obs_index,:]/g)
grad_output=grad_softmax(T)
#Y*(1/g)*dg/dT
C0=np.array([[temp1[:,k]*grad_output[:,k,j] for k in range(K)] for j in range(K)]).transpose().sum(1)
#gradient update for current layer
grad.append(np.array([[C0[:,q]*Z[:,r] for r in range(M[layer-1]+1)] for q in range(K)]).transpose().sum(0))

#gradient update for previous layers (and average over batch)
grad[0]=np.array([[[C0[:,j]*grad[0][:,j,r,q] for j in range(K)] 
    for r in range(p+1)] for q in range(M[0])]).transpose().sum(1).sum(0)

for h in range(1,n_layers):
    grad[h]=np.array([[[C0[:,j]*grad[h][:,j,r,q] for j in range(K)] 
        for r in range(M[h-1]+1)] for q in range(M[h])]).transpose().sum(1).sum(0)