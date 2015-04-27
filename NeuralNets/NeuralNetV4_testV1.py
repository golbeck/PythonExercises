####################################################################################
####################################################################################
import numpy as np
from scipy import special
import os

####################################################################################
####################################################################################
def NN_classifier(alpha,beta,X_in,activation_fn,output_fn):
    #class prediction for a single hidden layer neural network classifier
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #beta: np.array of weights for hidden layer; dim (M+1,K)
    #X_in: np.array of inputs; dim (n,p) (each of the n rows is a separate observation, p is the number of features)
    #activation_fn: activation function
    #output_fn: output function


    #dimension of data
    #number of rows (observations)
    n=X_in.shape[0]   
    #number of input features 
    p=X_in.shape[1]
    #add bias vector to inputs
    X=np.column_stack((np.ones(n),np.copy(X_in)))
    Z=activation_fn(np.dot(X,alpha))
    #add bias vector to hidden layer 
    Z=np.column_stack((np.ones(n),np.copy(Z)))
    #linear combination of hidden layer outputs
    T=np.dot(Z,beta)
    #outputs
    g=output_fn(T)
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
    #linear combination of inputs
    T=np.dot(X_in,alpha[layer][:,:])
    #hidden layer outputs
    Z=activation_fn(T)
    #add bias vector to hidden layer; dim (batch_size,M[0]+1)
    Z=np.column_stack((np.ones(n),Z))

    for layer in range(1,n_layers):
        #linear combination of inputs
        T=np.dot(Z,alpha[layer][:,:])
        #hidden layer outputs
        Z=activation_fn(T)
        #add bias vector to hidden layer 
        Z=np.column_stack((np.ones(n),Z))

    ##############################################################################################
    #output of last hidden layer
    layer=n_layers
    #linear combination of hidden layer outputs
    T=np.dot(Z,alpha[layer][:,:])
    #outputs
    g=output_fn(T)

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

# #preprocessing
# from sklearn import preprocessing
# X_in=preprocessing.scale(X_in)

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
eps_alpha=0.01
eps_penalty=0.01
# n=30
# p=5
# X_in=np.random.normal(size=n*p).reshape(n,p)
# K=3
# Y_in=np.random.normal(size=n*K).reshape(n,K)
#number of neurons in each hidden layer
M=np.array([10,10])
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

alpha_init=np.copy(alpha)
# #input parameters for first layer activation function
# alpha.append(np.random.normal(size=(p+1)*M[0]).reshape(p+1,M[0]))
# #parameters for inputs to all other hidden layer activation functions and the output function (K units)
# for layer in range(1,n_layers+1):
#     alpha.append(np.random.normal(size=(M[layer-1]+1)*M[layer]).reshape(M[layer-1]+1,M[layer]))

rng_state = np.random.get_state()  
#start at the given random state supplied by the user (for debugging)
np.random.set_state(rng_state)
#randomly permuate the features and outputs using the same shuffle for each epoch
np.random.shuffle(X_in)
np.random.set_state(rng_state)
np.random.shuffle(Y_in)      

#number of observations used in each gradient update
batch_size=500
#number of complete iterations through training data set
epochs=200
n_mini_batch=n/batch_size
#add bias vector to inputs
X_in=np.column_stack((np.ones(n),X_in))
#number of rows in the dependent variable
n_Y=Y_in.shape[0]
#number of classes
K=Y_in.shape[1]

print [(alpha[h].min(), alpha[h].max()) for h in range(len(alpha))]

##################################################################################################
##################################################################################################
##################################################################################################
#TRAIN NETWORK
##################################################################################################
##################################################################################################
##################################################################################################
#initialize iterator
epoch_iter=0
while(epoch_iter<epochs):
    print "epoch iteration %s" %epoch_iter
    #save rng state to apply the same permutation to both X and Y
    rng_state = np.random.get_state()
    #randomly permuate the features and outputs using the same shuffle for each epoch
    np.random.shuffle(X_in)
    np.random.set_state(rng_state)
    np.random.shuffle(Y_in)        

    #iterate through the entire observation set, updating the gradient via mini-batches
    for batch_number in range(n_mini_batch):
        print "batch number %s" %batch_number
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
        #hidden layer outputs
        Z=special.expit(T)
        #add bias vector to hidden layer; dim (batch_size,M[0]+1)
        Z=np.column_stack((np.ones(batch_size),Z))
        #dim (batch_size,M[1],p+1,M[0])
        grad.append(np.array([[[alpha[layer+1][q+1,s]*grad_act[:,q]*X_in[obs_index,r] for s in range(M[layer+1])] 
            for r in range(p+1)] for q in range(M[layer])]).transpose())

        for layer in range(1,n_layers):
            print "layer %s" %layer
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

            #hidden layer outputs
            Z=special.expit(T)
            #add bias vector to hidden layer 
            Z=np.column_stack((np.ones(batch_size),Z))

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

        #gradient update for previous layers
        grad[0]=np.array([[[C0[:,j]*grad[0][:,j,r,q] for j in range(K)] 
            for r in range(p+1)] for q in range(M[0])]).transpose().sum(1).sum(0)

        for h in range(1,n_layers):
            grad[h]=np.array([[[C0[:,j]*grad[h][:,j,r,q] for j in range(K)] 
                for r in range(M[h-1]+1)] for q in range(M[h])]).transpose().sum(1).sum(0)

        for i in range(n_layers+1):
            #L2 regularization term
            M_temp=alpha[i].shape[0]
            M_range=range(1,M_temp)
            alpha[i][M_range,:]=alpha[i][M_range,:]-eps_alpha*(-grad[i][M_range,:]+eps_penalty*alpha[i][M_range,:])
            #no regularization for bias parameters
            alpha[i][0,:]=alpha[i][0,:]-eps_alpha*(-grad[i][0,:])

        print [(alpha[h].min(), alpha[h].max()) for h in range(len(alpha))]

    #update epoch iteration
    epoch_iter+=1
    ##############################################################################################
    ##############################################################################################
    #predict classes and compute accuracy rate
    layer=0
    #linear combination of inputs
    T=np.dot(X_in,alpha[layer])
    #hidden layer outputs
    Z=special.expit(T)
    #add bias vector to hidden layer; dim (batch_size,M[0]+1)
    Z=np.column_stack((np.ones(n),Z))

    for layer in range(1,n_layers):
        #linear combination of inputs
        T=np.dot(Z,alpha[layer])
        #hidden layer outputs
        Z=special.expit(T)
        #add bias vector to hidden layer 
        Z=np.column_stack((np.ones(n),Z))

    #output of last hidden layer
    layer=n_layers
    #linear combination of hidden layer outputs
    T=np.dot(Z,alpha[layer])
    #convert to class number
    y_pred=softmax_fn(T).argmax(1)+1
    #convert class matrix to array of class labels (starting at 1) for use in confusion matrix
    y_dat=Y_in.argmax(1)+1
    CF=confusion_matrix_multi(y_pred,y_dat,K)
    accuracy=CF.diagonal().sum(0)/n
    print "observation (20,200,2000): %s" %y_dat[[19,199,1999]]
    print "prediction (20,200,2000): %s" %y_pred[[19,199,1999]]
    print softmax_fn(T)[[19,199,1999],:]
    print "accuracy rate %s" %accuracy
    cost=cost_fn(eps_penalty,alpha,X_in,Y_in,special.expit,softmax_fn)
    print cost


##################################################################################################
##################################################################################################
##################################################################################################
#TEST ALPHA GRADIENT
##################################################################################################
##################################################################################################
##################################################################################################
# batch_number=0
# batch_size=100
# #observations used to update alpha, beta
# obs_index=range(batch_number*batch_size,(batch_number+1)*batch_size)
# #test negative log likelihood function and alpha and beta gradients
# #test alpha gradient
# eps_alpha=0.01
# eps_penalty=0.01
# alpha_grad_test=[]
# for h in range(len(alpha)):
#     print h
#     alpha_grad_test.append(np.zeros(alpha[h].shape))
#     alpha_u=np.copy(alpha)
#     alpha_d=np.copy(alpha)
#     for i in range(alpha[h].shape[0]):
#         for j in range(alpha[h].shape[1]):
#             alpha_u[h]=np.copy(alpha[h])
#             alpha_u[h][i,j]=alpha_u[h][i,j]+eps_alpha
#             alpha_d[h]=np.copy(alpha[h])
#             alpha_d[h][i,j]=alpha_d[h][i,j]-eps_alpha
#             alpha_grad_test[h][i,j]=(cost_fn(eps_penalty,alpha_u,X_in[obs_index,:],Y_in[obs_index,:],special.expit,softmax_fn)
#                 -cost_fn(eps_penalty,alpha_d,X_in[obs_index,:],Y_in[obs_index,:],special.expit,softmax_fn))/(2*eps_alpha)
# print alpha_grad_test


# #initialize gradient list
# grad=[]
# ##############################################################################################
# #input to first layer
# layer=0
# batch_number=0
# batch_size=100
# #observations used to update alpha, beta
# obs_index=range(batch_number*batch_size,(batch_number+1)*batch_size)
# #linear combination of inputs
# T=np.dot(X_in[obs_index,:],alpha[layer][:,:])
# #gradient of activation function with respect to T
# grad_act=grad_sigmoid(T)
# #hidden layer outputs
# Z=special.expit(T)
# #add bias vector to hidden layer; dim (batch_size,M[0]+1)
# Z=np.column_stack((np.ones(batch_size),Z))
# #dim (batch_size,M[1],p+1,M[0])
# grad.append(np.array([[[alpha[layer+1][q+1,s]*grad_act[:,q]*X_in[obs_index,r] for s in range(M[layer+1])] 
#     for r in range(p+1)] for q in range(M[layer])]).transpose())

# for layer in range(1,n_layers):
#     print "layer %s" %layer
#     #linear combination of inputs
#     T=np.dot(Z,alpha[layer][:,:])
#     #gradient of activation function with respect to T
#     grad_act=grad_sigmoid(T)
#     #dim (batch_size,K,M[layer-1]+1,M[layer])
#     grad.append(np.array([[[alpha[layer+1][q+1,s]*grad_act[:,q]*Z[:,r] for s in range(M[layer+1])] 
#         for r in range(M[layer-1]+1)] for q in range(M[layer])]).transpose())
#     #gradient update for input parameters
#     grad[0]=np.array([[[[alpha[layer+1][s+1,j]*grad_act[:,s]*grad[0][:,s,r,q] for j in range(M[layer+1])] 
#         for s in range(M[layer])] for r in range(p+1)] for q in range(M[0])]).transpose().sum(2)
#     if layer>1:
#         for h in range(1,layer):
#             grad[h]=np.array([[[[alpha[layer+1][s+1,j]*grad_act[:,s]*grad[h][:,s,r,q] for j in range(M[layer+1])] 
#                 for s in range(M[layer])] for r in range(M[h-1]+1)] for q in range(M[h])]).transpose().sum(2)

#     #hidden layer outputs
#     Z=special.expit(T)
#     #add bias vector to hidden layer 
#     Z=np.column_stack((np.ones(batch_size),Z))

# ##############################################################################################
# #output of last hidden layer
# layer=n_layers
# #linear combination of hidden layer outputs
# T=np.dot(Z,alpha[layer][:,:])
# #outputs
# g=softmax_fn(T)

# #compute 
# temp1=(Y_in[obs_index,:]/g)
# grad_output=grad_softmax(T)
# #Y*(1/g)*dg/dT
# C0=np.array([[temp1[:,k]*grad_output[:,k,j] for k in range(K)] for j in range(K)]).transpose().sum(1)
# #gradient update for current layer
# grad.append(np.array([[C0[:,q]*Z[:,r] for r in range(M[layer-1]+1)] for q in range(K)]).transpose().sum(0))

# #gradient update for previous layers
# grad[0]=np.array([[[C0[:,j]*grad[0][:,j,r,q] for j in range(K)] 
#     for r in range(p+1)] for q in range(M[0])]).transpose().sum(1).sum(0)
# grad[0]-=eps_penalty*alpha[0]

# for h in range(1,n_layers):
#     grad[h]=np.array([[[C0[:,j]*grad[h][:,j,r,q] for j in range(K)] 
#         for r in range(M[h-1]+1)] for q in range(M[h])]).transpose().sum(1).sum(0)
#     grad[h]-=eps_penalty*alpha[h]