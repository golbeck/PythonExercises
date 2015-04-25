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


def MLP_stoch_gradV0(epochs,batch_size,eps_alpha,eps_penalty,alpha,M,X_in,Y_in,activation_fn,output_fn,grad_activation_fn,grad_output_fn):
    #epochs: maximum number of iterations through the entire data set for gradient descent
    #batch_size: number of observations used to update alpha and beta
    #eps_alpha: alpha gradient multiplier (assumed to be same for all alpha)
    #eps_penalty: L2 regularization penalty term parameter
    #alpha: list of np.array of weights 
    #M: array of number of neurons for each hidden layer, including the output layer
    #X_in: np.array of inputs; dim (n,p) (each of the n rows is a separate observation, p is the number of features)
    #Y_in: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function
    #grad_activation_fn: gradient of activation function
    #grad_output_fn: gradient of output function

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
    if(n!=n_Y):
        print "number of rows in X and Y are not the same"
        return -9999.

    print alpha[0][1,1], alpha[1][2,2]

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

    return alpha


##################################################################################################
##################################################################################################
####################################################################################
#GET INPUTS
####################################################################################
####################################################################################
####################################################################################

#load data
pwd_temp=os.getcwd()
dir1='/home/sgolbeck/workspace/PythonExercises/NeuralNets'
# dir1='/home/golbeck/Workspace/PythonExercises/NeuralNets'
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
eps_alpha=0.01
eps_penalty=1.00
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
weight_L=-0.7
weight_H=0.7
#input parameters for first layer activation function
alpha.append(np.random.uniform(low=weight_L,high=weight_H,size=(p+1)*M[0]).reshape(p+1,M[0]))
#parameters for inputs to all other hidden layer activation functions and the output function (K units)
for layer in range(1,n_layers+1):
    alpha.append(np.random.uniform(low=weight_L,high=weight_H,size=(M[layer-1]+1)*M[layer]).reshape(M[layer-1]+1,M[layer]))

# #input parameters for first layer activation function
# alpha.append(np.random.normal(size=(p+1)*M[0]).reshape(p+1,M[0]))
# #parameters for inputs to all other hidden layer activation functions and the output function (K units)
# for layer in range(1,n_layers+1):
#     alpha.append(np.random.normal(size=(M[layer-1]+1)*M[layer]).reshape(M[layer-1]+1,M[layer]))

#number of observations used in each gradient update
batch_size=500
#number of complete iterations through training data set
epochs=5

parameters=MLP_stoch_gradV0(epochs,batch_size,eps_alpha,eps_penalty,alpha,M,X_in,Y_in,special.expit,softmax_fn,grad_sigmoid,grad_softmax)