####################################################################################
####################################################################################
import numpy as np
from scipy import special
import os
####################################################################################
####################################################################################
def NN_fit_grad_descent_V0(k_max,eps_alpha,eps_beta,alpha,beta,X_in,Y,activation_fn,output_fn,grad_activation_fn,grad_output_fn):
    #gradient descent for fitting a single hidden layer neural network classifier
    #k_max: maximum number of iterations for gradient descent
    #eps_alpha: alpha gradient multiplier (assumed to be same for all alpha)
    #eps_beta: beta gradient multiplier (assumed to be same for all beta)
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #beta: np.array of weights for hidden layer; dim (M+1,K)
    #X_in: np.array of inputs; dim (n,p) (each of the n rows is a separate observation, p is the number of features)
    #Y: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function
    #grad_activation_fn: gradient of activation function
    #grad_output_fn: gradient of output function

    #dimension of data
    #number of rows (observations)
    n=X_in.shape[0]   
    #number of input features 
    p=X_in.shape[1]
    #add bias vector to inputs
    X=np.column_stack((np.ones(n),np.copy(X_in)))
    #number of rows in the dependent variable
    n_Y=Y.shape[0]
    #number of classes
    K=Y.shape[1]
    #check if X and Y have the same number of observations
    if(n!=n_Y):
        print "number of rows in X and Y are not the same"
        return -9999.
    #initialize iterator
    k_iter=0
    while(k_iter<k_max):
        #hidden layer outputs
        Z=activation_fn(np.dot(X,alpha))
        #add bias vector to hidden layer 
        Z=np.column_stack((np.ones(n),np.copy(Z)))
        #number of hidden layers (including bias)
        M=Z.shape[1]

        #linear combination of hidden layer outputs
        T=np.dot(Z,beta)
        #outputs
        g=output_fn(T)

        #compute 
        temp1=(Y/g)
        temp2=grad_output_fn(T)
        #Y*(1/g)*dg/dT
        C=np.array([[temp1[:,k]*temp2[:,k,j] for k in range(K)] for j in range(K)]).transpose()

        #sum over output classes
        D1=C.sum(axis=1)
        #sum[Y*(1/g)*dg/dT]*Z
        #dim (n,M,K): (obs index,hidden layer index including bias,class index)
        cost_grad_beta=-np.array([[Z[:,j] for j in range(M)]*D1[:,k] for k in range(K)]).transpose()

        #sum[sum[Y*(1/g)*dg/dT]*beta]
        D3=np.array([[D1[i,:]*beta[j,:] for j in range(1,M)] for i in range(n)]).sum(axis=2)
        #sum[sum[Y*(1/g)*dg/dT]*beta]*dZ/d(X*alpha)
        grad_act=grad_activation_fn(np.dot(X,alpha))
        D4=D3*grad_act
        #dim (n,p+1,M)
        cost_grad_alpha=-np.array([[D4[:,j]*X[:,k] for k in range(p+1)] for j in range(M-1)]).transpose()
        k_iter+=1

        #update beta
        beta=beta-eps_beta*cost_grad_beta.sum(0)
        #update alpha
        alpha=alpha-eps_alpha*cost_grad_alpha.sum(0)

    return [alpha,beta]
####################################################################################
####################################################################################
def NN_fit_grad_descent(epochs,steps_epoch,eps_alpha,eps_beta,alpha,beta,X_in,Y,activation_fn,output_fn,grad_activation_fn,grad_output_fn):
    #gradient descent for fitting a single hidden layer neural network classifier

    #epochs: maximum number of iterations for gradient descent
    #steps_epoch: number of updates to alpha, beta within each epoch
    #eps_alpha: alpha gradient multiplier (assumed to be same for all alpha)
    #eps_beta: beta gradient multiplier (assumed to be same for all beta)
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #beta: np.array of weights for hidden layer; dim (M+1,K)
    #X_in: np.array of inputs; dim (n,p) (each of the n rows is a separate observation, p is the number of features)
    #Y: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function
    #grad_activation_fn: gradient of activation function
    #grad_output_fn: gradient of output function

    #convert class matrix to array of class labels (starting at 1) for use in confusion matrix
    y_in=Y.argmax(1)+1
    #dimension of data
    #number of rows (observations)
    n=X_in.shape[0]   
    #number of observations per each step through an epoch
    n_obs=n/steps_epoch
    #number of input features 
    p=X_in.shape[1]
    #add bias vector to inputs
    X=np.column_stack((np.ones(n),np.copy(X_in)))
    #number of rows in the dependent variable
    n_Y=Y.shape[0]
    #number of classes
    K=Y.shape[1]
    #check if X and Y have the same number of observations
    if(n!=n_Y):
        print "number of rows in X and Y are not the same"
        return -9999.
    #initialize iterator
    epoch_iter=0
    while(epoch_iter<epochs):
        for n_step in range(steps_epoch):
            #observations used to update alpha, beta
            obs_index=range(n_step*n_obs,(n_step+1)*n_obs)
            #hidden layer outputs
            Z=activation_fn(np.dot(X[obs_index,:],alpha))
            #add bias vector to hidden layer 
            Z=np.column_stack((np.ones(n_obs),np.copy(Z)))
            #number of hidden layers (including bias)
            M=Z.shape[1]

            #linear combination of hidden layer outputs
            T=np.dot(Z,beta)
            #outputs
            g=output_fn(T)

            #compute 
            temp1=(Y[obs_index,:]/g)
            temp2=grad_output_fn(T)
            #Y*(1/g)*dg/dT
            C=np.array([[temp1[:,k]*temp2[:,k,j] for k in range(K)] for j in range(K)]).transpose()

            #sum over output classes
            D1=C.sum(axis=1)
            #sum[Y*(1/g)*dg/dT]*Z
            #dim (n,M,K): (obs index,hidden layer index including bias,class index)
            cost_grad_beta=-np.array([[Z[:,j] for j in range(M)]*D1[:,k] for k in range(K)]).transpose()

            #sum[sum[Y*(1/g)*dg/dT]*beta]
            D3=np.array([[D1[i,:]*beta[j,:] for j in range(1,M)] for i in range(n_obs)]).sum(axis=2)
            #sum[sum[Y*(1/g)*dg/dT]*beta]*dZ/d(X*alpha)
            grad_act=grad_activation_fn(np.dot(X[obs_index,:],alpha))
            D4=D3*grad_act
            #dim (n,p+1,M)
            cost_grad_alpha=-np.array([[D4[:,j]*X[obs_index,k] for k in range(p+1)] for j in range(M-1)]).transpose()

            #update beta
            beta=beta-eps_beta*cost_grad_beta.sum(0)
            #update alpha
            alpha=alpha-eps_alpha*cost_grad_alpha.sum(0)
        #iterate over the remaining observations if the whole data set has not been pass through
        if(np.max(obs_index)<n-1):
            #observations used to update alpha, beta
            obs_index=range(np.max(obs_index),n)
            #hidden layer outputs
            Z=activation_fn(np.dot(X[obs_index,:],alpha))
            #add bias vector to hidden layer 
            Z=np.column_stack((np.ones(n_obs),np.copy(Z)))
            #number of hidden layers (including bias)
            M=Z.shape[1]

            #linear combination of hidden layer outputs
            T=np.dot(Z,beta)
            #outputs
            g=output_fn(T)

            #compute 
            temp1=(Y[obs_index,:]/g)
            temp2=grad_output_fn(T)
            #Y*(1/g)*dg/dT
            C=np.array([[temp1[:,k]*temp2[:,k,j] for k in range(K)] for j in range(K)]).transpose()

            #sum over output classes
            D1=C.sum(axis=1)
            #sum[Y*(1/g)*dg/dT]*Z
            #dim (n,M,K): (obs index,hidden layer index including bias,class index)
            cost_grad_beta=-np.array([[Z[:,j] for j in range(M)]*D1[:,k] for k in range(K)]).transpose()

            #sum[sum[Y*(1/g)*dg/dT]*beta]
            D3=np.array([[D1[i,:]*beta[j,:] for j in range(1,M)] for i in range(n_obs)]).sum(axis=2)
            #sum[sum[Y*(1/g)*dg/dT]*beta]*dZ/d(X*alpha)
            grad_act=grad_activation_fn(np.dot(X[obs_index,:],alpha))
            D4=D3*grad_act
            #dim (n,p+1,M)
            cost_grad_alpha=-np.array([[D4[:,j]*X[obs_index,k] for k in range(p+1)] for j in range(M-1)]).transpose()

            #update beta
            beta=beta-eps_beta*cost_grad_beta.sum(0)
            #update alpha
            alpha=alpha-eps_alpha*cost_grad_alpha.sum(0)

        #update epoch iterator
        epoch_iter+=1
        #predict classes and compute error rate
        Z=activation_fn(np.dot(X,alpha))
        Z=np.column_stack((np.ones(n),np.copy(Z)))
        T=np.dot(Z,beta)
        #convert to class number
        y_pred=output_fn(T).argmax(1)+1
        CF=confusion_matrix_multi(y_pred,y_in,K)
        error_rate=CF.diagonal().sum(0)/n
        print error_rate

    return [alpha,beta]
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
def cost_fn(alpha,beta,X_in,Y,activation_fn,output_fn):
    #negative log likelihood function for classification
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #beta: np.array of weights for hidden layer; dim (M+1,K)
    #X_in: np.array of inputs; dim (n,p) (each of the n rows is a separate observation, p is the number of features)
    #Y: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function
    

    #dimension of data
    #number of rows (observations)
    n=X_in.shape[0]   
    #number of input features 
    p=X_in.shape[1]
    #add bias vector to inputs
    X=np.column_stack((np.ones(n),np.copy(X_in)))
    #number of rows in the dependent variable
    n_Y=Y.shape[0]
    #number of classes
    K=Y.shape[1]
    #check if X and Y have the same number of observations
    if(n!=n_Y):
        print "number of rows in X and Y are not the same"
        return -9999.
    
    #hidden layer outputs
    Z=activation_fn(np.dot(X,alpha))
    #add bias vector to hidden layer 
    Z=np.column_stack((np.ones(n),np.copy(Z)))
    
    #linear combination of hidden layer outputs
    T=np.dot(Z,beta)
    #outputs
    g=output_fn(T)    
    #generate negative log likelihood function
    f=-(np.log(g)*Y).sum()
    return f
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
def grad_alpha_cost(alpha,beta,X_in,Y,activation_fn,output_fn,grad_activation_fn,grad_output_fn):
    #negative log likelihood function for classification
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #beta: np.array of weights for hidden layer; dim (M+1,K)
    #X_in: np.array of inputs; dim (n,p) (each of the n rows is a separate observation, p is the number of features)
    #Y: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function


    #dimension of data
    #number of rows (observations)
    n=X_in.shape[0]   
    #number of input features 
    p=X_in.shape[1]
    #add bias vector to inputs
    X=np.column_stack((np.ones(n),np.copy(X_in)))
    #number of rows in the dependent variable
    n_Y=Y.shape[0]
    #number of classes
    K=Y.shape[1]
    #check if X and Y have the same number of observations
    if(n!=n_Y):
        print "number of rows in X and Y are not the same"
        return -9999.
    
    #hidden layer outputs
    Z=activation_fn(np.dot(X,alpha))
    #add bias vector to hidden layer 
    Z=np.column_stack((np.ones(n),np.copy(Z)))
    #number of hidden layers (including bias)
    M=Z.shape[1]
    
    #linear combination of hidden layer outputs
    T=np.dot(Z,beta)
    #outputs
    g=output_fn(T)
    
    #compute 
    temp1=(Y/g)
    temp2=grad_output_fn(T)
    #Y*(1/g)*dg/dT
    C=np.array([[temp1[:,k]*temp2[:,k,j] for k in range(K)] for j in range(K)]).transpose()

    #sum over output classes
    D1=C.sum(axis=1)

    #sum[sum[Y*(1/g)*dg/dT]*beta]
    D3=np.array([[D1[i,:]*beta[j,:] for j in range(1,M)] for i in range(n)]).sum(axis=2)
    #sum[sum[Y*(1/g)*dg/dT]*beta]*dZ/d(X*alpha)
    grad_act=grad_activation_fn(np.dot(X,alpha))
    D4=D3*grad_act
    #dim (n,p+1,M)
    cost_grad_alpha=-np.array([[D4[:,j]*X[:,k] for k in range(p+1)] for j in range(M-1)]).transpose()
    return cost_grad_alpha
####################################################################################
####################################################################################
def grad_beta_cost(alpha,beta,X_in,Y,activation_fn,output_fn,grad_activation_fn,grad_output_fn):
    #negative log likelihood function for classification
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #beta: np.array of weights for hidden layer; dim (M+1,K)
    #X_in: np.array of inputs; dim (n,p) (each of the n rows is a separate observation, p is the number of features)
    #Y: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function

    #dimension of data
    #number of rows (observations)
    n=X_in.shape[0]   
    #number of input features 
    p=X_in.shape[1]
    #add bias vector to inputs
    X=np.column_stack((np.ones(n),np.copy(X_in)))
    #number of rows in the dependent variable
    n_Y=Y.shape[0]
    #number of classes
    K=Y.shape[1]
    #check if X and Y have the same number of observations
    if(n!=n_Y):
        print "number of rows in X and Y are not the same"
        return -9999.
    
    #hidden layer outputs
    Z=activation_fn(np.dot(X,alpha))
    #add bias vector to hidden layer 
    Z=np.column_stack((np.ones(n),np.copy(Z)))
    #number of hidden layers (including bias)
    M=Z.shape[1]
    
    #linear combination of hidden layer outputs
    T=np.dot(Z,beta)
    #outputs
    g=output_fn(T)
    
    #compute 
    temp1=(Y/g)
    temp2=grad_output_fn(T)
    #Y*(1/g)*dg/dT
    C=np.array([[temp1[:,k]*temp2[:,k,j] for k in range(K)] for j in range(K)]).transpose()

    #sum over output classes
    D1=C.sum(axis=1)
    #sum[Y*(1/g)*dg/dT]*Z
    #dim (n,K,M): (obs index,class index,hidden layer index)
    cost_grad_beta=-np.array([[Z[:,j] for j in range(M)]*D1[:,k] for k in range(K)]).transpose()
    return cost_grad_beta
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
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
#test output
X_in=np.array([[1,2,3],[4,5,6],[8,9,10],[12,8,20],[8,4,3],[1,2,1]])
p=X_in.shape[1]
Y=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[1,0,0,0,0]])
K=Y.shape[1]
M_hidden=3   #not including bias
#generate random values for the model parameters
#dim (p+1,M-1), (#features including input bias,#hidden layers not including bias)
alpha=np.random.normal(size=(p+1)*M_hidden).reshape(p+1,M_hidden)
#dim (M,K), (#hidden layers including bias hidden layer,#classes)
beta=np.random.normal(size=(M_hidden+1)*K).reshape(M_hidden+1,K)


#test gradient descent
epochs=200
steps_epoch=X_in.shape[0]/3
eps_alpha=0.01
eps_beta=0.01
params=NN_fit_grad_descent(epochs,steps_epoch,eps_alpha,eps_beta,alpha,beta,X_in,Y,special.expit,softmax_fn,grad_sigmoid,grad_softmax)

#test negative log likelihood function and alpha and beta gradients
cost_fn(alpha,beta,X_in,Y,special.expit,softmax_fn)
cost_grad_beta0= grad_beta_cost(alpha,beta,X_in,Y,special.expit,softmax_fn,grad_sigmoid,grad_softmax)
cost_grad_alpha0= grad_alpha_cost(alpha,beta,X_in,Y,special.expit,softmax_fn,grad_sigmoid,grad_softmax)
#test alpha gradient
eps_alpha=0.001
alpha_grad_test=np.zeros(alpha.shape)
for i in range(alpha.shape[0]):
    for j in range(alpha.shape[1]):
        alpha_u=np.copy(alpha)
        alpha_u[i,j]=alpha_u[i,j]+eps_alpha
        alpha_d=np.copy(alpha)
        alpha_d[i,j]=alpha_d[i,j]-eps_alpha
        alpha_grad_test[i,j]=(cost_fn(alpha_u,beta,X_in,Y,special.expit,softmax_fn)
            -cost_fn(alpha_d,beta,X_in,Y,special.expit,softmax_fn))/(2*eps_alpha)
print alpha_grad_test
print cost_grad_alpha0.sum(0)
#test beta gradient
eps_beta=0.001
beta_grad_test=np.zeros(beta.shape)
for i in range(beta.shape[0]):
    for j in range(beta.shape[1]):
        beta_u=np.copy(beta)
        beta_u[i,j]=beta_u[i,j]+eps_beta
        beta_d=np.copy(beta)
        beta_d[i,j]=beta_d[i,j]-eps_beta
        beta_grad_test[i,j]=(cost_fn(alpha,beta_u,X_in,Y,special.expit,softmax_fn)
            -cost_fn(alpha,beta_d,X_in,Y,special.expit,softmax_fn))/(2*eps_beta)
print beta_grad_test
print cost_grad_beta0.sum(0)
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
#################################################################################### 
####################################################################################
#################################################################################### 
####################################################################################
####################################################################################  
####################################################################################  
####################################################################################
####################################################################################  
####################################################################################
####################################################################################  
####################################################################################
####################################################################################  
#load data
pwd_temp=%pwd
dir1='/home/sgolbeck/workspace/PythonExercises/NeuralNets'
if pwd_temp!=dir1:
    os.chdir(dir1)
dir1=dir1+'/data' 
import scipy.io as sio
dat=sio.loadmat(dir1+'/ex3data1.mat')
X_in=np.array(dat['X'])
y_in=np.array(dat['y'])
#shuffle the rows of X and Y in unison
rng_state = np.random.get_state()
np.random.shuffle(X_in)
np.random.set_state(rng_state)
np.random.shuffle(y_in)
#create a matrix with 0-1 class labels
K=y_in.max()
Y=np.zeros((y_in.shape[0],K))
for i in range(y_in.shape[0]):
    Y[i,y_in[i]-1]=1
p=X_in.shape[1]

M_hidden=20   #not including bias
#generate random values for the model parameters
#dim (p+1,M-1), (#features including input bias,#hidden layers not including bias)
alpha=np.random.normal(size=(p+1)*M_hidden).reshape(p+1,M_hidden)
#dim (M,K), (#hidden layers including bias hidden layer,#classes)
beta=np.random.normal(size=(M_hidden+1)*K).reshape(M_hidden+1,K)


#gradient descent
epochs=200
steps_epoch=X_in.shape[0]/50
eps_alpha=0.01
eps_beta=0.01
params=NN_fit_grad_descent(epochs,steps_epoch,eps_alpha,eps_beta,alpha,beta,X_in,Y,special.expit,softmax_fn,grad_sigmoid,grad_softmax)


#convert class matrix to array of class labels (starting at 1) for use in confusion matrix
y_in=Y.argmax(1)+1
#dimension of data
#number of rows (observations)
n=X_in.shape[0]   
#number of observations per each step through an epoch
n_obs=n/steps_epoch
#number of input features 
p=X_in.shape[1]
#add bias vector to inputs
X=np.column_stack((np.ones(n),np.copy(X_in)))
#number of rows in the dependent variable
n_Y=Y.shape[0]
#number of classes
K=Y.shape[1]
#initialize iterator
epoch_iter=0
while(epoch_iter<epochs):
    for n_step in range(steps_epoch):
        #observations used to update alpha, beta
        obs_index=range(n_step*n_obs,(n_step+1)*n_obs)
        #hidden layer outputs
        Z=special.expit(np.dot(X[obs_index,:],alpha))
        #add bias vector to hidden layer 
        Z=np.column_stack((np.ones(n_obs),np.copy(Z)))
        #number of hidden layers (including bias)
        M=Z.shape[1]

        #linear combination of hidden layer outputs
        T=np.dot(Z,beta)
        #outputs
        g=softmax_fn(T)

        #compute 
        temp1=(Y[obs_index,:]/g)
        temp2=grad_softmax(T)
        #Y*(1/g)*dg/dT
        C=np.array([[temp1[:,k]*temp2[:,k,j] for k in range(K)] for j in range(K)]).transpose()

        #sum over output classes
        D1=C.sum(axis=1)
        #sum[Y*(1/g)*dg/dT]*Z
        #dim (n,M,K): (obs index,hidden layer index including bias,class index)
        cost_grad_beta=-np.array([[Z[:,j] for j in range(M)]*D1[:,k] for k in range(K)]).transpose()

        #sum[sum[Y*(1/g)*dg/dT]*beta]
        D3=np.array([[D1[i,:]*beta[j,:] for j in range(1,M)] for i in range(n_obs)]).sum(axis=2)
        #sum[sum[Y*(1/g)*dg/dT]*beta]*dZ/d(X*alpha)
        grad_act=grad_sigmoid(np.dot(X[obs_index,:],alpha))
        D4=D3*grad_act
        #dim (n,p+1,M)
        cost_grad_alpha=-np.array([[D4[:,j]*X[obs_index,k] for k in range(p+1)] for j in range(M-1)]).transpose()

        #update beta
        beta=beta-eps_beta*cost_grad_beta.sum(0)
        #update alpha
        alpha=alpha-eps_alpha*cost_grad_alpha.sum(0)
    #iterate over the remaining observations if the whole data set has not been pass through
    if(np.max(obs_index)<n-1):
        #observations used to update alpha, beta
        obs_index=range(np.max(obs_index),n)
        #hidden layer outputs
        Z=special.expit(np.dot(X[obs_index,:],alpha))
        #add bias vector to hidden layer 
        Z=np.column_stack((np.ones(n_obs),np.copy(Z)))
        #number of hidden layers (including bias)
        M=Z.shape[1]

        #linear combination of hidden layer outputs
        T=np.dot(Z,beta)
        #outputs
        g=softmax_fn(T)

        #compute 
        temp1=(Y[obs_index,:]/g)
        temp2=grad_softmax(T)
        #Y*(1/g)*dg/dT
        C=np.array([[temp1[:,k]*temp2[:,k,j] for k in range(K)] for j in range(K)]).transpose()

        #sum over output classes
        D1=C.sum(axis=1)
        #sum[Y*(1/g)*dg/dT]*Z
        #dim (n,K,M): (obs index,class index,hidden layer index including bias
        cost_grad_beta=-np.array([[Z[:,j] for j in range(M)]*D1[:,k] for k in range(K)]).transpose()

        #sum[sum[Y*(1/g)*dg/dT]*beta]
        D3=np.array([[D1[i,:]*beta[j,:] for j in range(1,M)] for i in range(n_obs)]).sum(axis=2)
        #sum[sum[Y*(1/g)*dg/dT]*beta]*dZ/d(X*alpha)
        grad_act=grad_sigmoid(np.dot(X[obs_index,:],alpha))
        D4=D3*grad_act
        #dim (n,p+1,M)
        cost_grad_alpha=-np.array([[D4[:,j]*X[obs_index,k] for k in range(p+1)] for j in range(M-1)]).transpose()

        #update beta
        beta=beta-eps_beta*cost_grad_beta.sum(0)
        #update alpha
        alpha=alpha-eps_alpha*cost_grad_alpha.sum(0)

    #update epoch iterator
    epoch_iter+=1
    #predict classes and compute error rate
    Z=special.expit(np.dot(X,alpha))
    Z=np.column_stack((np.ones(n),np.copy(Z)))
    T=np.dot(Z,beta)
    #convert to class number
    y_pred=softmax_fn(T).argmax(1)+1
    CF=confusion_matrix_multi(y_pred,y_in,K)
    error_rate=CF.diagonal().sum(0)/n
    print error_rate