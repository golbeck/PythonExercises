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
def NN_fit_stoch_grad_descentV0(rng_state,epochs,batch_size,eps_alpha,eps_beta,lr_frac,alpha,beta,X_in,Y_in,activation_fn,output_fn,grad_activation_fn,grad_output_fn):
    #gradient descent for fitting a single hidden layer neural network classifier
    #updates the gradient "n_mini_batch" times for each run through the entire data set

    #rng_state: state of the RNG for reproducibility
    #epochs: maximum number of iterations through the entire data set for gradient descent
    #batch_size: number of observations used to update alpha and beta
    #eps_alpha: alpha gradient multiplier (assumed to be same for all alpha)
    #eps_beta: beta gradient multiplier (assumed to be same for all beta)
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #beta: np.array of weights for hidden layer; dim (M+1,K)
    #lr_frac: for each iteration, decrease learning rate by a factor of lr_frac
    #X_in: np.array of inputs; dim (n,p) (each of the n rows is a separate observation, p is the number of features)
    #Y_in: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function
    #grad_activation_fn: gradient of activation function
    #grad_output_fn: gradient of output function

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
    n_mini_batch=n/batch_size
    #number of input features 
    p=X_in.shape[1]
    #add bias vector to inputs
    X=np.column_stack((np.ones(n),np.copy(X_in)))
    #number of rows in the dependent variable
    n_Y=Y_in.shape[0]
    #number of classes
    K=Y_in.shape[1]
    #check if X and Y have the same number of observations
    if(n!=n_Y):
        print "number of rows in X and Y are not the same"
        return -9999.
    #initialize iterator
    epoch_iter=0
    while(epoch_iter<epochs):

        #randomly permuate the features and outputs using the same shuffle for each epoch
        np.random.shuffle(X)
        np.random.set_state(rng_state)
        np.random.shuffle(Y_in)        
        #update rng state after shuffling in order to apply the same permutation to both X_in and Y
        rng_state = np.random.get_state()

        #update learning rates
        eps_alpha=lr_frac*eps_alpha
        eps_beta=lr_frac*eps_beta

        #iterate through the entire observation set, updating the gradient via mini-batches
        for batch_number in range(n_mini_batch):
            #observations used to update alpha, beta
            obs_index=range(batch_number*batch_size,(batch_number+1)*batch_size)
            #hidden layer outputs
            Z=activation_fn(np.dot(X[obs_index,:],alpha))
            #add bias vector to hidden layer 
            Z=np.column_stack((np.ones(batch_size),np.copy(Z)))
            #number of hidden layers (including bias)
            M=Z.shape[1]

            #linear combination of hidden layer outputs
            T=np.dot(Z,beta)
            #outputs
            g=output_fn(T)

            #compute 
            temp1=(Y_in[obs_index,:]/g)
            temp2=grad_output_fn(T)
            #Y*(1/g)*dg/dT
            C=np.array([[temp1[:,k]*temp2[:,k,j] for k in range(K)] for j in range(K)]).transpose()

            #sum over output classes
            D1=C.sum(axis=1)
            #sum[Y*(1/g)*dg/dT]*Z
            #dim (n,M,K): (obs index,hidden layer index including bias,class index)
            cost_grad_beta=-np.array([[Z[:,j] for j in range(M)]*D1[:,k] for k in range(K)]).transpose()

            #sum[sum[Y*(1/g)*dg/dT]*beta]
            D3=np.array([[D1[i,:]*beta[j,:] for j in range(1,M)] for i in range(batch_size)]).sum(axis=2)
            #sum[sum[Y*(1/g)*dg/dT]*beta]*dZ/d(X*alpha)
            grad_act=grad_activation_fn(np.dot(X[obs_index,:],alpha))
            D4=D3*grad_act
            #dim (n,p+1,M)
            cost_grad_alpha=-np.array([[D4[:,j]*X[obs_index,k] for k in range(p+1)] for j in range(M-1)]).transpose()

            #update beta
            beta=beta-eps_beta*cost_grad_beta.sum(0)
            #update alpha
            alpha=alpha-eps_alpha*cost_grad_alpha.sum(0)

            # #output accuracy rate for mini-batch update
            # #predict classes and compute accuracy rate
            # Z=activation_fn(np.dot(X,alpha))
            # Z=np.column_stack((np.ones(n),np.copy(Z)))
            # T=np.dot(Z,beta)
            # #convert to class number
            # y_pred=output_fn(T).argmax(1)+1
            # #convert class matrix to array of class labels (starting at 1) for use in confusion matrix
            # y_dat=Y_in.argmax(1)+1
            # CF=confusion_matrix_multi(y_pred,y_dat,K)
            # accuracy_rate=CF.diagonal().sum(0)/n
            # print epoch_iter, accuracy_rate

        # iterate over the remaining observations if the whole data set has not been pass through
        # if(np.max(obs_index)<n-1):
        #     #observations used to update alpha, beta
        #     obs_index=range(np.max(obs_index),n)
        #     #hidden layer outputs
        #     Z=activation_fn(np.dot(X[obs_index,:],alpha))
        #     #add bias vector to hidden layer 
        #     Z=np.column_stack((np.ones(batch_size),np.copy(Z)))
        #     #number of hidden layers (including bias)
        #     M=Z.shape[1]

        #     #linear combination of hidden layer outputs
        #     T=np.dot(Z,beta)
        #     #outputs
        #     g=output_fn(T)

        #     #compute 
        #     temp1=(Y_in[obs_index,:]/g)
        #     temp2=grad_output_fn(T)
        #     #Y*(1/g)*dg/dT
        #     C=np.array([[temp1[:,k]*temp2[:,k,j] for k in range(K)] for j in range(K)]).transpose()

        #     #sum over output classes
        #     D1=C.sum(axis=1)
        #     #sum[Y*(1/g)*dg/dT]*Z
        #     #dim (n,M,K): (obs index,hidden layer index including bias,class index)
        #     cost_grad_beta=-np.array([[Z[:,j] for j in range(M)]*D1[:,k] for k in range(K)]).transpose()

        #     #sum[sum[Y*(1/g)*dg/dT]*beta]
        #     D3=np.array([[D1[i,:]*beta[j,:] for j in range(1,M)] for i in range(batch_size)]).sum(axis=2)
        #     #sum[sum[Y*(1/g)*dg/dT]*beta]*dZ/d(X*alpha)
        #     grad_act=grad_activation_fn(np.dot(X[obs_index,:],alpha))
        #     D4=D3*grad_act
        #     #dim (n,p+1,M)
        #     cost_grad_alpha=-np.array([[D4[:,j]*X[obs_index,k] for k in range(p+1)] for j in range(M-1)]).transpose()

        #     #update beta
        #     beta=beta-eps_beta*cost_grad_beta.sum(0)
        #     #update alpha
        #     alpha=alpha-eps_alpha*cost_grad_alpha.sum(0)

        #update epoch iterator
        epoch_iter+=1
        #output accuracy rate after each epoch
        #predict classes and compute accuracy rate
        Z=activation_fn(np.dot(X,alpha))
        Z=np.column_stack((np.ones(n),np.copy(Z)))
        T=np.dot(Z,beta)
        #convert to class number
        y_pred=output_fn(T).argmax(1)+1
        #convert class matrix to array of class labels (starting at 1) for use in confusion matrix
        y_dat=Y_in.argmax(1)+1
        CF=confusion_matrix_multi(y_pred,y_dat,K)
        accuracy=CF.diagonal().sum(0)/n
        print epoch_iter, accuracy
    return [accuracy,alpha,beta]

####################################################################################
####################################################################################
def NN_fit_stoch_grad_descent_mom(rng_state,epochs,batch_size,eps_alpha,eps_beta,mom_rate,alpha,beta,X_in,Y_in,activation_fn,output_fn,grad_activation_fn,grad_output_fn):
    #gradient descent for fitting a single hidden layer neural network classifier
    #updates the gradient "n_mini_batch" times for each run through the entire data set

    #rng_state: state of the RNG for reproducibility
    #epochs: maximum number of iterations through the entire data set for gradient descent
    #batch_size: number of observations used to update alpha and beta
    #eps_alpha: alpha gradient multiplier (assumed to be same for all alpha)
    #eps_beta: beta gradient multiplier (assumed to be same for all beta)
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #beta: np.array of weights for hidden layer; dim (M+1,K)
    #mom_rate: momentum smoother
    #X_in: np.array of inputs; dim (n,p) (each of the n rows is a separate observation, p is the number of features)
    #Y_in: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function
    #grad_activation_fn: gradient of activation function
    #grad_output_fn: gradient of output function

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
    n_mini_batch=n/batch_size
    #number of input features 
    p=X_in.shape[1]
    #add bias vector to inputs
    X=np.column_stack((np.ones(n),np.copy(X_in)))
    #number of rows in the dependent variable
    n_Y=Y_in.shape[0]
    #number of classes
    K=Y_in.shape[1]
    #check if X and Y have the same number of observations
    if(n!=n_Y):
        print "number of rows in X and Y are not the same"
        return -9999.

    #weight changes for momentum updates; initialize to 0
    v_alpha=np.zeros(alpha.shape)
    v_beta=np.zeros(beta.shape)

    #initialize iterator
    epoch_iter=0
    while(epoch_iter<epochs):

        #randomly permuate the features and outputs using the same shuffle for each epoch
        np.random.shuffle(X)
        np.random.set_state(rng_state)
        np.random.shuffle(Y_in)        
        #update rng state after shuffling in order to apply the same permutation to both X_in and Y
        rng_state = np.random.get_state()

        #iterate through the entire observation set, updating the gradient via mini-batches
        for batch_number in range(n_mini_batch):
            #observations used to update alpha, beta
            obs_index=range(batch_number*batch_size,(batch_number+1)*batch_size)

            #input to first hidden layer computed outside of loop
            #hidden layer outputs
            Z=activation_fn(np.dot(X[obs_index,:],alpha[0,:,:]))
            #add bias vector to hidden layer 
            Z=np.column_stack((np.ones(batch_size),np.copy(Z)))
            #compute gradient of activation with respect to input array T=X*alpha
            grad_act=grad_activation_fn(np.dot(X[obs_index,:],alpha[0,:,:]))
            #dimension (batch_size,M[1],M[0]) #M: number of hidden layers not including bias unit 
            C=[[alpha[0,l,j]*grad_act[:,l] for j in range(1,M[1]+1)] for l in range(1,M[0]+1)].transpose()

            #loop over all additional hidden layers except final one
            for layer in range(1,M.shape[0]-1):
                Z=activation_fn(np.dot(Z[obs_index,:],alpha[layer,:,:]))
                #add bias vector to hidden layer 
                Z=np.column_stack((np.ones(batch_size),np.copy(Z)))
                #compute gradient of activation with respect to input array T=X*alpha
                grad_act=grad_activation_fn(np.dot(X[obs_index,:],alpha[layer,:,:]))
                #dimension (batch_size,M[1],M[0]) #M: number of hidden layers not including bias unit 
                C0=[[alpha[layer,l,j]*grad_act[:,l] for j in range(1,M[layer+1]+1)] for l in range(1,M[layer]+1)].transpose()
                C=[[[C0[:,j,l]*C0[:,l,q] for j in range(1,M[layer+1]+1)] for l in range(1,M[layer]+1)] for q in range(1,M[layer-1]+1)].transpose()
                C=C.sum(3)


            Z=activation_fn(np.dot(Z[obs_index,:],alpha[layer+1,:,:]))
            #add bias vector to hidden layer 
            Z=np.column_stack((np.ones(batch_size),np.copy(Z)))    
            #linear combination of hidden layer outputs
            T=np.dot(Z,beta)
            #outputs
            g=output_fn(T)

            #compute 
            temp1=(Y_in[obs_index,:]/g)
            temp2=grad_output_fn(T)
            #Y*(1/g)*dg/dT
            C=np.array([[temp1[:,k]*temp2[:,k,j] for k in range(K)] for j in range(K)]).transpose()

            #sum over output classes
            D1=C.sum(axis=1)
            #sum[Y*(1/g)*dg/dT]*Z
            #dim (n,M,K): (obs index,hidden layer index including bias,class index)
            cost_grad_beta=-np.array([[Z[:,j] for j in range(M)]*D1[:,k] for k in range(K)]).transpose()

            #sum[sum[Y*(1/g)*dg/dT]*beta]
            D3=np.array([[D1[i,:]*beta[j,:] for j in range(1,M)] for i in range(batch_size)]).sum(axis=2)
            #sum[sum[Y*(1/g)*dg/dT]*beta]*dZ/d(X*alpha)
            grad_act=grad_activation_fn(np.dot(X[obs_index,:],alpha))
            D4=D3*grad_act
            #dim (n,p+1,M)
            cost_grad_alpha=-np.array([[D4[:,j]*X[obs_index,k] for k in range(p+1)] for j in range(M-1)]).transpose()

            #momentum updates
            v_beta=mom_rate*v_beta-eps_beta*cost_grad_beta.sum(0)
            v_alpha=mom_rate*v_alpha-eps_alpha*cost_grad_alpha.sum(0)
            #update beta
            beta=beta+v_beta
            #update alpha
            alpha=alpha+v_alpha

            # #output accuracy rate for mini-batch update
            # #predict classes and compute accuracy rate
            # Z=activation_fn(np.dot(X,alpha))
            # Z=np.column_stack((np.ones(n),np.copy(Z)))
            # T=np.dot(Z,beta)
            # #convert to class number
            # y_pred=output_fn(T).argmax(1)+1
            # #convert class matrix to array of class labels (starting at 1) for use in confusion matrix
            # y_dat=Y_in.argmax(1)+1
            # CF=confusion_matrix_multi(y_pred,y_dat,K)
            # accuracy_rate=CF.diagonal().sum(0)/n
            # print epoch_iter, accuracy_rate

        # iterate over the remaining observations if the whole data set has not been pass through
        # if(np.max(obs_index)<n-1):
        #     #observations used to update alpha, beta
        #     obs_index=range(np.max(obs_index),n)
        #     #hidden layer outputs
        #     Z=activation_fn(np.dot(X[obs_index,:],alpha))
        #     #add bias vector to hidden layer 
        #     Z=np.column_stack((np.ones(batch_size),np.copy(Z)))
        #     #number of hidden layers (including bias)
        #     M=Z.shape[1]

        #     #linear combination of hidden layer outputs
        #     T=np.dot(Z,beta)
        #     #outputs
        #     g=output_fn(T)

        #     #compute 
        #     temp1=(Y_in[obs_index,:]/g)
        #     temp2=grad_output_fn(T)
        #     #Y*(1/g)*dg/dT
        #     C=np.array([[temp1[:,k]*temp2[:,k,j] for k in range(K)] for j in range(K)]).transpose()

        #     #sum over output classes
        #     D1=C.sum(axis=1)
        #     #sum[Y*(1/g)*dg/dT]*Z
        #     #dim (n,M,K): (obs index,hidden layer index including bias,class index)
        #     cost_grad_beta=-np.array([[Z[:,j] for j in range(M)]*D1[:,k] for k in range(K)]).transpose()

        #     #sum[sum[Y*(1/g)*dg/dT]*beta]
        #     D3=np.array([[D1[i,:]*beta[j,:] for j in range(1,M)] for i in range(batch_size)]).sum(axis=2)
        #     #sum[sum[Y*(1/g)*dg/dT]*beta]*dZ/d(X*alpha)
        #     grad_act=grad_activation_fn(np.dot(X[obs_index,:],alpha))
        #     D4=D3*grad_act
        #     #dim (n,p+1,M)
        #     cost_grad_alpha=-np.array([[D4[:,j]*X[obs_index,k] for k in range(p+1)] for j in range(M-1)]).transpose()

        #     #update beta
        #     beta=beta-eps_beta*cost_grad_beta.sum(0)
        #     #update alpha
        #     alpha=alpha-eps_alpha*cost_grad_alpha.sum(0)

        #update epoch iterator
        epoch_iter+=1
        #output accuracy rate after each epoch
        #predict classes and compute accuracy rate
        Z=activation_fn(np.dot(X,alpha))
        Z=np.column_stack((np.ones(n),np.copy(Z)))
        T=np.dot(Z,beta)
        #convert to class number
        y_pred=output_fn(T).argmax(1)+1
        #convert class matrix to array of class labels (starting at 1) for use in confusion matrix
        y_dat=Y_in.argmax(1)+1
        CF=confusion_matrix_multi(y_pred,y_dat,K)
        accuracy=CF.diagonal().sum(0)/n
        print epoch_iter, accuracy
    return [accuracy,alpha,beta]
####################################################################################
####################################################################################
def NN_CV(k_folds,rng_state,epochs,batch_size,eps_alpha,eps_beta,lr_frac,alpha,beta,X_in,Y_in,activation_fn,output_fn,grad_activation_fn,grad_output_fn):
    #k-folds cross-validation on a neural network

    #k_folds: number of partitions of data for use in cross-validation
    #rng_state: state of the RNG for reproducibility
    #epochs: maximum number of iterations through the entire data set for gradient descent
    #batch_size: number of observations used to update alpha and beta
    #eps_alpha: alpha gradient multiplier (assumed to be same for all alpha)
    #eps_beta: beta gradient multiplier (assumed to be same for all beta)
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #beta: np.array of weights for hidden layer; dim (M+1,K)
    #lr_frac: for each iteration, decrease learning rate by a factor of lr_frac
    #X_in: np.array of inputs; dim (n,p) (each of the n rows is a separate observation, p is the number of features)
    #Y_in: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function
    #grad_activation_fn: gradient of activation function
    #grad_output_fn: gradient of output function

    #number of classes
    K=Y_in.shape[1]
    #fraction of data used in training
    train_frac=1.0-1.0/np.float(k_folds)
    #training set size
    n_train=np.int(n*train_frac)
    #test set size
    n_test=n-n_train
    #accuracy rate array 
    accuracy=np.zeros(k_folds)
    #cross-validation loop
    for i_fold in range(k_folds):
        #test set indices
        test_indices=range(i_fold*n_test,(i_fold+1)*n_test)
        #training set indices
        train_indices=range(0,i_fold*n_test)+range((i_fold+1)*n_test,n)

        #train model
        params=NN_fit_stoch_grad_descentV0(rng_state,epochs,batch_size,eps_alpha,eps_beta,lr_frac,alpha,beta,X_in[train_indices,:],Y_in[train_indices,:],activation_fn,output_fn,grad_activation_fn,grad_output_fn)
        #used model on test set
        Y_pred=NN_classifier(params[1],params[2],X_in[test_indices,:],activation_fn,output_fn)
        y_pred=Y_pred.argmax(1)+1
        #convert class matrix to array of class labels (starting at 1) for use in confusion matrix
        y_dat=Y_in[test_indices,:].argmax(1)+1
        CF=confusion_matrix_multi(y_pred,y_dat,K)
        accuracy[i_fold]=CF.diagonal().sum(0)/n_test
    return accuracy.mean()
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

#load data
pwd_temp=os.getcwd()
dir1='/home/sgolbeck/workspace/PythonExercises/NeuralNets'
if pwd_temp!=dir1:
    os.chdir(dir1)
dir1=dir1+'/data' 
import scipy.io as sio
dat=sio.loadmat(dir1+'/ex3data1.mat')
X=np.array(dat['X'])
y_mat=np.array(dat['y'])
#create a matrix with 0-1 class labels
K=y_mat.max()
Y=np.zeros((y_mat.shape[0],K))
for i in range(y_mat.shape[0]):
    Y[i,y_mat[i]-1]=1
p=X.shape[1]


#randomly permuate the features and outputs using the same shuffle for each epoch
rng_state = np.random.get_state()  
np.random.shuffle(X)
np.random.set_state(rng_state)
np.random.shuffle(Y)      
rng_state = np.random.get_state()  

M_hidden=20   #not including bias
#generate random values for the model parameters
#dim (p+1,M-1), (#features including input bias,#hidden layers not including bias)
alpha=np.random.normal(size=(p+1)*M_hidden).reshape(p+1,M_hidden)
#dim (M,K), (#hidden layers including bias hidden layer,#classes)
beta=np.random.normal(size=(M_hidden+1)*K).reshape(M_hidden+1,K)

#stochastic gradient descent
eps_alpha=0.02
eps_beta=0.02
#number of observations used in each gradient update
batch_size=50
#number of complete iterations through training data set
epochs=5

#total number of obs in data set
n=X.shape[0]

####################################################################################
####################################################################################
#k-fold cross validation
k_folds=10

# #grid search to determine good values of hyper parameters
# eps_alpha_array=np.array([0.001,0.01,0.1,0.2,0.5])
# eps_beta_array=np.array([0.001,0.01,0.1,0.2,0.5])
# n_alpha=eps_alpha_array.shape[0]
# n_beta=eps_beta_array.shape[0]
# accuracy_mat=np.zeros([n_alpha,n_beta])
# for i in range(n_alpha):
#     for j in range(n_beta):
#         accuracy_mat[i,j]=NN_CV(k_folds,rng_state,epochs,batch_size,eps_alpha_array[i],eps_beta_array[j],lr_frac,alpha,beta,X,Y,special.expit,softmax_fn,grad_sigmoid,grad_softmax)
#         print accuracy_mat[i,j]
# #returns indices of highest accuracy rate
# ind_best=np.unravel_index(accuracy_mat.argmax(), accuracy_mat.shape)
# eps_alpha=eps_alpha_array[ind_best[0]]
# eps_beta=eps_beta_array[ind_best[1]]

# #grid search to determine number of hidden layers
# M_array=np.array([10,25,50,75,100])
# n_M=M_array.shape[0]
# accuracy_array=np.zeros(n_M)
# for i in range(n_M):
#     M_hidden=M_array[i]  #not including bias
#     #generate random values for the model parameters
#     #dim (p+1,M-1), (#features including input bias,#hidden layers not including bias)
#     alpha=np.random.normal(size=(p+1)*M_hidden).reshape(p+1,M_hidden)
#     #dim (M,K), (#hidden layers including bias hidden layer,#classes)
#     beta=np.random.normal(size=(M_hidden+1)*K).reshape(M_hidden+1,K)
#     accuracy_array[i]=NN_CV(k_folds,rng_state,epochs,batch_size,eps_alpha,eps_beta,lr_frac,alpha,beta,X,Y,special.expit,softmax_fn,grad_sigmoid,grad_softmax)
#     print accuracy_array[i]
# #returns index of highest accuracy rate
# ind_best=accuracy_array.argmax()
# M_hidden=M_array[ind_best]

#####################################################################################
#####################################################################################
#parameters selected from CV
eps_alpha=0.01
eps_beta=0.01
M_hidden=50
#learning rate fraction
lr_frac=0.9
#number of observations used in each gradient update
batch_size=50
#number of complete iterations through training data set
epochs=10
#generate random values for the model parameters
#dim (p+1,M-1), (#features including input bias,#hidden layers not including bias)
alpha=np.random.normal(size=(p+1)*M_hidden).reshape(p+1,M_hidden)
#dim (M,K), (#hidden layers including bias hidden layer,#classes)
beta=np.random.normal(size=(M_hidden+1)*K).reshape(M_hidden+1,K)
# accuracy_rate=NN_CV(k_folds,rng_state,epochs,batch_size,eps_alpha,eps_beta,lr_frac,alpha,beta,X,Y,special.expit,softmax_fn,grad_sigmoid,grad_softmax)
# print accuracy_rate
mom_rate=0.85
params=NN_fit_stoch_grad_descent_mom(rng_state,epochs,batch_size,eps_alpha,eps_beta,mom_rate,alpha,beta,X,Y,special.expit,softmax_fn,grad_sigmoid,grad_softmax)
print params[0]