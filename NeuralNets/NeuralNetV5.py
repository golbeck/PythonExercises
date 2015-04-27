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


def MLP_stoch_grad_mom(epochs,batch_size,mom_rate,eps_alpha,eps_penalty,alpha,M,X_in,Y_in,activation_fn,output_fn,grad_activation_fn,grad_output_fn):
    #multilayer neural network (perceptron) training with mini-batch stochastic grad descent and exponential smoothing (momentum)
    #epochs: maximum number of iterations through the entire data set for gradient descent
    #batch_size: number of observations used to update alpha and beta
    #mom_rate: smoothing rate for parameter updates
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

    #list of gradient-based updates to parameters
    update_param=[]
    #parameters for inputs to all other hidden layer activation functions and the output function (K units)
    for layer in range(n_layers+1):
        update_param.append(np.zeros(alpha[layer].shape))


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
            grad_act=grad_activation_fn(T)
            #add bias vector to hidden layer; dim (batch_size,M[0]+1)
            Z=np.column_stack((np.ones(batch_size),activation_fn(T)))
            #dim (batch_size,M[1],p+1,M[0])
            grad.append(np.array([[[alpha[layer+1][q+1,s]*grad_act[:,q]*X_in[obs_index,r] for s in range(M[layer+1])] 
                for r in range(p+1)] for q in range(M[layer])]).transpose())

            for layer in range(1,n_layers):
                #linear combination of inputs
                T=np.dot(Z,alpha[layer])
                #gradient of activation function with respect to T
                grad_act=grad_activation_fn(T)
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
                Z=np.column_stack((np.ones(batch_size),activation_fn(T)))

            ##############################################################################################
            #output of last hidden layer
            layer=n_layers
            #linear combination of hidden layer outputs
            T=np.dot(Z,alpha[layer])
            #outputs
            g=output_fn(T)

            #compute 
            temp1=(Y_in[obs_index,:]/g)
            grad_output=grad_output_fn(T)
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
                #use momentum smoothing for the updates
                update_param[i][M_range,:]=mom_rate*update_param[i][M_range,:]-eps_alpha*(-grad[i][M_range,:]+eps_penalty*alpha[i][M_range,:])
                #apply smoothed update
                alpha[i][M_range,:]=alpha[i][M_range,:]+update_param[i][M_range,:]
                #no regularization for bias parameters
                update_param[i][0,:]=mom_rate*update_param[i][0,:]-eps_alpha*(-grad[i][0,:])
                #apply smoothed update
                alpha[i][0,:]=alpha[i][0,:]+update_param[i][0,:]

        #update epoch iteration
        epoch_iter+=1
        ##############################################################################################
        ##############################################################################################
        #predict classes and compute accuracy rate
        layer=0
        #add bias vector to hidden layer; dim (batch_size,M[0]+1)
        Z=np.column_stack((np.ones(n),activation_fn(np.dot(X_in,alpha[layer]))))

        for layer in range(1,n_layers):
            #add bias vector to hidden layer 
            Z=np.column_stack((np.ones(n),activation_fn(np.dot(Z,alpha[layer]))))

        #output of last hidden layer
        layer=n_layers
        #convert to class number
        y_pred=output_fn(np.dot(Z,alpha[layer])).argmax(1)+1
        #convert class matrix to array of class labels (starting at 1) for use in confusion matrix
        y_dat=Y_in.argmax(1)+1
        #compute confusion matrix using predicted outputs (y_pred) and actual labels (y_dat)
        CF=confusion_matrix_multi(y_pred,y_dat,K)
        print CF
        accuracy=CF.diagonal().sum(0)/n
        print "accuracy rate %s" %accuracy

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
eps_alpha=0.01
eps_penalty=0.01
mom_rate=0.85
#number of neurons in each hidden layer
M=np.array([10,10])
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

#number of observations used in each gradient update
batch_size=500
#number of complete iterations through training data set
epochs=20
#train network and return parameters
parameters=MLP_stoch_grad_mom(epochs,batch_size,mom_rate,eps_alpha,eps_penalty,alpha,M,X_in,Y_in,special.expit,softmax_fn,grad_sigmoid,grad_softmax)