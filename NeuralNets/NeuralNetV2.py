#test output
from scipy import special
X=np.array([[1,2,3],[4,5,6]])
Y=np.array([[1,0,0,0,0],[0,1,0,0,0]])
#dim (p+1,M-1), (#features including input bias,#hidden layers not including bias)
alpha=np.array([[0.1,0.2,0.2],[0.3,0.4,0.5],[0.5,0.6,0.8],[0.7,0.8,0.9]])
#dim (M,K), (#hidden layers including bias hidden layer,#classes)
beta=np.array([[0.1,0.2,0.3,0.4,0.41],[0.5,0.6,0.7,0.8,0.91],[0.9,0.10,0.11,0.12,0.131],[0.9,0.10,0.11,0.12,0.144]])
cost_fn(alpha,beta,X,Y,special.expit,softmax_fn)
cost_grad_beta0= grad_beta_cost(alpha,beta,X,Y,special.expit,softmax_fn,grad_sigmoid,grad_softmax)
cost_grad_alpha0= grad_alpha_cost(alpha,beta,X,Y,special.expit,softmax_fn,grad_sigmoid,grad_softmax)


####################################################################################
####################################################################################
#test alpha gradient
eps_alpha=0.001
alpha_grad_test=np.zeros(alpha.shape)
for i in range(alpha.shape[0]):
    for j in range(alpha.shape[1]):
        alpha_u=np.copy(alpha)
        alpha_u[i,j]=alpha_u[i,j]+eps_alpha
        alpha_d=np.copy(alpha)
        alpha_d[i,j]=alpha_d[i,j]-eps_alpha
        alpha_grad_test[i,j]=(cost_fn(alpha_u,beta,X,Y,special.expit,softmax_fn)
            -cost_fn(alpha_d,beta,X,Y,special.expit,softmax_fn))/(2*eps_alpha)
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
        beta_grad_test[i,j]=(cost_fn(alpha,beta_u,X,Y,special.expit,softmax_fn)
            -cost_fn(alpha,beta_d,X,Y,special.expit,softmax_fn))/(2*eps_beta)
print beta_grad_test
print cost_grad_beta0.sum(0)
####################################################################################
####################################################################################
#dimension of data
#number of rows (observations)
n=X.shape[0]   
#number of input features 
p=X.shape[1]
#add bias vector to inputs
X=np.column_stack((np.ones(n),np.copy(X)))
#number of rows in the dependent variable
n_Y=Y.shape[0]
#number of classes
K=Y.shape[1]
#check if X and Y have the same number of observations
if(n!=n_Y):
    print "number of rows in X and Y are not the same"

#hidden layer outputs
Z=special.expit(np.dot(X,alpha))
#add bias vector to hidden layer 
Z=np.column_stack((np.ones(n),np.copy(Z)))
#number of hidden layers (including bias)
M=Z.shape[1]
#linear combination of hidden layer outputs
T=np.dot(Z,beta)
#outputs
g=softmax_fn(T)



####################################################################################
####################################################################################

#test softmax gradient
eps_T=0.001
softmax_grad_test=np.zeros([T.shape[1],T.shape[1]])
for i in range(T.shape[1]):
    T_u=np.copy(T)
    T_u[0,i]=T_u[0,i]+eps_T
    T_d=np.copy(T)
    T_d[0,i]=T_d[0,i]-eps_T
    for j in range(T.shape[1]):
        softmax_grad_test[j,i]=(softmax_fn(T_u)[0,j]-softmax_fn(T_d)[0,j])/(2*eps_T)
print softmax_grad_test
####################################################################################
####################################################################################

f=-(np.log(g)*Y).sum()
#compute 
temp1=(Y/g)
temp2=grad_softmax(T)
#Y*(1/g)*dg/dT
C=np.array([[temp1[:,k]*temp2[:,k,j] for k in range(K)] for j in range(K)]).transpose()
#test output of C
temp3=Y[0,0]/g[0,0]*temp2[0,0,1]
print temp3, C[0,0,1]
temp3=Y[0,1]/g[0,1]*temp2[0,1,0]
print temp3, C[0,1,0]
#sum over output classes
D1=C.sum(axis=1)
#sum[Y*(1/g)*dg/dT]*Z
#dim (n,K,M): (obs index,class index,hidden layer index including bias
cost_grad_beta=-np.array([[Z[:,j] for j in range(M)]*D1[:,k] for k in range(K)]).transpose()

#alpha gradient
#sum[sum[Y*(1/g)*dg/dT]*beta]
D3=np.array([[D1[i,:]*beta[j,:] for j in range(1,M)] for i in range(n)]).sum(axis=2)
#sum[sum[Y*(1/g)*dg/dT]*beta]*dZ/d(X*alpha)
grad_activation_fn=grad_sigmoid(np.dot(X,alpha))
D4=D3*grad_activation_fn
#dim (n,p+1,M)
cost_grad_alpha=-np.array([[D4[:,j]*X[:,k] for k in range(p+1)] for j in range(M-1)]).transpose()


f=-(np.log(g)*Y).sum()

####################################################################################
####################################################################################
#gradient descent

k_iter=0
abs_diff=1e8
while((abs_diff>tol)&(k_iter<k_max)):
    #hidden layer outputs
    Z=special.expit(np.dot(X,alpha))
    #add bias vector to hidden layer 
    Z=np.column_stack((np.ones(n),np.copy(Z)))
    #linear combination of hidden layer outputs
    T=np.dot(Z,beta)
    #outputs
    g=softmax_fn(T)

    #compute 
    temp1=(Y/g)
    temp2=grad_softmax(T)
    #Y*(1/g)*dg/dT
    C=np.array([[temp1[:,k]*temp2[:,k,j] for k in range(K)] for j in range(K)]).transpose()

    #sum over output classes
    D1=C.sum(axis=1)
    #sum[Y*(1/g)*dg/dT]*Z
    #dim (n,K,M): (obs index,class index,hidden layer index)
    cost_grad_beta=-np.array([[D1[:,k] for k in range(K)]*Z[:,j] for j in range(M)]).transpose()

    #alpha gradient
    #sum[sum[Y*(1/g)*dg/dT]*beta]
    D3=np.array([[D1[i,:]*beta[j,:] for j in range(M)] for i in range(n)]).sum(axis=2)
    #dim (n,p+1,M)
    cost_grad_alpha=np.array([[D3[:,j]*X[:,k] for k in range(p+1)] for j in range(M)]).transpose()

    beta=beta-eps_beta*grad_beta

    alpha=alpha-eps_alpha*grad_alpha

    J_new=cost_fn(alpha,beta,X,Y,special.expit,softmax_fn)
    abs_diff=abs(J-J_new)
    J=J_new
    k_iter+=1
#        print k,J
return theta    


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
def cost_fn(alpha,beta,X,Y,activation_fn,output_fn):
    #negative log likelihood function for classification
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #beta: np.array of weights for hidden layer; dim (M+1,K)
    #X: np.array of inputs; dim (n,p) (each of the n rows is a separate observation, p is the number of features)
    #Y: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function
    
    #dimension of data
    #number of rows (observations)
    n=X.shape[0]   
    #number of input features 
    p=X.shape[1]
    #add bias vector to inputs
    X=np.column_stack((np.ones(n),np.copy(X)))
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
def grad_alpha_cost(alpha,beta,X,Y,activation_fn,output_fn,grad_activation_fn,grad_output_fn):
    #negative log likelihood function for classification
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #beta: np.array of weights for hidden layer; dim (M+1,K)
    #X: np.array of inputs; dim (n,p) (each of the n rows is a separate observation, p is the number of features)
    #Y: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function

    #dimension of data
    #number of rows (observations)
    n=X.shape[0]   
    #number of input features 
    p=X.shape[1]
    #add bias vector to inputs
    X=np.column_stack((np.ones(n),np.copy(X)))
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
    #dim (n,K,M): (obs index,class index,hidden layer index including bias
    cost_grad_beta=-np.array([[Z[:,j] for j in range(M)]*D1[:,k] for k in range(K)]).transpose()

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
def grad_beta_cost(alpha,beta,X,Y,activation_fn,output_fn,grad_activation_fn,grad_output_fn):
    #negative log likelihood function for classification
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #beta: np.array of weights for hidden layer; dim (M+1,K)
    #X: np.array of inputs; dim (n,p) (each of the n rows is a separate observation, p is the number of features)
    #Y: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function

    #dimension of data
    #number of rows (observations)
    n=X.shape[0]   
    #number of input features 
    p=X.shape[1]
    #add bias vector to inputs
    X=np.column_stack((np.ones(n),np.copy(X)))
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
#################################################################################### 
####################################################################################
####################################################################################    
pwd_temp=%pwd
dir1='/home/golbeck/Workspace/PythonExercises/NeuralNets'
if pwd_temp!=dir1:
    os.chdir(dir1)
dir1=dir1+'/data' 
dat=np.loadtxt(dir1+'/ex2data1.txt',unpack=True,delimiter=',',dtype={'names': ('X1', 'X2', 'Y'),'formats': ('f4', 'f4', 'i4')})
n=len(dat)
Y=dat[n-1]
m=len(Y)
X=np.array([dat[i] for i in range(n-1)]).T
#de-mean and standardize data
X=(X-X.mean(0))/X.std(0)
#add in bias term
X=np.column_stack((np.ones(m),np.copy(X)))
    
tol=1e-6
k_max=400
theta=np.random.normal(size=n)
eps=0.1
k_max=400
k=0
bias=1
theta=fit_logistic_class(bias,theta,X,Y,eps,tol,k_max)    
y_out=prob_logistic_pred(theta,X)[:,1]
print confusion_matrix(y_out,Y)

####################################################################################
####################################################################################  
####################################################################################
####################################################################################  
####################################################################################
####################################################################################  
####################################################################################
####################################################################################  


