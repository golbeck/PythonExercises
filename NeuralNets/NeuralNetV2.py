import numpy as np
import os

def x(a,b):
    print "param 1 %s param 2 %s"%(a,b)

def y(z,t):
    print z(*t)

def z(y1,t1,y2,t2):
    print y1(*t1)
    print y2(*t2)

def z1(a,b,y1,y2):
    print y1(a,b)
    print y2(a,b)

#test output
X=np.array([[1,2,3],[4,5,6]])
Y=np.array([[1,0,0,0],[0,1,0,0]])
alpha=np.array([[1,1],[1,1],[1,1],[1,1]])
beta=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
cost_fn(alpha,beta,X,Y,special.expit,special.expit)



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
    #check if X and Y have the same number of observations
    if(n!=n_Y):
        print "number of rows in X and Y are not the same"
        return -9999.
    
    #hidden layer outputs
    Z=activation_fn(np.dot(X,alpha))
    #add bias vector to hidden layer 
    Z=np.column_stack((np.ones(n),np.copy(Z)))
    
    
    
    temp0=-np.log(g)*Y-np.log(1-g)*(1-Y)
    temp1=theta_reg**2
    J=(temp0.sum(axis=0)+0.5*eps*temp1.sum(axis=0))/n
    return J
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
def cost_fn(bias,theta,X,Y,eps):
    #cost function for logistic regression
    #bias: 0: no bias; 1: bias term
    #theta: np.array of parameters
    #X: np.array of inputs (each of the m rows is a separate observation)
    #Y: np.array of outpus
    #eps: regularization constant (set to zero if unregularized)
    #dimension of data
    #number of rows (observations)
    n=X.shape[0]   
    #number of columns (features + bias) 
    m=X.shape[1]
    #number of rows in the dependent variable
    n_Y=Y.shape[0]
    #check if X and Y have the same number of observations
    if(n!=n_Y):
        print "number of rows in X and Y are not the same"
        return -9999.
    
    #only the non-bias features are in the regularization terms in the cost func
    theta_reg=np.copy(theta)
    if(bias==1):
        theta_reg[0]=0.0
        
    #compute logistic function
    g=1/(1+np.exp(-np.dot(X,theta)))
    #log likelihood func
    temp0=-np.log(g)*Y-np.log(1-g)*(1-Y)
    temp1=theta_reg**2
    J=(temp0.sum(axis=0)+0.5*eps*temp1.sum(axis=0))/n
    return J
    
####################################################################################
#################################################################################### 
def prob_logistic_pred(theta,X):
    #theta: np.array of parameters
    #X: np.array of inputs (each of the m rows is a separate observation)
    #compute logistic function
    g=1/(1+np.exp(-np.dot(X,theta)))
    y_out=g>0.5
    y=np.column_stack((g,y_out))
    return y

####################################################################################
#################################################################################### 
def fit_logistic_class(bias,theta,X,y,eps,tol,k_max):
    #theta: np.array of parameters
    #bias: 0: no bias; 1: bias term
    #X: np.array of inputs (each of the m rows is a separate observation)
    #y: np.array of outputs
    #eps: regularization constant (set to zero if unregularized)
    #tol: stopping tolerance for change in cost function
    #k_max: maximum number of iterations for gradient descent
    #compute logistic function
    J=cost_fn(bias,theta,X,Y,eps)
    k=0
    abs_diff=1e8
    while((abs_diff>tol)&(k<k_max)):
        theta-=grad_cost(bias,theta,X,Y,eps)
        J_new=cost_fn(bias,theta,X,Y,eps)
        abs_diff=abs(J-J_new)
        J=J_new
        k+=1
#        print k,J
    return theta
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
#compare to statsmodels logistic regression method
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
 
# fit the model
logit=sm.Logit(Y,X)
result=logit.fit()
#classify if prediction > 0.5
Y_out=result.predict(X)
Y_out_ind=Y_out>0.5

#confusion matrix
cm = confusion_matrix(Y,Y_out_ind)
print(cm)

####################################################################################
####################################################################################  
####################################################################################
####################################################################################  
####################################################################################
####################################################################################  
####################################################################################
####################################################################################  
import scipy
dat=scipy.io.loadmat(dir1+'/ex3data1.mat')
Y_all=np.array(dat['y'])
#reshape to 1d np.array
Y_all=Y_all.ravel()
X=np.array(dat['X'])
m=X.shape[0]
n=X.shape[1]
#add in bias term
bias=1
if(bias==1):
    n=X.shape[1]+1
    X=np.column_stack((np.ones(m),np.copy(X)))

tol=1e-6
eps=0.1
k_max=10000
k=0
Y_class=np.arange(1,11)
prob_out=np.zeros((m,10))
for i in Y_class:
    Y=(Y_all==i).astype(int)
    theta=np.random.normal(size=n)
    theta=fit_logistic_class(bias,theta,X,Y,eps,tol,k_max)
    y=prob_logistic_pred(theta,X)
    prob_out[:,i-1]=y[:,0]
    
Y_out=prob_out.argmax(axis=1)+1
CM=confusion_matrix_multi(Y_out,Y_all,10)
print CM
error_rate=CM.diagonal().sum(0)/m
print error_rate
####################################################################################
####################################################################################
####################################################################################  
####################################################################################
####################################################################################  
####################################################################################
####################################################################################  

Y_class=np.arange(1,11)
prob_out_skl=np.zeros((m,10))
for i in Y_class:
    Y=(Y_all==i).astype(int)
    logit=sm.Logit(Y,X)
    result=logit.fit()
    #classify if prediction > 0.5
    Y_out=result.predict(X)
    prob_out_skl[:,i-1]=Y_out_ind
    
Y_out_skl=prob_out_skl.argmax(axis=1)+1
CM=confusion_matrix_multi(Y_out_skl,Y_all,10)
####################################################################################  
####################################################################################
####################################################################################  
####################################################################################
####################################################################################  

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
Y_out_SVC=OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, Y_all).predict(X)

cm = confusion_matrix(Y_all,Y_out_SVC)
print(cm)






















