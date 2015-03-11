import numpy as np
####################################################################################
####################################################################################
def grad_cost(theta,X,Y,eps):
    #computes the gradient with respect to the parameters (theta) of the logistic regression model
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
        
    #compute logistic function
    g=1/(1+np.exp(-np.dot(X,theta)))
    #only the non-bias features are present in regularization terms in the cost func
    theta_reg=np.copy(theta)
    theta_reg[0]=0.0
    #gradient with respect to theta
    J_grad=(np.dot(X.T,g-Y)+eps*theta_reg)/n
    return J_grad
    
####################################################################################
#################################################################################### 
def cost_fn(theta,X,Y,eps):
    #cost function for logistic regression
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
    y_out=g>=0.5
    y=np.column_stack((g,y_out))
    return y


####################################################################################
#################################################################################### 
def confusion_matrix(theta,X,y):
    #theta: np.array of parameters
    #X: np.array of inputs (each of the m rows is a separate observation)
    #compute logistic function
    g=1/(1+np.exp(-np.dot(X,theta)))
    y_out=g>0.5
    m=y.shape[0]
    tempTP=0
    tempTN=0
    tempFP=0
    tempFN=0
    for i in range(m):
        if(y_out[i]==y[i]):
            if(y[i]==1.):
                tempTP+=1
            else:
                tempTN+=1
        if(y_out[i]!=y[i]):
            if(y_out[i]==1.):
                tempFP+=1
            else:
                tempFN+=1
    CF=np.array([[tempTP,tempFN],[tempFP,tempTN]])
    return CF

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
X=(X-X.mean(0))/X.std(0)
X=np.column_stack((np.ones(m),np.copy(X)))
    
N=100
tol=1e-6
abs_diff=1e8
theta=np.random.normal(size=n)
eps=0.1
J=cost_fn(theta,X,Y,eps)
k_max=400
k=0
while((abs_diff>tol)&(k<k_max)):
    theta-=grad_cost(theta,X,Y,eps)
    J_new=cost_fn(theta,X,Y,eps)
    abs_diff=abs(J-J_new)
    J=J_new
    k+=1
    print k,J
    
print theta    
print confusion_matrix(theta,X,Y)



####################################################################################
####################################################################################  
#compare to statsmodels logistic regression method
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
 
# fit the model
logit=sm.Logit(Y,X)
result=logit.fit()
print result.summary()
#odds ratio components
print np.exp(result.params)

#classify if prediction > 0.5
Y_out=result.predict(X)
Y_out_ind=Y_out>0.5

#confusion matrix
cm = confusion_matrix(Y,Y_out_ind)
print(cm)
