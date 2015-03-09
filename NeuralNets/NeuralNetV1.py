import numpy as np

def grad_cost(theta,X,Y,eps):
    #theta: np.array of parameters
    #X: np.matrix of inputs (each of the m rows is a separate observation)
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
    #only the non-bias features are in the regularization terms in the cost func
    theta_reg=theta
    theta_reg[0]=0.0
    #gradient with respect to theta
    J_grad=np.dot(X.T,g-Y)+eps*theta_reg
    return J_grad
