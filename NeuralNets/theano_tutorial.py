import numpy as np
import theano 
import theano.tensor as T


shared_var = theano.shared(np.array([[1, 2], [3, 4]], dtype=theano.config.floatX))
# The type of the shared variable is deduced from its initialization
print shared_var.type()

# We can set the value of a shared variable using set_value
shared_var.set_value(np.array([[3, 4], [2, 1]], dtype=theano.config.floatX))
# ..and get it using get_value
print shared_var.get_value()

shared_squared = shared_var**2
# The first argument of theano.function (inputs) tells Theano what the arguments to the compiled function should be.
# Note that because shared_var is shared, it already has a value, so it doesn't need to be an input to the function.
# Therefore, Theano implicitly considers shared_var an input to a function using shared_squared and so we don't need
# to include it in the inputs argument of theano.function.
function_1 = theano.function([], shared_squared)
print function_1()

# We can also update the state of a shared var in a function
subtract = T.matrix('subtract')
# updates takes a dict where keys are shared variables and values are the new value the shared variable should take
# Here, updates will set shared_var = shared_var - subtract
function_2 = theano.function([subtract], shared_var, updates={shared_var: shared_var - subtract})
print "shared_var before subtracting [[1, 1], [1, 1]] using function_2:"
print shared_var.get_value()
# Subtract [[1, 1], [1, 1]] from shared_var
function_2(np.array([[1, 1], [1, 1]]))
print "shared_var after calling function_2:"
print shared_var.get_value()
# Note that this also changes the output of function_1, because shared_var is shared!
print "New output of function_1() (shared_var**2):"
print function_1()


shared_var = theano.shared(np.array([[1, 2], [3, 4]], dtype=theano.config.floatX)*0.0)
print shared_var.get_value()



def gradient_updates_momentum(params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum
    
    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
   
    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((param, param - learning_rate*param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update + (1. - momentum)*np.array([[1,1],[1,1]])))

    return updates


n_in=2
n_out=2
W = theano.shared(
            value=np.random.normal(size=n_in*n_out).reshape(n_in, n_out),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
b = theano.shared(
            value=np.random.normal(size=n_in*n_out).reshape(n_in, n_out),
            name='b',
            borrow=True
        )


params = [W, b]

learning_rate=0.1
momentum=0.5
temp=gradient_updates_momentum(params, learning_rate, momentum)