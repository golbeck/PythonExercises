import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd

######################################################################
#using numpy and Theano types
######################################################################
#matrix definition, dimensions and elements
numpy.asarray([[1., 2], [3, 4], [5, 6]])
numpy.asarray([[1., 2], [3, 4], [5, 6]]).shape
numpy.asarray([[1., 2], [3, 4], [5, 6]])[2, 0]

#broadcasting
a = numpy.asarray([1.0, 2.0, 3.0])
b = 2.0
a * b

######################################################################
#algebra
import theano.tensor as T
from theano import function
#define two symbols (Variables) representing the quantities that you want to add.
#In Theano, all symbols must be typed. In particular, T.dscalar is the type we assign to “0-dimensional arrays (scalar) of doubles (d)”. It is a Theano Type.
#By calling T.dscalar with a string argument, you create a Variable representing a floating-point scalar quantity with the given name
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
#The output of the function f is a numpy.ndarray with zero dimensions.
#The first argument to function is a list of Variables that will be provided as inputs to the function. The second argument is a single Variable or a list of Variables. For either case, the second argument is what we want to see as output when we apply the function
f = function([x, y], z)
f(16.3, 12.1)


######################################################################
#Adding two Matrices
#dmatrix is the Type for matrices of doubles
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
f([[1, 2], [3, 4]], [[10, 20], [30, 40]])
#can use numpy arrays as arguments
f(numpy.array([[1, 2], [3, 4]]), numpy.array([[10, 20], [30, 40]]))

######################################################################
#compute the function elementwise on matrices of doubles
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = function([x], s)
logistic([[0, 1], [-1, -2]])

#multiple outputs
#dmatrices produces as many outputs as names that you provide
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
#3 different outputs
f = function([a, b], [diff, abs_diff, diff_squared])
f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

#Setting a Default Value for an Argument
#define a function that adds two numbers, except that if you only provide one number, the other input is assumed to be one
from theano import Param
x, y = T.dscalars('x', 'y')
z = x + y
f = function([x, Param(y, default=1)], z)
f(33)
f(33, 2)
#inputs with default values must follow inputs without default values (like Python’s functions). There can be multiple inputs with default values. These parameters can be set positionally or by name, as in standard Python:
x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
f = function([x, Param(y, default=1), Param(w, default=2, name='w_by_name')], z)
f(33)
f(33, 2)
f(33, 0, 1)
f(33, w_by_name=1)
#this will be the same as f(33,0,1) or f(33,y=0,w_by_name=1):
f(33, w_by_name=1, y=0)

#accumulator function. It adds its argument to the internal state, and returns the old state value.
from theano import shared
state = shared(0)
inc = T.iscalar('inc')
#updates must be supplied with a list of pairs of the form (shared-variable, new expression)
accumulator = function([inc], state, updates=[(state, state+inc)])
#example
state.get_value()
accumulator(1)
state.get_value()
accumulator(300)
state.get_value()
#reset state
state.set_value(0)


######################################################################
######################################################################
#random numbers
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
srng = RandomStreams(seed=234)
#2x2 matrix of uniform and normal rv's
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
#generate instance of uniform rv's
f_val0 = f()
#different numbers from f_val0
f_val1 = f()
g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
#a random variable is drawn at most once during any single function execution. So the nearly_zeros function is guaranteed to return approximately 0
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

#can seed just one random variable by seeding or assigning to the .rng attribute, using .rng.set_value()
rng_val = rv_u.rng.get_value(borrow=True)   # Get the rng for rv_u
rng_val.seed(89234)                         # seeds the generator
rv_u.rng.set_value(rng_val, borrow=True)    # Assign back seeded rng

#seed all of the random variables allocated by a RandomStreams object by that object’s seed method
srng.seed(902340)  # seeds rv_u and rv_n with different seeds each



######################################################################
######################################################################
#logistic regression
######################################################################
######################################################################
import numpy
import theano
import theano.tensor as T
rng = numpy.random

N = 400
feats = 784
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")
print "Initial model:"
print w.get_value(), b.get_value()

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(inputs=[x,y],outputs=[prediction, xent],updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print "Final model:"
print w.get_value(), b.get_value()
print "target values for D:", D[1]
print "prediction on D:", predict(D[0])


