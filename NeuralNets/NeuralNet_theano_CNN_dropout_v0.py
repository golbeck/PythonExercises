"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time
import datetime

import cPickle

import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


####################################################################################
####################################################################################
####################################################################################
def load_data():
    ####################################################################################
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    dat=np.array(pd.read_csv("train.csv"))

    #dimensions of MNIST
    w=28
    h=28

    X_in=dat[:,1:]/255.0
    n=X_in.shape[0]
    #convert training set to an array of 3D matrices: (1,28,28)
    X_image=np.zeros((n,1,w,h))
    for j in range(w):
        for k in range(h):
            l=j*w+k
            X_image[:,0,j,k]=X_in[:,l]

    frac=0.95
    n_train_set=int(0.95*n)
    train_set=(X_image[range(n_train_set),:],dat[range(n_train_set),0])
    train_set_x, train_set_y = shared_dataset(train_set)

    valid_set=(X_image[range(n_train_set,n),:],dat[range(n_train_set,n),0])
    valid_set_x, valid_set_y = shared_dataset(valid_set)

    #convert test set to array of 3D matrices: (1,28,28)
    test_in=np.array(pd.read_csv("test.csv"))/255.0
    n=test_in.shape[0]
    #convert training set to an array of 2D matrices
    X_image_test=np.zeros((n,1,w,h))
    for j in range(w):
        for k in range(h):
            l=j*w+k
            X_image_test[:,0,j,k]=test_in[:,l]
    test_set_x=theano.shared(np.asarray(X_image_test,dtype=theano.config.floatX),
                                 borrow=True)
    rval = [(train_set_x, train_set_y),(valid_set_x, valid_set_y), test_set_x]
    return rval
####################################################################################
####################################################################################
####################################################################################
def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output
####################################################################################
####################################################################################
####################################################################################
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    ####################################################################################
    ####################################################################################
    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=np.zeros((n_in, n_out),dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros((n_out,),dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

####################################################################################
####################################################################################
####################################################################################
class HiddenLayer(object):
    ####################################################################################
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,W_update=None,b_update=None,
                 activation=T.tanh, p_in):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer

        :type p_in: float
        :param p_in: dropout probability
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),high=np.sqrt(6. / (n_in + n_out)),size=(n_in, n_out)),
                dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        if p_in is not None:
            self.output = _dropout_from_layer(rng, self.output, p_in)
        # parameters of the model
        self.params = [self.W, self.b]

####################################################################################
####################################################################################
####################################################################################
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """
    ####################################################################################
    def __init__(self, rng, input, filter_shape, image_shape, pool_size, p_in):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        :type p_in: float
        :param p_in: dropout probability
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(pool_size))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )


        if p_in[0] is not None:
            conv_out = _dropout_from_layer(rng, conv_out, p_in[0])

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=pool_size,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        if p_in[1] is not None:
            self.output = _dropout_from_layer(rng, self.output, p_in[1])

        # store parameters of this layer
        self.params = [self.W, self.b]


####################################################################################
####################################################################################
####################################################################################
class CNN(object):
    """convolutional neural network """
    ####################################################################################
    def __init__(self, rng, input, n_kerns, filter_shape, pool_size,batch_size,
                    n_in,n_hidden, n_out, activation, p_in):

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, n_kerns[0], 12, 12)
        layer0_input=input
        layer0_input = _dropout_from_layer(rng, layer0_input, p_in[0])

        self.layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, n_in[0], n_in[1]),
            filter_shape=(n_kerns[0], 1, filter_shape[0], filter_shape[1]),
            pool_size=pool_size,
            p_in=[p_in[1],p_in[2]]
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, n_kerns[1], 4, 4)

        #dimensions of image after convolution and downsampling of previous layer
        n_w=(n_in[0]-filter_shape[0]+1)/pool_size[0]
        n_h=(n_in[1]-filter_shape[1]+1)/pool_size[1]

        self.layer1 = LeNetConvPoolLayer(
            rng,
            input=self.layer0.output,
            image_shape=(batch_size, n_kerns[0], n_w, n_h),
            filter_shape=(n_kerns[1], n_kerns[0], filter_shape[0], filter_shape[1]),
            pool_size=pool_size,
            p_in=[p_in[3],p_in[4]]
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, n_kerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        self.layer2_input = self.layer1.output.flatten(2)

        #dimensions of image after convolution and downsampling of previous layer
        n_w=(n_w-filter_shape[0]+1)/pool_size[0]
        n_h=(n_h-filter_shape[1]+1)/pool_size[1]

        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(
            rng,
            input=self.layer2_input,
            n_in=n_kerns[1] * n_w * n_h,
            n_out=n_hidden[0],
            activation=activation,
            p_in=p_in[5]
        )

        # construct a fully-connected sigmoidal layer
        self.layer3 = HiddenLayer(
            rng,
            input=self.layer2.output,
            n_in=n_hidden[0],
            n_out=n_hidden[1],
            activation=activation
            p_in=p_in[6]
        )

        # construct a fully-connected sigmoidal layer
        self.layer4 = HiddenLayer(
            rng,
            input=self.layer3.output,
            n_in=n_hidden[1],
            n_out=n_hidden[2],
            activation=activation,
            p_in=p_in[7]
        )

        # classify the values of the fully-connected sigmoidal layer
        self.logRegressionLayer = LogisticRegression(
            input=self.layer4.output, 
            n_in=n_hidden[2], 
            n_out=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.layer0.W).sum()
            + abs(self.layer1.W).sum()
            + abs(self.layer2.W).sum()
            + abs(self.layer3.W).sum()
            + abs(self.layer4.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.layer0.W ** 2).sum()
            + (self.layer1.W ** 2).sum()
            + (self.layer2.W ** 2).sum()
            + (self.layer3.W ** 2).sum()
            + (self.layer4.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = (self.layer0.params + self.layer1.params 
                        + self.layer2.params + self.layer3.params
                        + self.layer4.params 
                        + self.logRegressionLayer.params)

        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)

        #outputs
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x

        # compute prediction as class whose probability is maximal
        self.y_out = T.argmax(self.p_y_given_x, axis=-1)
        self.loss = lambda y: self.negative_log_likelihood(self.y)

    ####################################################################################
    def mse(self, y):
        # error between output and target
        return T.mean((self.y_out - y) ** 2)

    ####################################################################################
    def predict_y(self):
        return self.y_out

    ####################################################################################
    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    ####################################################################################
    def errors(self, y):
        """Return a float representing the number of errors in the sequence
        over the total number of examples in the sequence ; zero one
        loss over the size of the sequence
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_out.ndim:
            raise TypeError('y should have the same shape as self.y_out',
                ('y', y.type, 'y_out', self.y_out.type))

        return T.mean(T.neq(self.y_out, y))

####################################################################################
####################################################################################
####################################################################################
class TrainCNN(object):
    """
        builds a CNN and trains the network
    """
    ####################################################################################
    def __init__(self, n_kerns=[5,10],filter_shape=[4,4],pool_size=(2,2),n_in=5, rng=None, 
                 n_hidden=np.array([50,50]), n_out=5, learning_rate=0.1, rate_adj=0.40,
                 n_epochs=100, L1_reg=0.00, L2_reg=0.00, learning_rate_decay=0.40,
                 activation='tanh',final_momentum=0.99, initial_momentum=0.5,
                 momentum_epochs=400.0,batch_size=100,p_in=[0.9,0.7]):

        #initialize the inputs (tunable parameters) and activations
        if rng is None:
            self.rng = np.random.RandomState(1)
        else:
            self.rng = rng

        #number of filters for each convolution layer: np array
        self.n_kerns=n_kerns
        #dimension of convolutional filter
        self.filter_shape=filter_shape
        #downsampling dims
        self.pool_size=pool_size
        #np array [width,height] pixel dims
        self.n_in = n_in
        #number of neurons in each hidden layer
        self.n_hidden = n_hidden
        #number of classes
        self.n_out = int(n_out)

        self.learning_rate = float(learning_rate)
        self.rate_adj=float(rate_adj)
        self.learning_rate_decay = float(learning_rate_decay)
        self.n_epochs = int(n_epochs)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activation = activation
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_epochs = int(momentum_epochs)
        self.batch_size=int(batch_size)
        self.p_in=p_in

        #build the network
        self.ready()
    ####################################################################################
    def ready(self):
        #builds the network given the tunable parameters and activation
        # input 
        self.x = T.tensor4('x')
        # target 
        self.y = T.vector(name='y', dtype='int32')

        if self.activation == 'tanh':
            activation = T.tanh
        elif self.activation == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activation == 'relu':
            activation = lambda x: x * (x > 0)
        elif self.activation == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError


        self.CNN = CNN(rng=self.rng, input=self.x, n_kerns=self.n_kerns, filter_shape=self.filter_shape, 
                        pool_size=self.pool_size,batch_size=self.batch_size,n_in=self.n_in,
                        n_hidden=self.n_hidden, n_out=self.n_out,activation=activation,p_in=self.p_in)

    ####################################################################################
    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)

        for param in self.CNN.params:
            param.set_value(i.next().get_value(borrow=True))



    ####################################################################################
    def model_trained(self,path):
        """ load saved parameters from .zip file specified in 'path' and predict model on test set
        """

        #load parameters saved from training
        save_file = open(path)
        weights=cPickle.load(save_file)
        save_file.close()
        #set value of CNN params using trained parameters
        self._set_weights(weights)

        #data used for the predictions
        datasets = load_data()
        test_set_x = datasets[1]

        # compiling a Theano function that predicts the classes for a set of inputs
        predict_model = theano.function(
            inputs=[self.x],
            outputs=self.CNN.predict_y()
        )

        test_predictions=predict_model(test_set_x)
        columns = ['ImageId', 'Label']
        index = range(1,test_predictions.shape[0]+1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)
        df['ImageId']=index
        df['Label']=test_predictions
        df.head(10)
        df.to_csv("test_predictionsTheano_temp.csv",sep=",",index=False)


    ####################################################################################
    def fit(self,path,validation_frequency=10):
        """ Fit model
        validation_frequency : int
            in terms of number of epochs
        """

        datasets = load_data()

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x = datasets[2]
        #number of observations
        n_train = train_set_x.get_value(borrow=True).shape[0]
        n_valid = valid_set_x.get_value(borrow=True).shape[0]

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
        # index to a case
        index = T.lscalar('index')    
        # learning rate (may change)
        l_r = T.scalar('l_r', dtype=theano.config.floatX)
        # momentum
        mom = T.scalar('mom', dtype=theano.config.floatX)  

        #the cost function used for grad descent
        cost = (
            self.CNN.negative_log_likelihood(self.y)
            + self.L1_reg * self.CNN.L1
            + self.L2_reg * self.CNN.L2_sqr
        )

        #given training data, compute the error
        compute_train_error = theano.function(
                inputs=[index],
                outputs=self.CNN.errors(self.y),
                givens={
                    self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                    self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
                    }
            )

        #given training data, compute the error
        compute_valid_error = theano.function(
                inputs=[index],
                outputs=self.CNN.errors(self.y),
                givens={
                    self.x: valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                    self.y: valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
                    }
            )

        # compute the gradient of cost with respect to theta = (W, W_in, W_out)
        # gradients on the weights using BPTT
        gparams = []
        for param in self.CNN.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        #apply the updates using the gradients and previous updates
        updates=[]
        for param, gparam in zip(self.CNN.params, gparams):
            #update from last iteration
            weight_update = self.CNN.updates[param]
            #current update with momentum
            upd = mom * weight_update - l_r * gparam
            #update the weight parameters and their grad descent updates
            updates.append((self.CNN.updates[param],upd))
            updates.append((param,param + upd))

        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates`
        train_model = theano.function(
                inputs=[index, l_r, mom],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                    self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
                    }
        )
        # compiling a Theano function that predicts the classes for a set of inputs
        predict_model = theano.function(
            inputs=[index],
            outputs=self.CNN.predict_y(),
            givens={
                self.x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size]
                }
        )

        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        #initial error is set to 100%
        best_valid_loss = 1.0
        #start clock
        start_time = time.clock()
        epoch = 0

        tol=0.005
        improvement_threshold=0.9
        patience=20
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / self.batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / self.batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / self.batch_size
        #train the network over epochs
        done_looping = False

        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for idx in xrange(n_train_batches):
                effective_momentum = self.initial_momentum + (self.final_momentum - self.initial_momentum)*epoch/self.momentum_epochs
                example_cost = train_model(idx, self.learning_rate,effective_momentum)

            if epoch % validation_frequency == 0:
                # compute loss on validation set
                valid_losses = [compute_valid_error(i) for i in xrange(n_valid_batches)]
                this_valid_loss = np.mean(valid_losses)

                # if we got the best validation score until now
                if this_valid_loss < best_valid_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_valid_loss < best_valid_loss *
                        improvement_threshold
                    ):
                        patience += 2*validation_frequency
                        #save parameters if best performance
                        save_file = open(path, 'wb')  # this will overwrite current contents
                        cPickle.dump(self.CNN.params, save_file, -1)  # the -1 is for HIGHEST_PROTOCOL and it triggers much more efficient storage than np's default
                        save_file.close()

                    best_valid_loss = this_valid_loss
                else:
                    #if no improvement seen, adjust learning rate
                    self.learning_rate*=self.rate_adj

                print(
                    'epoch %i, validation error %f, best validation error %f, learning rate %f' %
                    (
                        epoch,
                        this_valid_loss * 100.,
                        best_valid_loss *100,
                        self.learning_rate
                    )
                )

            if epoch>patience:
                done_looping = True
                break

            self.learning_rate *= self.learning_rate_decay


        # compute loss on training set
        train_losses = [compute_train_error(i) for i in xrange(n_train_batches)]
        this_train_loss = np.mean(train_losses)
        print(
                'training error %f' %this_train_loss
            )

        end_time = time.clock()
        print ('The code ran for %.2fm' % ((end_time - start_time) / 60.))


        #load parameters for best validation set performance
        save_file = open(path)
        weights=cPickle.load(save_file)
        save_file.close()
        #set value of CNN params 
        self._set_weights(weights)
        #for test set predictions, update batch size to the full set
        test_predictions=np.array([predict_model(i) for i in xrange(n_test_batches)]).ravel()
        columns = ['ImageId', 'Label']
        index = range(1,test_predictions.shape[0]+1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)
        df['ImageId']=index
        df['Label']=test_predictions
        df.head(10)
        df.to_csv("CNN_predictions_Theano.csv",sep=",",index=False)
####################################################################################
####################################################################################
####################################################################################
def test_CNN():
    """ Test CNN. """
    learning_rate=0.1
    n_kerns=[50,80]
    n_hidden = np.array([1500,1000,300])
    n_in = np.array([28,28])
    n_out = 10
    batch_size=50
    filter_shape=[5,5]
    pool_size=(2,2)
    n_epochs=400
    p_in=0.7*np.ones(1+2+len(n_hidden))
    p_in[0]=0.9

    rng = np.random.RandomState(2479)
    np.random.seed(0)

    model = TrainCNN(n_kerns=n_kerns,filter_shape=filter_shape,pool_size=pool_size,n_in=n_in, rng=rng, 
                 n_hidden=n_hidden, n_out=n_out, learning_rate=learning_rate, rate_adj=0.40,
                 n_epochs=n_epochs, L1_reg=0.00, L2_reg=0.00, learning_rate_decay=0.99,
                 activation='sigmoid',final_momentum=0.99, initial_momentum=0.5,
                 momentum_epochs=100.0,batch_size=batch_size,p_in=p_in)

    path='params.zip'
    model.fit(path=path,validation_frequency=10)


    # model_fit = TrainMLP(n_in=n_in, rng=rng, n_hidden=n_hidden, n_out=n_out,
    #                 L1_reg=0.00, L2_reg=0.00,
    #                 learning_rate=0.1, learning_rate_decay=0.99, rate_adj=0.5,                    
    #                 final_momentum=0.99, initial_momentum=0.5,momentum_epochs=200.0,
    #                 n_epochs=40, activation='sigmoid',batch_size=batch_size)

    # path='params.zip'
    # model_fit.model_trained(path=path)
####################################################################################
####################################################################################
####################################################################################
####################################################################################
if __name__ == "__main__":
    pwd_temp=os.getcwd()
    # dir1='/home/sgolbeck/workspace/Kaggle_MNIST'
    dir1='/home/golbeck/Workspace/Kaggle_MNIST'
    dir1=dir1+'/data' 
    if pwd_temp!=dir1:
        os.chdir(dir1)
    test_CNN()