import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import os
import time
import datetime
import cPickle as pickle


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
    train_set=(dat[:,1:],dat[:,0])
    test_set_x=np.array(pd.read_csv("test.csv"))
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), test_set_x]
    return rval
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
                 activation=T.tanh):
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
        # parameters of the model
        self.params = [self.W, self.b]



####################################################################################
####################################################################################
####################################################################################
class MLP(object):
    """    
        Multilayer Perceptron class
    """
    ####################################################################################
    def __init__(self, rng, input, n_in, n_hidden, n_out, activation=T.tanh):

        self.input = input
        self.activation = activation

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer1 = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden[0],
            activation=activation
        )

        #second layer
        self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer1.output,
            n_in=n_hidden[0],
            n_out=n_hidden[1],
            activation=activation
        )

        # The logistic regression layer gets as input the hidden units
        # of the final hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer2.output,
            n_in=n_hidden[1],
            n_out=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer1.W).sum()
            + abs(self.hiddenLayer2.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer1.W ** 2).sum()
            + (self.hiddenLayer2.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer1.params + self.hiddenLayer2.params + self.logRegressionLayer.params

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
class TrainMLP(object):
    """
        builds a MLP and trains the network
    """
    ####################################################################################
    def __init__(self, n_in=5, rng=0.0, n_hidden=np.array([50,50]), n_out=5, learning_rate=0.01, rate_adj=0.40,
                 n_epochs=100, L1_reg=0.00, L2_reg=0.00, learning_rate_decay=0.40,
                 activation='tanh', output_type='real',
                 final_momentum=0.9, initial_momentum=0.5,
                 momentum_epochs=200.0,batch_size=100):

        #initialize the inputs (tunable parameters) and activations
        self.rng=rng
        self.n_in = int(n_in)
        self.n_hidden = n_hidden
        self.n_out = int(n_out)
        self.learning_rate = float(learning_rate)
        self.rate_adj=float(rate_adj)
        self.learning_rate_decay = float(learning_rate_decay)
        self.n_epochs = int(n_epochs)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activation = activation
        self.output_type = output_type
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_epochs = int(momentum_epochs)
        self.batch_size=int(batch_size)

        #build the network
        self.ready()
    ####################################################################################
    def ready(self):
        #builds the network given the tunable parameters and activation
        # input 
        self.x = T.matrix()
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


        self.MLP = MLP(rng=self.rng,input=self.x, n_in=self.n_in,
                       n_hidden=self.n_hidden, n_out=self.n_out,
                       activation=activation)

    ####################################################################################
    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)

        for param in self.MLP.params:
            param.set_value(i.next())


    ####################################################################################
    def fit(self,validation_frequency=10):
        """ Fit model
        validation_frequency : int
            in terms of number of epochs
        """

        datasets = load_data()

        train_set_x, train_set_y = datasets[0]
        test_set_x = datasets[1]
        #number of observations
        n_train = train_set_x.get_value(borrow=True).shape[0]

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
            self.MLP.negative_log_likelihood(self.y)
            + self.L1_reg * self.MLP.L1
            + self.L2_reg * self.MLP.L2_sqr
        )

        #given training data, compute the error
        compute_train_error = theano.function(
                inputs=[index],
                outputs=self.MLP.errors(self.y),
                givens={
                    self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                    self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
                    }
            )

        # compute the gradient of cost with respect to theta = (W, W_in, W_out)
        # gradients on the weights using BPTT
        gparams = []
        for param in self.MLP.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        #dictionary of updates: one for each param, and their grad descent update
        # updates = {}
        # for param, gparam in zip(self.MLP.params, gparams):
        #     #update from last iteration
        #     weight_update = self.MLP.updates[param]
        #     #current update with momentum
        #     upd = mom * weight_update - l_r * gparam
        #     #update the weight parameters and their grad descent updates
        #     updates[weight_update] = upd
        #     updates[param] = param + upd

        updates=[]
        for param, gparam in zip(self.MLP.params, gparams):
            #update from last iteration
            weight_update = self.MLP.updates[param]
            #current update with momentum
            upd = mom * weight_update - l_r * gparam
            #update the weight parameters and their grad descent updates
            updates.append((self.MLP.updates[param],upd))
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

        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        #initial error is set to infinity
        loss_threshold = np.inf
        #start clock
        start_time = time.clock()
        epoch = 0


        improvement_threshold=0.90
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / self.batch_size

        #train the network over epochs
        while (epoch < self.n_epochs):
            epoch = epoch + 1
            for idx in xrange(n_train_batches):
                effective_momentum = self.initial_momentum + (self.final_momentum - self.initial_momentum)*epoch/self.momentum_epochs
                example_cost = train_model(idx, self.learning_rate,effective_momentum)

            if epoch % validation_frequency == 0:
                # compute loss on training set
                train_losses = [compute_train_error(i) for i in xrange(n_train_batches)]
                this_train_loss = np.mean(train_losses)

                if this_train_loss>loss_threshold:
                    self.learning_rate*=self.rate_adj

                print(
                    'epoch %i, train error %f, learning rate %f' %
                    (
                        epoch,
                        this_train_loss * 100.,
                        self.learning_rate
                    )
                )

                loss_threshold=this_train_loss*improvement_threshold

            self.learning_rate *= self.learning_rate_decay

        end_time = time.clock()
        print ('The code ran for %.2fm' % ((end_time - start_time) / 60.))
####################################################################################
####################################################################################
####################################################################################
####################################################################################
def test_mlp():
    """ Test MLP. """
    n_hidden = np.array([10,10])
    n_in = 28*28
    n_out = 10
    batch_size=100

    rng = np.random.RandomState(1234)
    np.random.seed(0)

    model = TrainMLP(n_in=n_in, rng=rng, n_hidden=n_hidden, n_out=n_out,
                    learning_rate=0.1, learning_rate_decay=0.99,
                    n_epochs=20, activation='tanh',batch_size=batch_size)

    model.fit(validation_frequency=10)

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
    test_mlp()