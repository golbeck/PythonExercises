#            THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""
theano logistic regression. 
code adapted from http://deeplearning.net/tutorial/logreg.html
"""


import numpy
import theano
import theano.tensor as T


class LogisticRegression(object):

    def __init__(self, numdims, numclasses, weightcost):

        self.features = theano.tensor.matrix('features')
        self.labels = theano.tensor.ivector('labels')
        self.numdims = numdims
        self.numclasses = numclasses
        self.W = theano.shared(value=numpy.zeros((numdims,numclasses), dtype = theano.config.floatX), name='W')
        self.b = theano.shared(value=numpy.zeros((numclasses,), dtype = theano.config.floatX), name='b')
        self.p_y_given_x = T.nnet.softmax(T.dot(self.features, self.W)+self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.weightcost = weightcost
        self._negloglik = -T.mean(T.log(self.p_y_given_x)[T.arange(self.labels.shape[0]),self.labels])
        self._cost = self._negloglik + self.weightcost * (self.W**2).sum()
        self._grads = T.grad(self._cost, self.params)

        if self.labels.ndim != self.y_pred.ndim:
            raise TypeError('labels should have the same shape as self.y_pred', ('labels', target.type, 'y_pred', self.y_pred.type))
        if self.labels.dtype.startswith('int'):
            self._zeroone = T.mean(T.neq(self.y_pred, self.labels))
        else:
            raise NotImplementedError()

        self.predict = theano.function([self.features], self.y_pred)
        self.negloglik = theano.function([self.features, self.labels], self._negloglik)
        self.cost = theano.function([self.features, self.labels], self._cost)
        self.zeroone = theano.function([self.features, self.labels], self._zeroone)
        self.grads = theano.function([self.features, self.labels], self._grads)
        def get_cudandarray_value(x):
            if type(x)==theano.sandbox.cuda.CudaNdarray:
                return numpy.array(x.__array__()).flatten()
            else:
                return x.flatten()
        self.grad = lambda x,y: numpy.concatenate([get_cudandarray_value(g) for g in self.grads(x,y)])
        #self.grad = lambda x,y: numpy.concatenate([g.flatten() for g in self.grads(x,y)])

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(numpy.load(filename))



class GraddescentMinibatch(object):

    def __init__(self, model, features, labels, batchsize, learningrate, momentum=0.9, rng=None, verbose=True):
        self.model        = model
        self._cost        = self.model._cost
        self._grads       = self.model._grads
        self.params       = self.model.params
        self.x            = self.model.features
        self.y            = self.model.labels
        self.features     = features
        self.labels       = labels
        self.learningrate = learningrate
        self.verbose      = verbose
        self.batchsize    = batchsize
        self.numbatches   = self.features.get_value().shape[0] / batchsize
        self.momentum     = momentum 
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                            dtype=theano.config.floatX), name='inc_'+p.name)) for p in self.params])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n

        self.set_learningrate(self.learningrate)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.params, self._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad 
            self.updates[_param] = _param + self.incs[_param]

        self._updateincs = theano.function([self.index], self._cost, updates = self.inc_updates,
            givens = {self.x : self.features[self.index*self.batchsize:(self.index+1)*self.batchsize,:],
                      self.y : self.labels[self.index*self.batchsize:(self.index+1)*self.batchsize]})

        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self):
        cost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.numbatches):
        #for batch_index in range(self.numbatches):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            self._trainmodel(0)

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)


class ConjgradMinimize(object):

    def __init__(self, model, inputs, outputs, maxnumlinesearch=100, verbose=True):
        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        self.verbose = verbose
        self.maxnumlinesearch = maxnumlinesearch

    def f(self, params, inputs, outputs):
        paramsold = self.model.get_params()
        self.model.updateparams(params.flatten())
        result = self.model.cost(inputs, outputs)
        self.model.updateparams(paramsold)
        return result

    def g(self, params, inputs, outputs):
        paramsold = self.model.get_params()
        self.model.updateparams(params.flatten())
        result = self.model.grad(inputs, outputs)
        self.model.updateparams(paramsold)
        return result

    def step(self):
        import minimize
        p, g, numlinesearches = minimize.minimize(self.model.get_params(), self.f, self.g, (self.inputs, self.outputs), self.maxnumlinesearch, verbose=self.verbose)
        self.model.updateparams(p)
        if self.verbose:
            print 'used %f line searches, cost: %f' % (numlinesearches, self.model.cost(self.inputs, self.outputs))

