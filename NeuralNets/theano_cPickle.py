import cPickle
import numpy
import theano
path='test.zip'
numdims=2
numclasses=3

W = theano.shared(value=numpy.random.normal(size=(numdims,numclasses)).astype(dtype = theano.config.floatX), name='W')
b = theano.shared(value=numpy.ones((numclasses,), dtype = theano.config.floatX), name='b')
n_dims=theano.shared(numpy.array([W.get_value(borrow=True).shape,b.get_value(borrow=True).shape]),name='n_dims')

save_file = open(path, 'wb')  # this will overwrite current contents
cPickle.dump(W.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
cPickle.dump(b.get_value(borrow=True), save_file, -1)  # .. and it triggers much more efficient storage than numpy's default
cPickle.dump(n_dims.get_value(borrow=True),save_file,-1)
save_file.close()


####################################################################################
####################################################################################
####################################################################################
import cPickle
import numpy
import theano
path='test.zip'
numdims=2
numclasses=3
save_file = open(path)
n_dims = theano.shared(value=[], name='n_dims')
n_dims.set_value(cPickle.load(save_file),borrow=True)
for i in range(len(n_dims.get_value(borrow=True))):
	W = theano.shared(value=numpy.zeros((numdims,numclasses)).astype(dtype = theano.config.floatX), name='W')
	b = theano.shared(value=numpy.zeros((numclasses,), dtype = theano.config.floatX), name='b')
W.set_value(cPickle.load(save_file), borrow=True)
b.set_value(cPickle.load(save_file), borrow=True)
print W.get_value(borrow=True)
print b.get_value(borrow=True)
save_file.close()
