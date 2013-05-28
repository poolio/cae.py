import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict
from utils import vector_from_args, args_from_vector
from itertools import izip


class ModelGradient:
    def __init__(self, model, cost=None, nvis=784):
        self.model = model
        self.nvis = nvis
        self.cost = cost
        self.setup()

    def setup(self):
        self.X = T.matrix('X')
        if self.cost:
            init_grads, updates = self.cost.get_gradients(self.model, self.X, Y=None)
        else:
            init_grads, updates = self._get_gradients(self.X)
        params = self.model.get_params()
        # We need to replace parameters with purely symbolic variables in case some are shared
        symbolic_params = [self._convert_variable(param) for param in params]
        givens = dict(zip(params, symbolic_params))
        # Create gradient and cost functions
        self.params = params
        self.symbolic_params = symbolic_params
        self.grads = theano.function(symbolic_params + [self.X], [init_grads[param] for param in params], givens=givens)
        self._loss = theano.function(symbolic_params + [self.X], self.model.cost(self.X), givens=givens)
        # Maps params -> their derivative

    def _convert_variable(self, x):
        if x.ndim == 1:
            return T.vector(x.name, dtype=x.dtype)
        else:
            return T.matrix(x.name, dtype=x.dtype)

    def batch_loss(self, theta, X, batch_size = 100, avg=True):
        nbatches = int(np.ceil(X.shape[0] / batch_size))
        loss = 0.0
        for i in xrange(nbatches):
            loss += self.loss(theta, X[i*batch_size : (i+1) * batch_size])
        if avg:
            loss /= batch_size
        return loss

    def loss(self, theta, X):
        # Reshape theta -> args
        args = args_from_vector(theta, self.params)
        args += [X]
        # Compute gradient from args
        loss = self._loss(*args)
        return loss

    def f_df(self, theta, X):
        # Reshape theta -> args
        args = args_from_vector(theta, self.params)
        args += [X]
        # Compute gradient from args
        grad = vector_from_args(self.grads(*args))
        loss = self._loss(*args)
        return loss,grad

    def get_params(self):
        return vector_from_args(self.params)

    def get_args(self, theta):
        args = args_from_vector(theta, self.params)
        return args

    def get_weights(self, theta):
        args = args_from_vector(theta, self.params)
        for arg in args:
            if arg.ndim == 2:
                return arg

    def init_weights(self, shape, dtype):
        self.model._initialize_weights_sparse(shape, sparse_init=20)
        self.model._initialize_hidbias()
        self.model._initialize_visbias(shape)
        self.setup()


    def _get_gradients(self, X, Y=None, ** kwargs):
        try:
            if Y is None:
                cost = self.model.cost(X=X, **kwargs)
            else:
                cost = self.model.cost(X=X, Y=Y, **kwargs)
        except TypeError,e:
            # If anybody knows how to add type(seslf) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            e.message += " while calling "+str(type(self))+".__call__"
            print str(type(self))
            print e.message
            raise e

        if cost is None:
            raise NotImplementedError(str(type(self))+" represents an intractable "
                    " cost and does not provide a gradient approximation scheme.")

        params = list(self.model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore')

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        return gradients, updates


if __name__ == '__main__':
    # Load train obj
    f = np.load('test.npz')
    model = f['model'].item()
    cost = f['cost'].item()
    m = ModelGradient(model, cost)
    p = model.weights.shape.eval()[0]
    X = np.random.randn(100,p).astype(np.float32)
    theta = m.get_params()
    m.f_df(theta, X)
    1/0

    # Load cost function
    W = np.random.randn(10,10)
    W = theano.shared(W)

    cost = (W**2).sum()
    grad = T.grad(cost, W)

    params = [W]
    1/0
