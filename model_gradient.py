import theano
import theano.tensor as T
import numpy as np
from utils import vector_from_args, args_from_vector

class ModelGradient:
    def __init__(self, model, cost):
        self.X = T.matrix('X')
        init_grads, updates = cost.get_gradients(model, self.X)
        params = model.get_params()
        # We need to replace parameters with purely symbolic variables in case some are shared
        symbolic_params = [self._convert_variable(param) for param in params]
        givens = dict(zip(params, symbolic_params))
        # Create gradient and cost functions
        self.params = params
        self.symbolic_params = symbolic_params
        self.grads = theano.function(symbolic_params + [self.X], [init_grads[param] for param in params], givens=givens)
        self.loss = theano.function(symbolic_params + [self.X], cost(model, self.X), givens=givens)
        # Maps params -> their derivative

    def _convert_variable(self, x):
        if x.ndim == 1:
            return T.vector(x.name, dtype=x.dtype)
        else:
            return T.matrix(x.name, dtype=x.dtype)


    def f_df(self, theta, X):
        # Reshape theta -> args
        args = args_from_vector(theta, self.params)
        args += [X]
        # Compute gradient from args
        grad = vector_from_args(self.grads(*args))
        loss = self.loss(*args)
        return loss,grad

    def get_params(self):
        return vector_from_args(self.params)


if __name__ == '__main__':
    # Load train obj
    f = np.load('test.npz')
    model = f['model'].item()
    cost = f['cost'].item()
    m = ModelGradient(model, cost)
    p = model.weights.shape.eval()[0]
    X = np.random.randn(10,p)
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
