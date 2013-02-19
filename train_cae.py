import sys
import numpy as np
import matplotlib.pyplot as plt
from cae import CAE
from sfo.sfo import sfo
from plot_patches import plot_patches


def fit_sgd(model, X, batch_size=20, epochs=10, learning_rate=0.1, verbose=False, callback=None):
    """
    Fit the model to the data X.
    
    Parameters
    ----------
    X: array-like, shape (n_examples, n_inputs)
        Training data, where n_examples in the number of examples
        and n_inputs is the number of features.
    """

    inds = range(X.shape[0])
    np.random.shuffle(inds)
    n_batches = len(inds) / batch_size
    theta = model.get_params()
    for epoch in range(epochs):
        loss = 0.
        for minibatch in range(n_batches):
            lossi, gradi = model.f_df(theta, X[inds[minibatch::n_batches]])
            theta -= learning_rate * gradi
            loss += lossi
        if verbose:
            print "Epoch %d, Loss = %.2f" % (epoch, loss / len(inds))
            sys.stdout.flush()
        if callback != None:
            callback(epoch)
    return theta

def fit_sfo(model, X, num_batches=10, epochs=10, **kwargs):
    xinit = model.get_params()
    costs = []
    # Create minibatches.
    # XXX: would be nice to have SFO automatically call f_df with appropriate subindicies of data
    # instead of having to incorporate that functionality into f_df (e.g. with X[idx...])
    batches = []
    for mb in xrange(num_batches):
        batches.append(X[mb::num_batches])
    optimizer = sfo(model.f_df, xinit, batches, **kwargs)
    for learning_step in range(epochs):
        x = optimizer.optimize(num_passes=1)
        #cost, grad = optimizer.full_F_dF()
        cost = model.loss(X)/X.shape[0]
        costs.append(cost)
        print learning_step, 'costs=', costs
    return x

def mnist_demo():
    f = np.load('/Users/poole/mnist_train.npz')
    X = f['X']
    cae = CAE(n_hiddens=256, W=None, c=None, b=None, jacobi_penalty=1.0)
    cae.init_weights(X.shape[1], X.dtype)
    #theta_sgd = fit_sgd(cae, X, verbose=True)
    theta_sfo = fit_sfo(cae, X, 100, 10, regularization='min', max_history_terms=2, subspace_dimensionality=101)

    cae.set_params(theta_sgd)
    loss = cae.loss(X)
    plot_patches(cae.W)

if __name__ == '__main__':
    mnist_demo()
