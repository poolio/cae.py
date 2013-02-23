#
# Train CAE using SGD, L-BFGS, or SFO
#
import sys
import numpy as np
import matplotlib.pyplot as plt
from cae import CAE
from sfo import SFO
from sfo import SAG
from plot_patches import plot_patches

def fit_sgd(model, X, batch_size=20, epochs=10, learning_rate=0.1, verbose=False, callback=None):
    inds = range(X.shape[0])
    np.random.shuffle(inds)
    n_batches = len(inds) / batch_size
    theta = model.get_params()

    num_batches = X.shape[0]/batch_size
    batches = []
    for mb in xrange(num_batches):
        batches.append(X[mb::num_batches])

    for epoch in range(epochs):
        loss = 0.
        for minibatch in range(n_batches):
            idx = np.random.randint(n_batches)
            lossi, gradi = model.f_df(theta, batches[idx])
            theta -= learning_rate * gradi
            loss += lossi
        if verbose:
            cost = model.loss(X)/X.shape[0]
            print "Epoch %d, Approx Loss = %.2f, Loss = %.2f" % (epoch, cost, loss/X.shape[0])
            sys.stdout.flush()
        if callback != None:
            callback(epoch)
    return theta

def fit_lbfgs(model, X, batch_size=1000, epochs=10, learning_rate=0.1, verbose=False, callback=None):
    from scipy.optimize import fmin_l_bfgs_b
    inds = range(X.shape[0])
    np.random.shuffle(inds)
    n_batches = len(inds) / batch_size
    theta = model.get_params()
    for epoch in range(epochs):
        for minibatch in range(n_batches):
            print '.'
            theta = fmin_l_bfgs_b(model.f_df, theta, args=(X[inds[minibatch::n_batches]],),maxfun=20)[0]
        if verbose:
            loss = model.loss(X)
            print "Epoch %d, Loss = %.2f" % (epoch, loss / len(inds))
            sys.stdout.flush()
        if callback != None:
            callback(epoch)
    return theta

def fit_sfo(model, X, num_batches, epochs, **kwargs):
    xinit = model.get_params()
    costs = []
    # Create minibatches.
    # XXX: might be nice to have SFO automatically call f_df with appropriate subindicies of data
    # instead of having to incorporate that functionality into f_df (e.g. with X[idx...])
    batches = []
    for mb in xrange(num_batches):
        batches.append(X[mb::num_batches])
    optimizer = SFO(model.f_df, xinit, batches, **kwargs)
    for learning_step in range(epochs):
        x = optimizer.optimize(num_passes=1)
        #cost, grad = optimizer.full_F_dF()
        cost = model.loss(X)/X.shape[0]
        costs.append(cost)
        print 'Epoch %d, Loss = %.2f' % (learning_step, cost)
    return x

def fit_sag(model, X, num_batches, epochs, **kwargs):
    xinit = model.get_params()
    costs = []
    # Create minibatches.
    # XXX: might be nice to have SFO automatically call f_df with appropriate subindicies of data
    # instead of having to incorporate that functionality into f_df (e.g. with X[idx...])
    batches = []
    num_batches = X.shape[0]/20
    for mb in xrange(num_batches):
        batches.append(X[mb::num_batches])
    optimizer = SAG(model.f_df, xinit, batches, L=10.0)# **kwargs)
    for learning_step in range(epochs):
        x = optimizer.optimize(num_passes=1)
        #cost, grad = optimizer.full_F_dF()
        cost = model.loss(X)/X.shape[0]
        costs.append(cost)
        print 'Epoch %d, Loss = %.2f' % (learning_step, cost)
    return x


def mnist_demo():
    epochs = 10
    num_batches = 100
    # Load data
    f = np.load('/home/poole/mnist_train.npz')
    X = f['X']
    X = X[:10000,:]

    cae = CAE(n_hiddens=256, W=None, c=None, b=None, jacobi_penalty=1.0)
    #Train SGD
    cae.init_weights_from_data(X)
    from pylearn2.utils import serial
    f2 = serial.load('/home/poole/models/mnist_comp/rae.pkl')
    #cae.W = f2.weights.eval()
    #cae.c = f2.visbias.eval()
    #cae.b = f2.hidbias.eval()
 #   theta_sgd = fit_sgd(cae, X, epochs=epochs, verbose=True)
    #Train SFO
   # cae.init_weights(X.shape[1], dtype=np.float32)
   # theta_sfo = fit_sfo(cae, X, num_batches, epochs, regularization='min', max_history_terms=2)
    theta_lbfgs= fit_sag(cae, X, num_batches, epochs)
    #theta_lbfgs= fit_lbfgs(cae, X, verbose=True)

    # Visualize parameters
    cae.set_params(theta_lbfgs)
    plot_patches(cae.W), plt.title('lbfgs')

    cae.set_params(theta_sfo)
    plot_patches(cae.W), plt.title('SFO')

    cae.set_params(theta_sgd)
    plot_patches(cae.W), plt.title('SGD')

    1/0

if __name__ == '__main__':
    mnist_demo()
