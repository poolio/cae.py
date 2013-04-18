#
# Train CAE using SGD, L-BFGS, or SFO
#
import sys
import numpy as np
import matplotlib.pyplot as plt
from cae import CAE
from sfo import SFO
from sfo import SAG
from visualization import plot_weights, save_weights
from model_gradient import ModelGradient

def fit_adagrad(model, X, batch_size=100, epochs=30, learning_rate=0.1, verbose=False, callback=None):
    inds = range(X.shape[0])
    np.random.shuffle(inds)
    n_batches = len(inds) / batch_size
    theta = model.get_params()

    num_batches = X.shape[0]/batch_size
    batches = []
    for mb in xrange(num_batches):
        batches.append(X[mb::num_batches])

    #model.W /= np.sqrt((model.W**2).sum(0))
    #save_weights(model.W, 'test%d.png'%(0))

    thresh = 3.

    reps = 1
    grad_history = np.ones_like(theta)
    K=5
    P = np.random.randn(K, theta.size)
    Ptheta = np.zeros((K, epochs*n_batches))
    for epoch in range(epochs):
        #Normalize weights
        #theta = model.get_params()
        loss = 0.
        for minibatch in range(n_batches):
            idx = np.random.randint(n_batches)
            gradii = np.zeros_like(theta)
            lossii = 0.
            for i in xrange(reps):
                lossi, gradi = model.f_df(theta, batches[idx])
                lossii+= lossi / reps
                gradii += gradi / reps

            learning_rates = learning_rate / (np.sqrt(grad_history))
            learning_rates[np.isinf(learning_rates)] = learning_rate
            theta -= learning_rates * gradii
            #print 'largest step=', np.max(np.abs(learning_rates*gradii))
#            theta[theta > thresh] = thresh
#            theta[theta < -thresh] = -thresh
            #print 'largest theta = ', np.max(np.abs(theta)), 'largest lr', np.max(learning_rates), 'min hist', np.min(grad_history)
            grad_history += gradi**2
            #print np.mean(gradi**2), np.mean(grad_history)-1
            loss += lossii
            Ptheta[:, epoch*n_batches+minibatch] = np.dot(P, theta)
        if verbose:
            cost, fff = model.f_df(theta, X)
            print "Epoch %d, Online Loss = %.2f, Loss = %.2f" % (epoch, cost, loss/n_batches)
            sys.stdout.flush()
        if callback != None:
            callback(epoch)
        #save_weights(model.W, 'test%d.png'%(epoch+1))
    return theta


def fit_sgd(model, X, batch_size=100, epochs=30, learning_rate=0.1, verbose=False, callback=None):
    inds = range(X.shape[0])
    np.random.shuffle(inds)
    n_batches = len(inds) / batch_size
    theta = model.get_params()

    num_batches = X.shape[0]/batch_size
    batches = []
    for mb in xrange(num_batches):
        batches.append(X[mb::num_batches])

    #model.W /= np.sqrt((model.W**2).sum(0))
    #save_weights(model.W, 'test%d.png'%(0))
    for epoch in range(epochs):
        #Normalize weights
    #    theta = model.get_params()
        loss = 0.
        for minibatch in range(n_batches):
            idx = np.random.randint(n_batches)
            lossi, gradi = model.f_df(theta, batches[idx])
            theta -= learning_rate * gradi
            loss += lossi
        if verbose:
            cost,gradblah = model.f_df(theta, X)
            print "Epoch %d, Online Loss = %.2f, Loss = %.2f" % (epoch, cost, loss/n_batches)
            sys.stdout.flush()
        if callback != None:
            callback(epoch)
    #    save_weights(model.W, 'test%d.png'%(epoch+1))
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
        cost = model.loss(X)
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
        cost = model.loss(X)
        costs.append(cost)
        print 'Epoch %d, Loss = %.2f' % (learning_step, cost)
    return x


def mnist_demo():
    epochs = 15
    num_batches = 100
    # Load data
    f = np.load('/home/poole/mnist_train.npz')
    X = f['X']
    #X = X[:10000,:]

    #f2 = np.load('/home/poole/code/scripts/test.npz')
    #model = f2['model'].item()
    #cost = f2['cost'].item()
    #cae = ModelGradient(model, cost)
    cae = CAE(n_hiddens=256, W=None, c=None, b=None, jacobi_penalty=1.00)
#    cae.init_weights_from_data(X)
    # Train SGD
    cae.init_weights(X.shape[1], dtype=np.float32)
    theta_sgd = fit_adagrad(cae, X, epochs=epochs, verbose=True, learning_rate=0.2, batch_size=20)
    #cae.init_weights(X.shape[1], dtype=np.float64)
    #theta_sgd = fit_sgd(cae, X, epochs=epochs, verbose=True, learning_rate=0.2)
    plot_weights(cae.W); plt.title('Adagrad')
    # Train SFO
    cae.init_weights(X.shape[1], dtype=np.float64)
    theta_sfo = fit_sfo(cae, X, num_batches, epochs)
    plot_weights(cae.W), plt.title('SFO')

    1/0

if __name__ == '__main__':
    mnist_demo()
