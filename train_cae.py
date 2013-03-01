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
def fit_sgd(model, X, batch_size=20, epochs=30, learning_rate=0.1, verbose=False, callback=None):

    inds = range(X.shape[0])
    np.random.shuffle(inds)
    n_batches = len(inds) / batch_size
    theta = model.get_params()

    num_batches = X.shape[0]/batch_size
    batches = []
    for mb in xrange(num_batches):
        batches.append(X[mb::num_batches])

    model.W /= np.sqrt((model.W**2).sum(0))
    save_weights(model.W, 'test%d.png'%(0))
    for epoch in range(epochs):
        #Normalize weights
        theta = model.get_params()
        loss = 0.
        for minibatch in range(n_batches):
            idx = np.random.randint(n_batches)
            lossi, gradi = model.f_df(theta, batches[idx])
            theta -= learning_rate * gradi
            loss += lossi
        if verbose:
            cost = model.loss(X)
            print "Epoch %d, Online Loss = %.2f, Loss = %.2f" % (epoch, cost, loss/n_batches)
            sys.stdout.flush()
        if callback != None:
            callback(epoch)
        save_weights(model.W, 'test%d.png'%(epoch+1))
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
            print "Epoch %d, Loss = %.4f" % (epoch, loss / len(inds))
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

    #optimizer.check_grad()

    for learning_step in range(epochs):
        x = optimizer.optimize(num_passes=1)
        #cost, grad = optimizer.full_F_dF()
        cost = model.loss(X)
        costs.append(cost)
        print 'Epoch %d, Loss = %.4f' % (learning_step, cost)
    return x


def f_df_wrapper(*args, **kwargs):
    global plpp_hist, plpp, cae, true_f_df
    plpp_hist.append( np.dot( plpp, args[0]) )
    return true_f_df(*args, **kwargs)


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
    global plpp_hist, plpp, true_f_df

    np.random.seed(5432109876) # make experiments repeatable


    epochs = 20
    num_batches = 50
    # Load data
    try:
        f = np.load('/home/poole/mnist_train.npz')
    except:
        f = np.load('mnist_train.npz')
    X = f['X']
    X = X[:10000,:]

    X = np.random.permutation(X)

    #X = X[:4000]

    cae = CAE(n_hiddens=256, W=None, c=None, b=None, jacobi_penalty=1.00)

    # put a wrapper around the objective and gradient so can store a history
    # of parameter values
    true_f_df = cae.f_df
    cae.f_df = f_df_wrapper
    # random projections to observe learning
    cae.init_weights(X.shape[1], dtype=np.float64)

    plpp = np.random.randn( 4, cae.get_params().shape[0] ) / np.sqrt(cae.get_params().shape[0])
    plpp_hist = []

    #Train SGD
    cae.init_weights(X.shape[1], dtype=np.float64)
    theta_sgd = fit_sgd(cae, X, epochs=epochs, verbose=True, learning_rate=0.2)
    plot_weights(cae.W), plt.title('SGD')
    # save and reset the history of parameter updates
    plpp_hist_sgd = plpp_hist
    plpp_hist = []

    #Train SFO
    # Train SFO
    cae.init_weights(X.shape[1], dtype=np.float64)
    theta_sfo = fit_sfo(cae, X, num_batches, epochs)
    plot_weights(cae.W); plt.title('SFO')
    plpp_hist_sfo = plpp_hist
    plpp_hist = []

    plt.figure(1004, figsize=(15,15))
    plt.clf()
    for nm in ('sfo', 'sgd'):
        if nm == 'sfo':
            plpp_hist = plpp_hist_sfo
            c = 'b.'
        else:
            plpp_hist = plpp_hist_sgd
            c = 'r.'

        plpc = np.array(plpp_hist).reshape((-1,plpp.shape[0])).T
        #print plpcc.shape
        #print array(locind_hist).shape
        for pii in range(plpc.shape[0]):
            for pjj in range(plpc.shape[0]):
                plt.subplot( plpc.shape[0], plpc.shape[0], pjj + pii*plpc.shape[0] + 1 )
                plt.plot( plpc[pjj,:-1], plpc[pii,:-1], c, label=nm )
        for pii in range(plpc.shape[0]):
            for pjj in range(plpc.shape[0]):
                plt.subplot( plpc.shape[0], plpc.shape[0], pjj + pii*plpc.shape[0] + 1 )
                plt.plot( plpc[pjj,[-1]], plpc[pii,[-1]], 'yo', label="%s current"%(nm) )
    plt.legend( loc='best' )
    plt.suptitle( 'Full history')
    plt.show()
    
if __name__ == '__main__':
    mnist_demo()
