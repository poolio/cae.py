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
from collections import defaultdict

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


def fit_sgd(model, X, num_batches=100, epochs=30, learning_rate=0.1, verbose=False, callback=None, **kwargs):
    inds = range(X.shape[0])
    np.random.shuffle(inds)
    n_batches = num_batches #len(inds) / batch_size
    theta = model.get_params()

    #num_batches = X.shape[0]/batch_size
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
            lossi, gradi = model.f_df(theta, batches[idx], **kwargs)
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
            print "Epoch %d, Loss = %.4f" % (epoch, loss / len(inds))
            sys.stdout.flush()
        if callback != None:
            callback(epoch)
    return theta

def fit_sfo(model, X, num_batches, epochs, kwargs_pt=None, **kwargs):
    xinit = model.get_params()
    costs = []
    # Create minibatches.
    # XXX: might be nice to have SFO automatically call f_df with appropriate subindicies of data
    # instead of having to incorporate that functionality into f_df (e.g. with X[idx...])
    batches = []
    for mb in xrange(num_batches):
        batches.append(X[mb::num_batches])
    print "initializing SFO"
    optimizer = SFO(model.f_df, xinit, batches, kwargs=kwargs_pt, **kwargs)

    #optimizer.check_grad()

    print "calling SFO"
    for learning_step in range(epochs):
        theta = optimizer.optimize(num_passes=1)
        #cost, grad = optimizer.full_F_dF()
        cost = model.loss(X)
        costs.append(cost)
        print 'Epoch %d, Loss = %.4f' % (learning_step, cost)
    return theta


def f_df_wrapper(*args, **kwargs):
    global plpp_hist, plpp, learner_name, cae, true_f_df, f_hist
    plpp_hist[learner_name].append( np.dot( plpp, args[0]) )
    f_hist[learner_name].append(kwargs['model_pt'].loss(kwargs['X_pt']))
    kk = dict(kwargs)
    del kk['model_pt']
    del kk['X_pt']
    return true_f_df(*args, **kk)


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


def make_figures():
    global plpp_hist, f_hist, learner_name, plpp, true_f_df


    minf = np.inf
    for nm in f_hist.keys():
        ff = np.asarray(f_hist[nm])
        minf2 = np.min(ff)
        minf = np.min([minf, minf2])
        
    plt.figure()
    plt.clf()
    for nm in sorted(plpp_hist.keys()):
        ff = np.asarray(f_hist[nm])
        ff[ff>ff[0]] = np.nan
        plt.semilogy( ff-minf, label=nm )
    plt.ylabel( 'Full batch objective - minimum' )
    plt.xlabel( 'Function calls' )
    plt.legend( loc='best' )
    plt.suptitle( 'Objective function')
    plt.show()

        
    plt.figure()
    plt.clf()
    for nm in sorted(plpp_hist.keys()):
        ff = np.asarray(f_hist[nm])
        ff[ff>ff[0]] = np.nan
        plt.semilogy( ff, label=nm )
    plt.ylabel( 'Full batch objective' )
    plt.xlabel( 'Function calls' )
    plt.legend( loc='best' )
    plt.suptitle( 'Objective function')
    plt.show()


    plt.figure(figsize=(15,15))
    plt.clf()
    for nm in plpp_hist.keys():
        #if nm == 'sfo':
        #    c = 'b.'
        #else:
        #    c = 'r.'

        plpc = np.array(plpp_hist[nm]).reshape((-1,plpp.shape[0])).T
        #print plpcc.shape
        #print array(locind_hist).shape
        for pii in range(plpc.shape[0]):
            for pjj in range(plpc.shape[0]):
                plt.subplot( plpc.shape[0], plpc.shape[0], pjj + pii*plpc.shape[0] + 1 )
                #plt.plot( plpc[pjj,:-1], plpc[pii,:-1], c, label=nm )
                plt.plot( plpc[pjj,:-1], plpc[pii,:-1], label=nm )
        for pii in range(plpc.shape[0]):
            for pjj in range(plpc.shape[0]):
                plt.subplot( plpc.shape[0], plpc.shape[0], pjj + pii*plpc.shape[0] + 1 )
                plt.plot( plpc[pjj,[-1]], plpc[pii,[-1]], 'yo', label="%s current"%(nm) )
    plt.legend( loc='best' )
    plt.suptitle( 'Full history')
    plt.show()


def init_cae(cae, X):
    np.random.seed(32314098976) # make experiments repeatable
    cae.init_weights(X.shape[1], dtype=np.float64)
    # TODO(jascha) save and restore the old seed
    #np.random.seed(sd)


def mnist_demo():
    global plpp_hist, f_hist, learner_name, plpp, true_f_df

    np.random.seed(5432109876) # make experiments repeatable

    epochs = 40
    num_batches = 100
    # Load data
    try:
        f = np.load('/home/poole/mnist_train.npz')
    except:
        f = np.load('mnist_train.npz')
    X = f['X']
    #X = X[:10000,:]

    X = np.random.permutation(X)

    #cae = CAE(n_hiddens=256, W=None, c=None, b=None, jacobi_penalty=1.00)

    # random projections to observe learning
    plpp = np.random.randn( 4, cae.get_params().shape[0] ) / np.sqrt(cae.get_params().shape[0])
    plpp_hist = defaultdict(list)
    f_hist = defaultdict(list)

    #f2 = np.load('/home/poole/code/scripts/test.npz')
    model = f2['model'].item()
    cost = f2['cost'].item()
    cae = ModelGradient(model, cost)

    # put a wrapper around the objective and gradient so can store a history
    # of parameter values
    true_f_df = cae.f_df
    cae.f_df = f_df_wrapper

    init_cae(cae, X)
    theta_sgd = fit_adagrad(cae, X, epochs=epochs, verbose=True, learning_rate=0.2, batch_size=20)
    plot_weights(cae.W), plt.title(learner_name), plt.draw()
    make_figures()


    #Train SFO
    kwargs_pt = {'model_pt':cae, 'X_pt':X}

    learner_name = 'SFO + nat grad + adapt eta'
    print learner_name
    init_cae(cae, X)
    theta_sfo = fit_sfo(cae, X, num_batches, epochs, kwargs_pt=kwargs_pt, natural_gradient=True, regularization = 'min', adapt_eta=True)
    plot_weights(cae.W), plt.title(learner_name), plt.draw()
    make_figures()

    learner_name = 'SFO + nat grad'
    print learner_name
    init_cae(cae, X)
    theta_sfo = fit_sfo(cae, X, num_batches, epochs, kwargs_pt=kwargs_pt, natural_gradient=True, regularization = 'min')
    plot_weights(cae.W), plt.title(learner_name), plt.draw()
    make_figures()

    learner_name = 'SFO'
    print learner_name
    init_cae(cae, X)
    theta_sfo = fit_sfo(cae, X, num_batches, epochs, kwargs_pt=kwargs_pt, natural_gradient=False, regularization = 'min')
    plot_weights(cae.W), plt.title(learner_name), plt.draw()

    # save and reset the history of parameter updates

    #Train SGD
    learning_rates = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6,)

    for learning_rate in learning_rates:
        learner_name = "SGD %g"%learning_rate
        print learner_name
        init_cae(cae, X)
        theta_sgd = fit_sgd(cae, X, num_batches=num_batches, epochs=epochs, verbose=True, learning_rate=2., model_pt=cae, X_pt=X)
        plot_weights(cae.W), plt.title(learner_name), plt.draw()
    
if __name__ == '__main__':
    mnist_demo()
