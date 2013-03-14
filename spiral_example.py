import numpy as np
import matplotlib.pyplot as plt
from cae import CAE
from train_cae import fit_adagrad, fit_sgd

def generate_data(n=10000):
    t = 12*np.random.rand(n) + 3
    x = (t)*0.04*np.sin(t)
    y = (t)*0.04*np.cos(t)
    X = np.vstack((x,y)).T
    return X



X = generate_data()
cae = CAE(n_hiddens=1000, W=None, c=None, b=None, jacobi_penalty=0.0)
cae.init_weights(X.shape[1], dtype=np.float64)
theta_sgd = fit_sgd(cae, X, epochs=30, verbose=True, learning_rate=0.1)


lim = 0.5
lims = np.arange(-lim, lim, 0.1)
x, y = np.meshgrid(lims, lims)
gridX = np.vstack( (x.flatten(), y.flatten())).T
rX = cae.reconstruct(gridX)
dX = rX-gridX

plt.close('all')
plt.scatter(X[:,0],X[:,1])
plt.quiver(gridX[:,0], gridX[:, 1], dX[:, 0], dX[:,1])
plt.show()

#plt.show()
