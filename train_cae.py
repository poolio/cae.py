import numpy as np
import cae
from plot_patches import plot_patches

f = np.load('/Users/poole/mnist_train.npz')
X = f['X']
cae = cae.CAE(n_hiddens=256, W=None, c=None, b=None, learning_rate=0.1, jacobi_penalty=1.0, batch_size=50, epochs=10)
cae.fit(X, verbose=True)
plot_patches(cae.W)
