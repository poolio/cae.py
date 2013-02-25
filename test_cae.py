import numpy as np
from cae import CAE

eps = 1e-6
n_samples = 10
n_vis = 2
n_hid = 3
cae = CAE(n_hiddens=n_hid, jacobi_penalty=0.0)
cae.init_weights(n_vis, dtype=np.float64)

theta0 = cae.get_params()
relerror = np.zeros_like(theta0)
raterror = np.zeros_like(theta0)

for idx in xrange(theta0.size):
    print float(idx)/theta0.size
    theta1 = theta0.copy()
    theta1[idx] += eps

    X = np.random.randn(n_samples, n_vis)

    f0, g0 = cae.f_df(theta0, X)
    f01, g01 = cae.f_df((theta0+theta1)/2., X)
    f1, g1 = cae.f_df(theta1, X)

    df = (f1-f0)/eps
    relerror[idx] = (df-g01[idx])/df
    raterror[idx] = df/g01[idx]

print relerror
print raterror
print np.max(np.abs(relerror))
print np.max(np.abs(raterror))
