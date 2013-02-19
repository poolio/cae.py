import numpy as np

#XXX: Shamefully stolen from pyautodiff
# https://github.com/jaberg/pyautodiff/blob/master/autodiff/fmin_scipy.py

def vector_from_args(args):
    args_sizes = [w.size for w in args]
    x_size = sum(args_sizes)
    x = np.empty(x_size, dtype='float64') # has to be float64 for fmin_l_bfgs_b
    i = 0
    for w in args:
        x[i: i + w.size] = w.flatten()
        i += w.size
    return x

def args_from_vector(x, orig_args):
    # unpack x_opt -> args-like structure `args_opt`
    rval = []
    i = 0
    for w in orig_args:
        rval.append(x[i: i + w.size].reshape(w.shape).astype(w.dtype))
        i += w.size
    return rval
