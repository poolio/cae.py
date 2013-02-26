import numpy as np
import matplotlib.pyplot as plt
import Image

def weight_image(W, width=None, height=None, cols=None, sort=False):
    p, n = W.shape
    if width is None:
        width = np.rint(np.sqrt(p))
        height = width
    if cols is None:
        cols = int(np.sqrt(n))
    # Sort by norm
    if sort:
        sidx = np.argsort((W**2).sum(0))[::-1]
        W = W[:,sidx]
    dimh = height
    dimw = width
    dimhp = height + 1
    dimwp = width + 1
    rows = np.int(n/cols)
    I = np.ones((height*rows+rows-1, width*cols+cols-1))
    for i in xrange(rows):
        for j in xrange(cols):
            I[i*dimhp:i*dimhp+dimh, j*dimwp:j*dimwp+dimw] = W[:,i*cols+j].reshape(height, width)
    return I

def save_weight_image(W, fn, sort=False):
    I = weight_image(W, sort=sort)
    plt.imsave(fn, I, cmap='gray')

def plot_weights(W, sort=False):
    I = weight_image(W, sort=sort)
    plt.figure()
    plt.clf()
    plt.imshow(I, cmap='gray').set_interpolation('nearest')
    plt.draw();plt.show()
    plt.axis('off')

if __name__ == '__main__':
    import gzip
    import cPickle
    f = gzip.open('../data/natural.pkl.gz', 'rb')
    train,val,test=cPickle.load(f)
    plot_patches(train[0][:500,:].T)
