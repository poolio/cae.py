def plot_patches(W, width=None, height=None, cols=8):
    import numpy as np
    import matplotlib.pyplot as plt
    p, n = W.shape

    if width is None:
        width = np.rint(np.sqrt(p))
        height = width

    # Normalize patches (XXX: do this)
    #cols = 8

    dimh = height
    dimw = width
    dimhp = height + 1
    dimwp = width + 1
    rows = np.int(n/cols)
    I = np.ones((height*rows+rows-1, width*cols+cols-1))
    for i in xrange(rows):
        for j in xrange(cols):
            I[i*dimhp:i*dimhp+dimh, j*dimwp:j*dimwp+dimw] = W[:,i*cols+j].reshape(height, width)

    plt.figure(77)
    plt.clf()
    plt.imshow(I, cmap='gray').set_interpolation('nearest')
    plt.draw();plt.show()

if __name__ == '__main__':
    import gzip
    import cPickle

    f = gzip.open('../data/natural.pkl.gz', 'rb')
    train,val,test=cPickle.load(f)
    plot_patches(train[0][:500,:].T)
