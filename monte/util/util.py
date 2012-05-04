#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from numpy import ones, dot, newaxis, log, exp, size, sqrt, mean, std,\
                  floor, ceil, sum, vstack, hstack, where, isscalar
from pylab import find, zeros, reshape, imshow, show


def logsumexp(x, dim=-1):
    """Compute log(sum(exp(x))) in a numerically stable way.
    
       Use second argument to specify along which dimensions the logsumexp
       shall be computed. If -1 (which is also the default), logsumexp is 
       computed along the last dimension. 
    """
    if len(x.shape) < 2:  #only one possible dimension to sum over?
        xmax = x.max()
        return xmax + log(sum(exp(x-xmax)))
    else:
        if dim != -1:
            x = x.transpose(range(dim) + range(dim+1, len(x.shape)) + [dim])
        lastdim = len(x.shape)-1
        xmax = x.max(lastdim)
        return xmax + log(sum(exp(x-xmax[...,newaxis]),lastdim))


#def logsumexp_lastdim(x):
#    """Compute log(sum(exp(x))) in numerically stable way along the last 
#       dimension."""
#    dim = len(x.shape)-1
#    xmax = x.max(dim)
#    return xmax + log(sum(exp(x-xmax[...,newaxis]),dim))


def onehot(x,numclasses=None):
    """ Convert integer encoding for class-labels (starting with 0 !)
        to one-hot encoding. 
      
        If numclasses (the number of classes) is not provided, it is assumed 
        to be equal to the largest class index occuring in the labels-array + 1.
        The output is an array who's shape is the shape of the input array plus
        an extra dimension, containing the 'one-hot'-encoded labels. 
    """
    if x.shape==():
        x = x[newaxis]
    if numclasses is None:
        numclasses = x.max() + 1
    result = zeros(list(x.shape) + [numclasses])
    z = zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[where(x==c)] = 1
        result[...,c] += z
    return result


def unhot(labels):
    """ Convert one-hot encoding for class-labels to integer encoding 
        (starting with 0!).

        The input-array can be of any shape. The one-hot encoding is assumed 
        to be along the last dimension.
    """
    return labels.argmax(len(labels.shape)-1)


def checkgrad(f,g,x,e,RETURNGRADS=False,*args):
    from pylab import norm
    """Check correctness of gradient function g at x by comparing to numerical
       approximation using perturbances of size e. Simple adaptation of 
       Carl Rasmussen's matlab-function checkgrad."""
    dy = g(x,*args)
    if isscalar(x):
        dh = zeros(1,dtype=float)
        l = 1
    else:
        l = len(x)
        dh = zeros(l,dtype=float)
    for j in range(l):
        dx = zeros(l,dtype=float)
        dx[j] = e
        y2 = f(x+dx,*args)
        y1 = f(x-dx,*args)
        dh[j] = (y2 - y1)/(2*e)
    print "analytic: \n", dy
    print "approximation: \n", dh
    if RETURNGRADS: return dy,dh
    else: return norm(dh-dy)/norm(dh+dy)


def checkmodelgrad(model,e,RETURNGRADS=False,*args):
    from pylab import norm
    """Check the correctness of passed-in model in terms of cost-/gradient-
       computation, using gradient approximations with perturbances of 
       size e. 
    """
    def updatemodelparams(model, newparams):
        model.params *= 0.0
        model.params += newparams.copy()
    def cost(params,*args):
        paramsold = model.params.copy()
        updatemodelparams(model,params.copy().flatten())
        result = model.cost(*args) 
        updatemodelparams(model,paramsold.copy())
        return result
    def grad(params,*args):
        paramsold = model.params.copy()
        updatemodelparams(model, params.copy().flatten())
        result = model.grad(*args)
        updatemodelparams(model, paramsold.copy())
        return result
    dy = model.grad(*args)
    l = len(model.params)
    dh = zeros(l,dtype=float)
    for j in range(l):
        dx = zeros(l,dtype=float)
        dx[j] = e
        y2 = cost(model.params+dx,*args)
        y1 = cost(model.params-dx,*args)
        dh[j] = (y2 - y1)/(2*e)
    print "analytic: \n", dy
    print "approximation: \n", dh
    if RETURNGRADS: return dy,dh
    else: return norm(dh-dy)/norm(dh+dy)


def differentiate(f,e,args):
    def grad(x):
        grad = zeros(len(x),dtype=float)
        for j in range(len(x)):
          dx = zeros(len(x),dtype=float)
          dx[j] = e
          y2 = f(x+dx,*args)
          y1 = f(x-dx,*args)
          grad[j] = (y2-y1)/(2*e)
        return grad
    return grad


def dispims(M, height, width, border=0, bordercolor=0.0, **kwargs):
    """ Display a whole stack (colunmwise) of vectorized matrices. Useful 
        eg. to display the weights of a neural network layer.
    """
    from pylab import cm, ceil
    numimages = M.shape[1]
    n0 = int(ceil(sqrt(numimages)))
    n1 = int(ceil(sqrt(numimages)))
    im = bordercolor*\
         ones(((height+border)*n1+border,(width+border)*n0+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[j*(height+border)+border:(j+1)*(height+border)+border,\
                   i*(width+border)+border:(i+1)*(width+border)+border] = \
                vstack((\
                  hstack((reshape(M[:,i*n1+j],(width,height)).T,\
                         bordercolor*ones((height,border),dtype=float))),\
                  bordercolor*ones((border,width+border),dtype=float)\
                  ))
    imshow(im.T,cmap=cm.gray,interpolation='nearest', **kwargs)
    show()


def subsample(im, xout, yout):
    """ Subsample an image. """
    xin, yin = im.shape
    return im[1:-1:floor(xin/xout),1:-1:yin/yout]


def princomps(data,numprincomps):
    """ Compute the first numprincomps principal components of 
        (columnwise represented) dataset data.
        Returns the transformation (first numprincomps eigenvectors as matrix)
        and the resulting low-dimensional codes.
    """
    from pylab import cov
    from numpy.linalg import eigh
    m = data.mean(1)[:,newaxis]
    u,v=eigh(cov(data-m,rowvar=1,bias=1))
    V = ((u**(-0.5))[-numprincomps:][newaxis,:]*v[:,-numprincomps:]).T
    W = ((u**(0.5))[-numprincomps:][newaxis,:]*v[:,-numprincomps:])
    return V, dot(V,data-m)


def standardize(x,RETURNMEANANDSTD=False):
    """ Change data to zero-mean and standard-deviation 1.0 in each dimension.
        Datapoints are assumed to be represented column-wise in input 
        array x."""
    m = mean(x,1)[:,newaxis]
    s = std(x,1)
    s[find(s<0.000001)] = 1.0
    s = s[:,newaxis]
    if not RETURNMEANANDSTD: return (x - m)/s
    else: return (x - m)/s, m, s
  

def unstandardize(x,m,s):
    """ Change standardized data back to normal.
        Datapoints are assumed to be represented column-wise in input 
        array x."""
    if len(x.shape)<2:
        x = x[:,newaxis]
    if len(m.shape)<2:
        m = m[:,newaxis]
    if len(s.shape)<2:
        s = s[:,newaxis]
    return (x * s) + m


def distmatrix(x,y=None):
    """ Compute distance matrix for the points in (columnwise) data-matrices 
        x and y.
    """
    if y is None: y=x
    if len(x.shape)<2:
        x = x[:,newaxis]
    if len(y.shape)<2:
        y = y[:,newaxis]
    x2 = sum(x**2,0)
    y2 = sum(y**2,0)
    return x2[:,newaxis] + y2[newaxis,:] - 2*dot(x.T,y)


def readIdx(filename):
    """Read an idx-file into a numpy array."""

    import numpy, struct

    datatypes = {0x08: numpy.ubyte,
                 0x09: numpy.byte,
                 0x0B: numpy.short,
                 0x0C: numpy.int,
                 0x0D: numpy.float,
                 0x0E: numpy.double}

    f = file(filename)
    f.read(2)   #first two bytes are 0
    (datatype,) = struct.unpack('>b', f.read(1))
    (datarank,) = struct.unpack('>b', f.read(1))
    dimensions = []
    for r in range(datarank):
        dimensions.append(struct.unpack('>i', f.read(4))[0])
    dimensions = tuple(dimensions)
    return numpy.fromstring(f.read(), dtype=datatypes[datatype]).\
                                                           reshape(*dimensions) 

def logrange(first=1.0, times=10, multiplier=0.1):
    """ Prodces a range of values on log-scale. 

        Argument first is the start value, which is multiplied times times 
        with the argument multiplier.  
    """
    return [first * multiplier**i for i in range(times)]


