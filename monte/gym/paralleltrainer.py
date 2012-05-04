#Copyright (C) 2007-2008 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

import trainer
import threading 
import Queue

from numpy import mod, inf, double, isnan, ceil
from pylab import zeros

class OnlinegradientNocostBatchThreads(trainer.Trainer):
    """ Online gradient descent trainer that distributes training-data-batches
        across multiple threads and averages the gradients. 

        Expects that the first argument to the model's cost and gradient 
        functions is a data-sequence (such as a list of data-cases or a list
        of batches). 
    """

    class batchThread(threading.Thread):

        def __init__(self, model, dataPool, gradients, args):
            self.dataPool = dataPool
            self.model = model
            self.gradients = gradients
            self.args = args
            threading.Thread.__init__ ( self )

        def run(self):
            while True:
                data_index = self.dataPool.get() 
                if data_index == None:
                    break
                self.gradients[:, data_index[1]] = \
                                     self.model.grad(data_index[0], *self.args)

    def __init__(self, model, momentum, stepsize, numthreads,
                                                callback=None,
                                                callbackargs=None,
                                                callbackiteration=1):
        self.model = model
        self.momentum = momentum
        self.stepsize = stepsize
        self.numthreads = numthreads
        self.inc = zeros(self.model.params.shape,dtype=float)
        trainer.Trainer.__init__(self,callback,callbackargs,callbackiteration)

    def step(self, data, *args):
        dataPool = Queue.Queue()
        gradients = \
                zeros( (len(self.model.params), self.numthreads), dtype=float)
        chunksize = int(ceil(float(len(data)) / self.numthreads))
        for j in range(self.numthreads):
            dataPool.put( (data[j*chunksize:(j+1)*chunksize], j) )
        for j in range(self.numthreads):
            dataPool.put(None)
        threads = [OnlinegradientNocostBatchThreads.batchThread(
                                        self.model, dataPool, gradients, args) 
                                              for i in range(self.numthreads)] 
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.inc[:] = self.momentum*self.inc - \
                    self.stepsize * gradients.sum(1) / double(self.numthreads)
        if isnan(sum(self.inc)): 
            print 'nan!'
            self.inc = zeros(self.inc.shape, dtype=float)
        self.model.params += self.inc
        trainer.Trainer.step(self, data, *args)

