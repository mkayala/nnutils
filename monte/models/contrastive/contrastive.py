#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from numpy import array, isscalar, concatenate, double
from pylab import randn, find, zeros

class Contrastive(object):
    """ Abstract super-class for all contrastive models. """

    def __init__(self,normalizeacrosscliques=True):
        self.normalizeacrosscliques=normalizeacrosscliques

    def posdata(self,data):
        """Provides the positive clique instantiations to train the model. 
        
        Returns input-instantiations (or input-clique-instantiations, if that 
        is what the template-model requires), output-clique-instantiations, 
        and hidden instantiations for the hiddens that this clique is 
        responsible for."""
        pass

    def negdata(self,data):
        """Provides the negative clique instantiations to train the model, 
           along with weights for the gradient updates. 
           
           The weights should be marginal probabilities for max-likelihood-
           learning, or can be all equal to one, for sampling-based learning."""
        pass

    def grad(self,data,weightcost):
        grad = zeros(0)
        if type(data)!=type([]):
            data = [data]
        numcases = len(data)
        numscoretypes = len(self.scorefuncs)
        if not type(weightcost) == type([]):
            weightcost = [weightcost] * numscoretypes
        posgrad = [None]*numscoretypes
        neggrad = [None]*numscoretypes
        for k in range(numscoretypes):
            if isscalar(weightcost[k]):
                weightcost[k] = \
                      array([weightcost[k]]*len(self.scorefuncs[k].params))
            posgrad[k] = zeros(self.scorefuncs[k].params.shape,dtype=float)
            neggrad[k] = zeros(self.scorefuncs[k].params.shape,dtype=float)
        for i in range(numcases):
            poscliques = self.posdata(data[i])
            negcliques = self.negdata(data[i])
            for k in range(numscoretypes):
                for posclique in poscliques[k]:
                    posgrad[k] += self.scorefuncs[k].grad(*posclique)
                if self.normalizeacrosscliques:
                    posgrad[k] = posgrad[k]/double(len(poscliques[k]))
                for weighting, negclique in negcliques[k]:
                    for w, neginst in zip(weighting,negclique):
                        neggrad[k] += w * self.scorefuncs[k].grad(*neginst)
                if self.normalizeacrosscliques:
                    neggrad[k] = neggrad[k]/double(len(poscliques[k]))
        for k in range(numscoretypes):
            grad = concatenate((grad,(posgrad[k]-neggrad[k])/double(numcases)\
                                     -weightcost[k]*self.scorefuncs[k].params))
        return -grad

    def f(self,x,data,weightcost):
        """Wrapper function around cost function to check grads, etc."""
        xold = self.params.copy()
        self.updateparams(x.copy().flatten())
        result = self.cost(data,weightcost) 
        self.updateparams(xold.copy())
        return result

    def g(self,x,data,weightcost):
        """Wrapper function around gradient to check grads, etc."""
        xold = self.params.copy()
        self.updateparams(x.copy().flatten())
        result = self.grad(data,weightcost)
        self.updateparams(xold.copy())
        return result

    def updateparams(self,newparams):
        self.params *= 0.0
        self.params += newparams.copy()


