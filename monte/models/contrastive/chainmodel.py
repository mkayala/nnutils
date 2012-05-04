#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from numpy import matrix, newaxis, arange, exp, double, asmatrix, argmax, where
from pylab import randn, zeros
from random import choice as randomchoice
from monte.models.contrastive.contrastive import Contrastive
from scorefunc import scorefunc
from monte.util.util import logsumexp


class Chainmodel(Contrastive):
    """ Base class for chain-structured discriminative models that are 
    trained using log-loss. 
    
    Subclasses need to instantiate a constructor that initializes the 
    score-functions to be used by the model.
    """
    def __init__(self,normalizeacrosscliques=False):
        Contrastive.__init__(self,normalizeacrosscliques)

    def posdata(self,data):
        inp = matrix(data[0])
        out = data[1]
        #------------------
        singlenodeinstantiations = []
        for i in range(out.shape[0]):
            singlenodeinstantiations.append(((inp[i,:],),(out[i],)))
        #------------------
        dualnodeinstantiations = []
        for i in range(out.shape[0]-1):
            dualnodeinstantiations.append(((None,),(out[i],out[i+1])))
        #------------------
        return (singlenodeinstantiations,dualnodeinstantiations)

    def negdata(self,data):
        inp = matrix(data[0])
        #out = data[1]
        #------------------
        singlenodemarginals, dualnodemarginals = list(self.outputmarginals(inp))
        singlenodeinstantiations = []
        for t in range(len(singlenodemarginals)):
            singlenodeinstantiations.append([])
            for i in range(self.numclasses):
                singlenodeinstantiations[t].append(((inp[t,:],),(i,)))
        #-------------------
        dualnodeinstantiations = []
        for t in range(len(dualnodemarginals)):
            dualnodemarginals[t] = dualnodemarginals[t].flatten()
            dualnodeinstantiations.append([])
            for i in range(self.numclasses):
                for j in range(self.numclasses):
                    dualnodeinstantiations[t].append(((None,),(i,j)))
        #--------------------
        return (zip(singlenodemarginals,singlenodeinstantiations),\
                               zip(dualnodemarginals,dualnodeinstantiations))

    def outputmarginals(self,input): 
        """Returns a tuple with the dual-node and single-node marginals for 
        the given input-sequence."""
        T = input.shape[0]
        alpha = zeros((self.numclasses,T),dtype=float)
        beta  = zeros((self.numclasses,T),dtype=float)
        p_s = []
        p_d = []
        for t in range(T):
            p_s.append(zeros(self.numclasses,dtype=float))
            p_d.append(zeros((self.numclasses,self.numclasses),dtype=float))
        #forward-backward:
        alpha[:,0][:,newaxis] = \
                         self.singlenodescore(input[0,:],range(self.numclasses))
        for t in range(1,T):
            for y in range(self.numclasses):
                alpha[y,t] = self.singlenodescore(input[t,:],y)+\
                    logsumexp(alpha[:,t-1].flatten()+\
                        self.dualnodescore(input,(arange(self.numclasses),y)))
        beta[:,-1][:,newaxis] = \
                      self.singlenodescore(input[-1,:],range(self.numclasses))
        for t in range(T-2,-1,-1):
            for y in range(self.numclasses):
                beta[y,t] = self.singlenodescore(input[t,:],y)+\
                    logsumexp(beta[:,t+1].flatten()+\
                        self.dualnodescore(input,(y,arange(self.numclasses))))
        logZ = logsumexp(alpha[:,-1])
        #get dual-node marginals:
        for t in range(T-1):
            for y in range(self.numclasses):
                for y_ in range(self.numclasses):
                    p_d[t][y,y_] = exp(alpha[y,t]+\
                                 self.dualnodescore(input,(y,y_))+\
                                 beta[y_,t+1]\
                                 -logZ)
        #get single-node marginals by further marginalizing:
        for t in range(T-1):
            for y in range(self.numclasses):
                p_s[t][y] = sum(p_d[t][y,:])
        p_s[-1] = sum(p_d[T-2][:,:],0)
        p_d = p_d[:-1]
        return p_s,p_d

    def cost(self,data,weightcost):
        cost = 0.0
        if type(data)!=type([]):
            data = [data]
        numcases = len(data)
        for i in range(numcases): 
            inp = matrix(data[i][0])
            out = data[i][1]
            T = inp.shape[0]
            alpha = zeros((self.numclasses,T),dtype=float)
            alpha[:,0][:,newaxis] = \
                           self.singlenodescore(inp[0,:],range(self.numclasses))
            for t in range(1,T):
                for y in range(self.numclasses):
                    alpha[y,t] = self.singlenodescore(inp[t,:],y)+\
                         logsumexp(alpha[:,t-1].flatten()+\
                             self.dualnodescore(inp,(arange(self.numclasses),y)))
            cost -= logsumexp(alpha[:,-1])
            for t in range(T):
                cost += self.singlenodescore(inp[t,:],out[t])
            for t in range(T-1):
                cost += self.dualnodescore(inp,(out[t],out[t+1]))
        cost = cost.flatten()[0]
        cost /= double(numcases)
        cost -= 0.5*weightcost*sum((self.params)**2)
        return -cost

    def viterbi(self,input): 
        input = asmatrix(input)
        T = input.shape[0]
        pointers = zeros((self.numclasses,T),dtype='int')
        delta = matrix(zeros((self.numclasses,T),dtype=float))
        delta[:,0] = self.singlenodescore(input[0,:],arange(self.numclasses))
        for t in range(1,T):
            for y in range(self.numclasses):
                vals = asmatrix(delta[:,t-1].flatten() + \
                      self.dualnodescore(input,(arange(self.numclasses),y))\
                                      + self.singlenodescore(input[t,:],y)).T
                delta[y,t] = max(vals)
                pointers[y,t] = argmax(vals.flatten())
        #backtrack:
        outputs = zeros(T,dtype='int')
        optlast = where(delta[:,-1]==max(delta[:,-1]))[0]
        outputs[-1] = randomchoice(optlast)   #break ties randomly
        for t in range(T-2,-1,-1):
            outputs[t] = pointers[outputs[t+1],t+1]
        return outputs

    def hammingloss(self,inputs,outputs):
        loss = 0
        for input, output in zip(inputs,outputs):
            loss += sum(self.viterbi(input).flatten()!=output.A.flatten())
        return loss


class ChainmodelLinear(Chainmodel):

    def __init__(self,inputdims,numclasses):
        self.numclasses = numclasses
        self.inputdims = inputdims
        self.params = 0.01 * randn(self.numclasses*self.inputdims+\
                                   self.numclasses+self.numclasses**2)
        self.singlenodescore=scorefunc.LinearScore(self.inputdims,\
                self.numclasses,\
                self.params[:self.numclasses*self.inputdims+self.numclasses])
        self.dualnodescore = scorefunc.LinearTwonodebias(numclasses,\
                self.params[self.numclasses*self.inputdims+self.numclasses:])
        self.scorefuncs = [self.singlenodescore,self.dualnodescore]
        Chainmodel.__init__(self)


class ChainmodelBackprop(Chainmodel):

    def __init__(self,inputdims,numclasses,numhid):
        self.numclasses = numclasses
        self.inputdims = inputdims
        self.numhid = numhid
        self.params = 0.01 * randn(self.numclasses*self.inputdims*self.numhid\
                                   +self.numhid+self.numhid*self.numclasses\
                                   +self.numclasses\
                                   +self.numclasses**2)
        self.singlenodescore=scorefunc.OnehiddenlayerbackpropScore(\
                               self.inputdims,\
                               self.numclasses,self.numhid,\
                               self.params[:-self.numclasses**2])
        self.dualnodescore = scorefunc.LinearTwonodebias(numclasses,\
                               self.params[-self.numclasses**2:])
        self.scorefuncs = [self.singlenodescore,self.dualnodescore]
        Chainmodel.__init__(self)


class ChainmodelLinearNegperturbnaive(ChainmodelLinear):
    """Experimental linear chain crf that uses simple random perturbations 
       for the negative phase. 
    """

    def __init__(self,inputdims,numclasses,numneg=1):
        self.numneg = numneg #number of negative cases to use per iteration 
                             #per training case
        ChainmodelLinear.__init__(self,inputdims,numclasses)

    def negdata(self,input,output):
        T = input.shape[0]
        #------------------
        singlenodeinstantiations = []
        for t in range(T):
            singlenodeinstantiations.append([])
            for n in range(self.numneg):
                singlenodeinstantiations[t].append(((input[t,:],),\
                                  (int(random.randint(self.numclasses)),)))
        #------------------
        dualnodeinstantiations = []
        for t in range(T-1):
            dualnodeinstantiations.append([])
            for n in range(self.numneg/2):
               dualnodeinstantiations[t].append(((None,),
                                 (output[t],
                                  int(random.randint(self.numclasses)))))
               dualnodeinstantiations[t].append(((None,),
                                 (int(random.randint(self.numclasses)),
                                  output[t+1])))
        #------------------
        return (zip(\
            [(1./self.numneg)*ones((self.numneg,1),dtype=float).flatten()],\
                          singlenodeinstantiations),\
                zip(\
            [(1./self.numneg)*self.numneg*ones((self.numneg,1),dtype=float).\
                                                                 flatten()],\
                dualnodeinstantiations))


