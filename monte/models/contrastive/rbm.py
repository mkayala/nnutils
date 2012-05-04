#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from numpy import zeros, asmatrix, reshape, matrix, exp, double, random, sum,\
                  newaxis, concatenate, asarray
from pylab import randn, find
from monte.models.contrastive.contrastive import Contrastive
from monte.models.contrastive.scorefunc import scorefunc

class Rbm(Contrastive):
    """Base class for standard restricted Boltzmann machine. 
    
       Subclasses need to define the methods hidprobs, visprobs, sample_hid 
       and sample_obs. """

    def __init__(self,numvis,numhid,sparsitygain=0.0,targethidprobs=0.2,\
                   cditerations=1,normalizeacrosscliques=True,\
                   meanfield_output=True,verbose=False):
        self.targethidprobs = targethidprobs
        self.numvis = numvis
        self.numhid = numhid
        self.cditerations = cditerations
        self.sparsitygain = sparsitygain
        self.meanfield_output = meanfield_output
        self.params = 0.01*randn(numvis*numhid+numvis+numhid)
        self.wyh=asmatrix(reshape(self.params[:numvis*numhid],(numvis,numhid)))
        self.wy = asmatrix(self.params[numvis*numhid:numvis*numhid+numvis]).T
        self.wh = asmatrix(self.params[numvis*numhid+numvis:]).T
        self.verbose = verbose
        Contrastive.__init__(self,normalizeacrosscliques)

    def __str__(self):
        return "rbm with " + str(self.numvis) + " observable and " \
             + str(self.numhid) + " hidden units"

    def grad(self,outputs,weightcost):
        """This overrides the base-class method grad for efficiency reasons."""
        grad = zeros(0)
        if len(outputs.shape) < 2: #got rank-1 arrays?
            outputs = outputs[:,newaxis]
        numcases = outputs.shape[1]
        outputs = asmatrix(outputs)
        poshidprobs = self.hidprobs(outputs)
        if self.cditerations == 1:
            hidstates   = self.sample_hid(poshidprobs)
            negdata     = self.visprobs(hidstates)
            if not self.meanfield_output:
                negdata = self.sample_obs(negdata)
            neghidprobs = self.hidprobs(negdata)
        else:
            neghidprobs = poshidprobs
            for c in range(self.cditerations):
                hidstates   = self.sample_hid(neghidprobs)
                negdata     = self.visprobs(hidstates)
                datastates  = self.sample_obs(negdata)
                neghidprobs = self.hidprobs(datastates)
        posprods  = outputs*poshidprobs.T
        negprods  = negdata*neghidprobs.T
        posvisact = sum(outputs,1).reshape(self.numvis,1)
        negvisact = sum(negdata,1).reshape(self.numvis,1)
        poshidact = sum(poshidprobs,1).reshape(self.numhid,1)
        neghidact = sum(neghidprobs,1).reshape(self.numhid,1)
        if self.sparsitygain > 0.0:
            spyh, sph = self.sparsityKLpenalty(poshidprobs,outputs)
        else:
            spyh = asmatrix(zeros((self.numvis,self.numhid)))
            sph  = asmatrix(zeros((self.numhid,1)))
        gradyh = ((posprods-negprods)/numcases)-weightcost*self.wyh-\
                            self.sparsitygain*spyh
        gradh = ((poshidact-neghidact)/numcases)-weightcost*self.wh-\
                            self.sparsitygain*sph
        grady = ((posvisact-negvisact)/numcases)-weightcost*self.wy
        grad  = concatenate((reshape(gradyh.A,self.numvis*self.numhid),\
                             reshape(grady.A,self.numvis),\
                             reshape(gradh.A,self.numhid)))
        if self.verbose:
            print "av. squared err: %f" % \
                      (sum(sum(asarray(outputs-negdata)**2))/double(numcases))
            print "av. hid activities: %f" % \
                       (sum(poshidprobs)/double(numcases*self.numhid))
        return -grad

    def hidprobs(self, outputs):
        """Compute probabilities of hiddens, given inputs and outputs. """
        pass

    def visprobs(self, hiddens):
        """Computes probabilities of outputs, given inputs and hiddens. """
        pass

    def sample_hid(self,probs):
        """Sample independently from hiddens."""
        pass 

    def sample_obs(self,probs):
        """Sample independently from observables."""
        pass 

    def sparsityKLpenalty(self,poshidprobs,inputs):
        spyh = (sum(poshidprobs,1).T.A/double(poshidprobs.shape[1])\
                           -self.targethidprobs)[newaxis,:] * \
                           sum(inputs.A,1)[:,newaxis]\
                            /double(poshidprobs.shape[1])
        sph = (sum(poshidprobs.A,1)/double(poshidprobs.shape[1])\
                        -self.targethidprobs)[:,newaxis] \
                         /double(poshidprobs.shape[1])
        return spyh, sph

    def sparsitysquarepenalty(self,poshidprobs,inputs):
        spyh = (sum(poshidprobs,1).T.A/double(poshidprobs.shape[1])\
                           -self.targethidprobs)[newaxis,:] * \
                     (inputs*asmatrix(poshidprobs.A*(1-poshidprobs.A)).T).A\
                            /double(poshidprobs.shape[1])
        sph = ((sum(poshidprobs.A,1)/double(poshidprobs.shape[1])\
                        -self.targethidprobs)[:,newaxis] * \
                     sum(poshidprobs.A*(1-poshidprobs.A),1)[:,newaxis]\
                         /double(poshidprobs.shape[1]))
        return spyh, sph

    def recons(self, outputs):
        if len(outputs.shape) < 2: #got rank-1 arrays?
            outputs = outputs[:,newaxis]
        outputs = asmatrix(outputs)
        return asarray(self.visprobs(self.hidprobs(outputs)))


class RbmBinBin(Rbm):

    def __str__(self):
        return "rbm_binbin with " + str(self.numvis) + " observable and " \
             + str(self.numhid) + " hidden units"

    def hidprobs(self, inputs):
        return 1./(1.+exp(-self.wyh.T*inputs - self.wh))

    def visprobs(self, hiddens):
        return 1./(1.+exp(-self.wyh*hiddens - self.wy))

    def sample_hid(self, hidprobs):
        return (hidprobs>random.rand(hidprobs.shape[0],hidprobs.shape[1])).\
                                                                 astype(float)

    def sample_obs(self, visprobs):
        return (visprobs>random.rand(visprobs.shape[0],visprobs.shape[1])).\
                                                                 astype(float)


class RbmBinGauss(Rbm):

    def __init__(self,numvis,numhid,sparsitygain=0.0,targethidprobs=0.2,\
                 cditerations=1,nu=1.0,verbose=False):
      Rbm.__init__(self,numvis,numhid,sparsitygain,targethidprobs,cditerations,\
                 verbose)
      self.meanfield_output = False
      self.nu = nu

    def __str__(self):
        return "RbmBinGauss with " + str(self.numvis) + " observable and " \
             + str(self.numhid) + " hidden units and nu = " + str(self.nu)

    def hidprobs(self, inputs):
        return 1./(1.+exp(-self.wyh.T*inputs - self.wh))

    def visprobs(self, hiddens):
        return (self.wyh/self.nu*hiddens + self.wy)*self.nu**2

    def sample_hid(self, hidprobs):
        return (hidprobs>random.rand(hidprobs.shape[0],hidprobs.shape[1])).\
                                                                  astype(float)

    def sample_obs(self, visprobs):
        return visprobs+random.randn(visprobs.shape[0],visprobs.shape[1])\
                                                                      *self.nu

