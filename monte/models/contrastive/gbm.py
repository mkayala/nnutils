#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from numpy import zeros, ones, newaxis, array, asarray, asmatrix,\
                  random, double, mod
from pylab import randn, find
from random import choice as randomchoice
from monte.util.util import *
from monte.models.contrastive.contrastive import Contrastive
from monte.models.contrastive.scorefunc import scorefunc
from monte.bp import neuralnet


class Gbm(Contrastive):
    """Global gated Boltzmann machine model (base class)."""

    def __init__(self,numin,numout,nummap,\
                 sparsitygain=0.0,targethidprobs=0.2,cditerations=1,\
                 normalizeacrosscliques=True,meanfield_output=True,\
                 premap=None,postmap=None,\
                 xy = False, xh = False, yh = True,\
                 verbose=True):
        self.optlevel = 1
        self.xy = xy
        self.xh = xh
        self.yh = yh
        self.numin   = numin
        self.numout  = numout
        self.nummap  = nummap
        self.premap  = premap
        self.postmap = postmap
        if premap is not None:
          self.numin  = self.premap[1].shape[0]
          self.numout = self.postmap[0].shape[1]
        self.sparsitygain = sparsitygain
        self.targethidprobs = targethidprobs
        self.cditerations = cditerations
        self.meanfield_output = meanfield_output
        self.params  = 0.01*random.randn(self.numin*self.numout*self.nummap+\
                                     self.yh * self.numout*self.nummap+\
                                     self.xy * self.numin*self.numout+\
                                     self.xh * self.numin*self.nummap+\
                                     self.numout+\
                                     self.nummap)
        self.scorefunc = scorefunc.GbmScore(self.numin,self.numout,self.nummap,\
                                       sparsitygain,targethidprobs,self.params,\
                                       self.xy,self.xh,self.yh)
        self.scorefuncs = [self.scorefunc] 
        self.verbose = verbose
        Contrastive.__init__(self,normalizeacrosscliques)

    def __str__(self):
        return "Gbm instance with " + str(self.numin) + " input-, " \
               + str(self.numout) + " output- and " + str(self.nummap) \
               + " hidden units"

    def posdata(self,data):
        if len(data[0].shape) < 2: #got rank-1 arrays?
            inp = data[0].reshape((data[0].shape[0],1))
            out = data[1].reshape((data[1].shape[0],1))
        else:
            inp = data[0]
            out = data[1]
        numcases = inp.shape[1]
        if self.premap is not None:
            inp=(self.premap[1]*asmatrix(inp+self.premap[0])).A
            out=(self.premap[1]*asmatrix(out+self.premap[0])).A
        self.modweights = self.scorefunc.modulatedweights(inp)
        self.hids = self.hidprobs(out, modweights=self.modweights)
        return (((inp,self.hids,out),),)

    def negdata(self,data):
        """Generate neg data"""
        if len(data[0].shape) < 2: #got rank-1 arrays?
            inp = data[0].reshape((data[0].shape[0],1))
            out = data[1].reshape((data[1].shape[0],1))
        else:
            inp = data[0]
            out = data[1]
        numcases = inp.shape[1]
        out_original = out
        if self.premap is not None:
            inp = (self.premap[1]*asmatrix(inp+self.premap[0])).A
            out = (self.premap[1]*asmatrix(out+self.premap[0])).A
        if self.cditerations==1:
            hidstates   = self.sample_hid(self.hids)
            negoutput   = self.outprobs(hidstates, modweights=self.modweights)
            if not self.meanfield_output:
                negoutput = self.sample_obs(negoutput)
            neghidprobs = self.hidprobs(negoutput, modweights=self.modweights)
        else:
            for c in range(self.cditerations):
                hidstates = self.sample_hid(self.hids)
                negoutput = self.outprobs(hidstates, modweights=self.modweights)
                datastates= self.sample_obs(negoutput)
                self.hids = self.hidprobs(datastates,modweights=self.modweights)
            neghidprobs = self.hidprobs(datastates, modweights=self.modweights)
        if self.verbose:
            if self.postmap is not None:
                print "av. squared err in the output-space: %f" % \
                     (sum(sum((\
                 out_original-(dot(self.postmap[0],negoutput)+self.postmap[1])\
                     )**2))/double(numcases))
            else:
                print "av. squared err: %f" % \
                      (sum(sum(asarray(out-negoutput)**2))/double(numcases))
        return ((([1.0], [(inp,neghidprobs,negoutput)]),),)

    def rdist(self,x1,x2,batchsize=300):
        """ 'Reconstruction distance.' Breaks the task up into batches, 
        to avoid memory problems"""
        if len(x1.shape)<2: x1 = x1[:,newaxis]
        if len(x2.shape)<2: x2 = x2[:,newaxis]
        dists = zeros((x1.shape[1],x2.shape[1]),dtype=float)
        if self.premap is not None:
            x1=(self.premap[1]*asmatrix(x1+self.premap[0])).A
        for i in range(x1.shape[1]/batchsize+1): 
            if self.verbose:
                print 'rdist batch' + str(i)
                print 'computing modweights'
            modweights = self.scorefunc.\
                             modulatedweights(x1[:,i*batchsize:(i+1)*batchsize])
            if self.verbose:
                print 'done'
            for j in range(x2.shape[1]):
                dists[i*batchsize:(i+1)*batchsize,j]=\
                        ((self.rawoutprobs(self.rawhidprobs(x2[:,j],\
                                            modweights=modweights),\
                                            modweights=modweights)-\
                                                x2[:,j][:,newaxis])**2).sum(0)
        return dists

    def drawsamples(self,inputs,outputs_init,numgibbsiterations):
        if len(outputs_init.shape) < 2: 
            outputs_init = outputs_init[:,newaxis]
        if len(inputs.shape) < 2: 
            inputs = inputs[:,newaxis]
        numsamples = outputs_init.shape[1]
        result = zeros((numsamples,self.numout))
        outputs = outputs_init
        modweights = self.scorefunc.modulatedweights(inputs)
        for i in range(numgibbsiterations):
            hiddens = \
                  self.sample_hid(self.hidprobs(outputs,modweights=modweights))
            outprobs = self.outprobs(hiddens,modweights=modweights)
            if not self.meanfield_output:
                outputs = self.sample_obs(outprobs)
        return outputs


class GbmBinBin(Gbm):

    def __str__(self):
        return "GbmBinBin instance with " + str(self.numin) + " input-, " \
             + str(self.numout) + " output- and " + str(self.nummap) \
             + " hidden units"

    def hidprobs(self, outputs, inputs=array([]), modweights=()):
        if len(outputs.shape)<2: #got rank-1 array?
            outputs=outputs[:,newaxis]
        numcases = outputs.shape[1]
        if modweights == (): 
            if len(inputs.shape)<2: #got rank-1 array?
                inputs=inputs[:,newaxis]
            wxyh_, wxy_, wxh_ = self.scorefunc.modulatedweights(inputs)
        else: wxyh_, wxy_, wxh_ = modweights
        if self.optlevel > 0:
            from scipy import weave
            result = zeros((self.nummap,numcases),dtype=float)
            code = r"""
              #include <math.h>
              for(int c=0;c<numcases;c++){
                for(int k=0;k<nummap;k++){
                  for(int j=0;j<numout;j++){
                   if(yh) result(k,c) -= outputs(j,c) * (wxyh_(c,j,k)+wyh(j,k));
                   else result(k,c) -= outputs(j,c) * wxyh_(c,j,k);
                  }
                  if(xh) result(k,c) -= wxh(c,k) + wh(k);
                  else result(k,c) -= wh(k);
                  result(k,c) = 1.0/(1.0+exp(result(k,c)));
                }
              }
            """
            vars = ['outputs','numcases','numin','nummap','numout',\
                                   'wxyh_','wh','wxh','wyh','result','xh','yh']
            global_dict = {'wxyh_':wxyh_,'wh':self.scorefunc.wh.A,\
                           'xh':int(self.xh),'yh':int(self.yh),\
                           'wxh':array(float(self.xh),ndmin=1) and wxh_.A,\
                           'wyh':\
                       array(float(self.yh),ndmin=1) and self.scorefunc.wyh.A,\
                           'outputs':outputs,\
                           'numout':self.numout,'numin':self.numin,\
                           'nummap':self.nummap,\
                           'numcases':numcases,\
                           'result':result}
            weave.inline(code,vars,global_dict=global_dict,\
                         type_converters=weave.converters.blitz)
            return result
        else:
            return 1.0/(1.0+exp(-(\
                 sum(outputs.T[:,:,newaxis]*
                          (wxyh_+(float(self.yh) and self.scorefunc.wyh.A)),1)\
                 + (float(self.xh) and wxh_.A)\
                 + self.scorefunc.wh.T.A\
                 ))).T

    def outprobs(self, hiddens, inputs=array([]), modweights=()):
        if len(hiddens.shape)<2: #got rank-1 array?
            hiddens=hiddens[:,newaxis]
        numcases = hiddens.shape[1]
        if modweights == (): 
            if len(inputs.shape)<2: #got rank-1 array?
                inputs=inputs[:,newaxis]
            wxyh_, wxy_, wxh_ = self.scorefunc.modulatedweights(inputs)
        else: wxyh_, wxy_, wxh_ = modweights
        return 1.0/(1.0+exp(-(\
               sum(hiddens.T[:,newaxis,:]*
                          (wxyh_+(float(self.yh) and self.scorefunc.wyh.A)),2)\
               + (float(self.xy) and wxy_.A)\
               + self.scorefunc.wy.T.A\
               ))).T

    def sample_hid(self,hidprobs):
        return (hidprobs > random.rand(hidprobs.shape[0],\
                                             hidprobs.shape[1])).astype(float)

    def sample_obs(self,outprobs):
        return (outprobs > random.rand(outprobs.shape[0],\
                                             outprobs.shape[1])).astype(float)

    def freeenergy(self, outputs, input=array([]), modweights=()):
        """Compute marginal free energies for all outputs stacked in array 
           outputs, for the single given input or single modulated weights."""
        if len(outputs.shape)<2: #got rank-1 array?
            outputs=outputs[:,newaxis]
        if len(input.shape)<2: #got rank-1 array?
            input=input[:,newaxis]
        if modweights == (): 
            wxyh_, wxy_, wxh_ = self.scorefunc.modulatedweights(input)
        else: 
            wxyh_, wxy_, wxh_ = modweights
        if wxyh_.shape[0] > 1: \
            raise "Only single input/modweights allowed currently."
        return -log(1.0+exp((outputs.T[:,:,newaxis]*\
                              (wxyh_[0,:,:]+\
                               wxy_.A[0,:][:,newaxis]+\
                               wxh_.A[0,:][newaxis,:]+\
                               self.scorefunc.wyh.A+\
                               self.scorefunc.wy.A+\
                               self.scorefunc.wh.A.T)[newaxis,:,:]
                             ).sum(1))).sum(1)


class GbmBinGauss(Gbm):
    def __init__(self,numin,numout,nummap,\
                 sparsitygain=0.0,targethidprobs=0.1,nu=1.0,cditerations=1,\
                 premap=None,postmap=None,\
                 xy = False, xh = False, yh = True,\
                 verbose=0):
        Gbm.__init__(self,numin,numout,nummap,\
                   sparsitygain,targethidprobs,cditerations=cditerations,\
                   premap=premap,postmap=postmap,\
                   xy=xy,xh=xh,yh=yh,verbose=verbose)
        self.meanfield_output = False
        self.nu = nu

    def __str__(self):
        return "GbmBinGauss instance with " + str(self.numin) + " input-, "\
             + str(self.numout) + " output- and " + str(self.nummap) \
             + " hidden units and nu = " + str(self.nu)

    def hidprobs(self, outputs, inputs=array([]), modweights=()):
        if len(outputs.shape)<2: #got rank-1 array?
            outputs=outputs[:,newaxis]
        numcases = outputs.shape[1]
        if modweights == (): 
            if len(inputs.shape)<2: #got rank-1 array?
                inputs=inputs[:,newaxis]
            wxyh_, wxy_, wxh_ = self.scorefunc.modulatedweights(inputs)
        else: 
            wxyh_, wxy_, wxh_ = modweights
        return 1.0/(1.0+exp(-(\
               sum(outputs.T[:,:,newaxis]*\
                  (wxyh_+(float(self.yh) and self.scorefunc.wyh.A)),1)/self.nu\
               + (float(self.xh) and wxh_.A)\
               + self.scorefunc.wh.T.A\
               ))).T

    def outprobs(self, hiddens, inputs=array([]), modweights=()):
        if len(hiddens.shape)<2: #got rank-1 array?
            hiddens=hiddens[:,newaxis]
        numcases = hiddens.shape[1]
        if modweights == (): 
            if len(inputs.shape)<2: #got rank-1 array?
                inputs=inputs[:,newaxis]
            wxyh_, wxy_, wxh_ = self.scorefunc.modulatedweights(inputs)
        else: 
            wxyh_, wxy_, wxh_ = modweights
        return (( 
               sum(hiddens.T[:,newaxis,:]*
                         (wxyh_+(float(self.yh) and self.scorefunc.wyh.A)),2)\
               + (float(self.xy) and wxy_.A)\
               + self.scorefunc.wy.T.A\
               ) / self.nu).T * self.nu**2

    def rawoutprobs(self, hiddens, inputs=array([]), modweights=()):
        if len(hiddens.shape)<2: #got rank-1 array?
            hiddens=hiddens[:,newaxis]
        if modweights == (): 
            if len(inputs.shape)<2: #got rank-1 array?
                inputs=inputs[:,newaxis]
            if self.premap is not None:
                inputs=dot(self.premap[1],inputs+self.premap[0])
            wxyh_, wxy_, wxh_ = self.scorefunc.modulatedweights(inputs)
        else: 
            wxyh_, wxy_, wxh_ = modweights
        if self.postmap is not None:
            return self.postmap[1]+\
                 (dot(self.postmap[0],(\
               #    (sum(hiddens.T[:,newaxis,:]*(wxyh_+self.scorefunc.wyh.A),2)\
               #     + wxy_\
                   (sum(hiddens.T[:,newaxis,:]*
                           (wxyh_+(float(self.yh) and self.scorefunc.wyh.A)),2)\
                 + (float(self.xy) and wxy_.A)\
                      + self.scorefunc.wy.T.A\
                      ) / self.nu).T * self.nu**2\
                    ))
        else:
            return ((sum(hiddens.T[:,newaxis,:]*(wxyh_+self.scorefunc.wyh.A),2)\
                + (float(self.xy) and wxy_.A)\
                + self.scorefunc.wy.T.A\
                ) / self.nu).T * self.nu**2

    def rawhidprobs(self, outputs, inputs=array([]), modweights=()):
        """Compute hiddens from raw outputs, ie. outputs after postmap."""
        if len(outputs.shape)<2: #got rank-1 array?
            outputs = outputs[:,newaxis]
        if modweights == ():
            if len(inputs.shape)<2: #got rank-1 array?
                inputs = inputs[:,newaxis]
            if self.premap is not None:
                inputs = dot(self.premap[1],inputs+self.premap[0])
            wxyh_, wxy_, wxh_ = self.scorefunc.modulatedweights(inputs)
        else: 
            wxyh_, wxy_, wxh_ = modweights
        if self.postmap is not None:
            outputs = dot(self.premap[1],outputs+self.premap[0])
        return 1.0/(1.0+exp(-(\
               sum(outputs.T[:,:,newaxis]*\
               (wxyh_+(float(self.yh) and self.scorefunc.wyh.A)),1)/self.nu\
               + (float(self.xh) and wxh_.A)\
               + self.scorefunc.wh.T.A\
               ))).T

    def freeenergy(self, outputs, input=array([]), modweights=()):
        """Compute marginal free energies for all outputs stacked in array 
           outputs, for the single given input or single modulated weights."""
        if len(outputs.shape)<2: #got rank-1 array?
            outputs=outputs[:,newaxis]
        if len(input.shape)<2: #got rank-1 array?
            input=input[:,newaxis]
        if self.premap is not None:
            input = dot(self.premap[1],input+self.premap[0]).A
        if self.postmap is not None:
            outputs = dot(self.premap[1],outputs+self.premap[0]).A
        if modweights == (): 
            wxyh_, wxy_, wxh_ = self.scorefunc.modulatedweights(input)
        else: 
            wxyh_, wxy_, wxh_ = modweights
        if wxyh_.shape[0] > 1: \
            raise "Only single input/modweights allowed currently."
        return -log(1.0+exp((outputs.T[:,:,newaxis]*\
                              (wxyh_[0,:,:]+\
                               wxy_.A[0,:][:,newaxis]+\
                               wxh_.A[0,:][newaxis,:]+\
                               self.scorefunc.wyh.A+\
                               self.scorefunc.wy.A+\
                               self.scorefunc.wh.A.T)[newaxis,:,:]
                             ).sum(1))).sum(1)
    def sample_hid(self,hidprobs):
        return (hidprobs > random.rand(*hidprobs.shape)).astype(float)
    def sample_obs(self,outprobs):
        return outprobs+random.randn(*outprobs.shape)*self.nu


class BpGbm(Gbm):
    """Global gated Boltzmann machine in which the inputs go through a 
    backprop net first (base class)."""

    def __init__(self,numrawin,numin,nummap,numout,prefunc=None,\
                   sparsitygain=0.0,targethidprobs=0.2,cditerations=1,\
                   normalizeacrosscliques=True,meanfield_output=True,\
                   premap=None,postmap=None,\
                   verbose=True):
        self.optlevel = 1
        self.xy = False
        self.xh = False
        self.yh = False
        self.numin = numin
        self.numrawin = numrawin
        self.numout   = numout
        self.nummap   = nummap
        self.meanfield_output = meanfield_output
        self.sparsitygain = sparsitygain
        self.targethidprobs = targethidprobs
        self.cditerations = cditerations
        self.normalizeacrosscliques = normalizeacrosscliques
        self.premap  = premap
        self.postmap = postmap
        if premap is not None:
            self.numrawin  = self.premap[1].shape[0]
            self.numout = self.postmap[0].shape[1]
        if prefunc is None:
            prefunc = neuralnet.Linearlayer(numrawin,numin,
               zeros(neuralnet.Linearlayer.numparams(numrawin,numin)))
        self.params  = 0.01*random.randn(self.numin*self.numout*self.nummap+\
                                     self.yh * self.numout*self.nummap+\
                                     self.xy * self.numin*self.numout+\
                                     self.xh * self.numin*self.nummap+\
                                     self.numout+\
                                     self.nummap+\
                                     size(prefunc.params))
        self.scorefunc = scorefunc.BpGbmScore(numrawin,numin,nummap,numout,
                                     prefunc.__class__,\
                                     sparsitygain,targethidprobs,self.params,
                                     self.xy,self.xh,self.yh)
        self.scorefuncs = [self.scorefunc] 
        self.verbose = verbose

    def __str__(self):
        return "BpGbm instance with " + str(self.numin) + " input-, " \
             + str(self.numout) + " output- and " + str(self.nummap) \
             + " hidden units"

    def posdata(self,data):
        if len(data[0].shape) < 2: #got rank-1 arrays?
            inp = data[0].reshape((data[0].shape[0],1))
            out = data[1].reshape((data[1].shape[0],1))
        else:
            inp = data[0]
            out = data[1]
        if self.premap is not None:
            inp=(self.premap[1]*asmatrix(inp+self.premap[0])).A
            out=(self.premap[1]*asmatrix(out+self.premap[0])).A
        self.modweights = self.scorefunc.modulatedweights(inp)
        self.hids = self.hidprobs(out,modweights=self.modweights)
        return (((inp,self.hids,out),),)

    def negdata(self,data):
        """Generate neg data"""
        if len(data[0].shape) < 2: #got rank-1 arrays?
            inp = data[0].reshape((data[0].shape[0],1))
            out = data[1].reshape((data[1].shape[0],1))
        else:
            inp = data[0]
            out = data[1]
        out_original = out
        if self.premap is not None:
            inp = (self.premap[1]*asmatrix(inp+self.premap[0])).A
            out = (self.premap[1]*asmatrix(out+self.premap[0])).A
        if self.cditerations==1:
            hidstates   = self.sample_hid(self.hids)
            negoutput   = self.outprobs(hidstates, modweights=self.modweights)
            if not self.meanfield_output:
                negoutput = self.sample_obs(negoutput)
            neghidprobs = self.hidprobs(negoutput, modweights=self.modweights)
        else:
            for c in range(self.cditerations):
                hidstates = self.sample_hid(self.hids)
                negoutput= self.outprobs(hidstates, modweights=self.modweights)
                datastates= self.sample_obs(negoutput)
                self.hids= self.hidprobs(datastates,modweights=self.modweights)
            neghidprobs = self.hidprobs(datastates, modweights=self.modweights)
        if self.verbose:
            numcases = inp.shape[1]
            if self.postmap is not None:
                print "av. squared err in the output-space: %f" % \
                   (sum(sum((out_original\
                             -(dot(self.postmap[0],negoutput)+self.postmap[1])\
                   )**2))/double(numcases))
            else:
                print "av. squared err: %f" % \
                    (sum(sum(asarray(out-negoutput)**2))/double(numcases))
        return ((([1], [(inp,neghidprobs,negoutput)]),),)


class BpGbmBinBin(BpGbm,GbmBinBin):
    pass


class BpGbmBinGauss(BpGbm,GbmBinGauss):
    def __init__(self,numrawin,numin,nummap,numout,prefunc=None,\
                 sparsitygain=0.0,targethidprobs=0.2,nu=1.0,cditerations=1,\
                 normalizeacrosscliques=True,meanfield_output=True,\
                 premap=None,postmap=None,\
                 verbose=True):
      self.nu = nu
      BpGbm.__init__(self,numrawin,numin,nummap,numout,prefunc,\
                 sparsitygain=sparsitygain,targethidprobs=targethidprobs,\
                 cditerations=cditerations,\
                 normalizeacrosscliques=normalizeacrosscliques,\
                 meanfield_output=meanfield_output,\
                 premap=premap,postmap=postmap,\
                 verbose=verbose)


class GbmChan(Contrastive):
    """Gated Boltzmann machine model that uses multiple input channels 
       (base class).
    """

    def __init__(self,numin,numout,nummap,\
                   sparsitygain=0.0,targethidprobs=0.2,cditerations=1,\
                   normalizeacrosscliques=True,meanfield_output=True,\
                   xy = False, xh = False, yh = True,\
                   verbose=True):
        self.optlevel = 0
        self.xy = xy
        self.xh = xh
        self.yh = yh
        self.numin   = numin
        self.numout  = numout
        self.nummap  = nummap
        self.sparsitygain = sparsitygain
        self.targethidprobs = targethidprobs
        self.cditerations = cditerations
        self.meanfield_output = meanfield_output
        self.params  = 0.01*random.randn(self.numin*self.numout*self.nummap+\
                                     self.yh * self.numout*self.nummap+\
                                     self.xy * self.numin*self.numout+\
                                     self.xh * self.numin*self.nummap+\
                                     self.numout+\
                                     self.nummap)
        self.scorefunc = \
                    scorefunc.gbmchanscore(self.numin,self.numout,self.nummap,\
                    sparsitygain,targethidprobs,self.params,\
                    self.xy,self.xh,self.yh)
        self.scorefuncs = [self.scorefunc] 
        self.verbose = verbose
        Contrastive.__init__(self,normalizeacrosscliques)

    def __str__(self):
        return "GbmChan instance with " + str(self.numin) + " input-, " \
             + str(self.numout) + " output- and " + str(self.nummap) \
             + " hidden units"

    def posdata(self,data):
        if len(data[0].shape) < 2: #got rank-1 arrays?
            inp = data[0].reshape((data[0].shape[0],1))
            out = data[1].reshape((data[1].shape[0],1))
        else:
            inp = data[0]
            out = data[1]
        numcases = inp.shape[1]
        self.modweights = self.scorefunc.modulatedweights(inp)
        self.hids = self.hidprobs(out, modweights=self.modweights)
        return (((inp,self.hids,out),),)

    def negdata(self,data):
        """Generate neg data"""
        if len(data[0].shape) < 2: #got rank-1 arrays?
            inp = data[0].reshape((data[0].shape[0],1))
            out = data[1].reshape((data[1].shape[0],1))
        else:
            inp = data[0]
            out = data[1]
        numcases = inp.shape[1]
        out_original = out
        if self.cditerations == 1:
            hidstates   = self.sample_hid(self.hids)
            negoutput   = self.outprobs(hidstates, modweights=self.modweights)
            if not self.meanfield_output:
                negoutput = self.sample_obs(negoutput)
            neghidprobs = self.hidprobs(negoutput, modweights=self.modweights)
        else:
            for c in range(self.cditerations):
                hidstates= self.sample_hid(self.hids)
                negoutput= self.outprobs(hidstates, modweights=self.modweights)
                datastates= self.sample_obs(negoutput)
                self.hids= self.hidprobs(datastates,modweights=self.modweights)
            neghidprobs = self.hidprobs(datastates, modweights=self.modweights)
        if self.verbose:
            print "av. squared err: %f" % \
                  (sum(sum(asarray(out-negoutput)**2))/double(numcases))
        return ((([1.0], [(inp,neghidprobs,negoutput)]),),)

    def rdist(self,x1,x2,batchsize=300):
        """ 'Reconstruction distance.' Breaks the task up into batches, 
        to avoid memory problems"""
        if len(x1.shape)<2: x1 = x1[:,newaxis]
        if len(x2.shape)<2: x2 = x2[:,newaxis]
        dists = zeros((x1.shape[1],x2.shape[1]),dtype=float)
        for i in range(x1.shape[1]/batchsize+1): 
            if self.verbose:
                print 'rdist batch' + str(i)
                print 'computing modweights'
            modweights = self.scorefunc.\
                           modulatedweights(x1[:,i*batchsize:(i+1)*batchsize])
            if self.verbose:
                print 'done'
            for j in range(x2.shape[1]):
                dists[i*batchsize:(i+1)*batchsize,j]=\
                        ((self.rawoutprobs(self.rawhidprobs(x2[:,j],\
                                            modweights=modweights),\
                                            modweights=modweights)-\
                                                x2[:,j][:,newaxis])**2).sum(0)
        return dists

    def drawsamples(self,inputs,outputs_init,numgibbsiterations):
        if len(outputs_init.shape) < 2: 
            outputs_init = outputs_init[:,newaxis]
        if len(inputs.shape) < 2: 
            inputs = inputs[:,newaxis]
        numsamples = outputs_init.shape[1]
        result = zeros((numsamples,self.numout))
        outputs = outputs_init
        modweights = self.scorefunc.modulatedweights(inputs)
        for i in range(numgibbsiterations):
            hiddens = self.sample_hid(self.hidprobs(outputs,\
                                                        modweights=modweights))
            outprobs = self.outprobs(hiddens,modweights=modweights)
            if not self.meanfield_output:
                outputs = self.sample_obs(outprobs)
        return outputs


class GbmChanBinBin(GbmChan,GbmBinBin):
    pass


class RbmGbmGbm(Contrastive):
    """ Combination of an rbm with two gbm for learning transformation-
        invariant features. Experimental.
    """

    def __init__(self,numdims,numhid,nummap,numcases,nu,gbm_model,\
                sparsitygain=0.0,targethidprobs=0.2,cditerations=1,\
                normalizeacrosscliques=True,verbose=False):
        #Note that numcases has to be specified beforehand in this model
        if not gbm_model:
            self.gbm = GbmBinGauss(numdims,numdims,nummap,nu,verbose)
        else: 
            self.gbm = gbm_model
        self.numdims = numdims
        self.numhid  = numhid
        self.nummap  = nummap
        self.numcases = numcases
        self.sparsitygain = sparsitygain
        self.targethidprobs = targethidprobs
        self.nu      = nu
        self.cditerations = cditerations
        self.params  = 0.01 * randn(numdims*numhid+numdims+\
                                    numhid*numhid*nummap+numhid*nummap+\
                                    numhid*numhid+numhid)
        self.vyh = \
               asmatrix(reshape(self.params[:numdims*numhid],(numdims,numhid)))
        self.vy = \
               asmatrix(self.params[numdims*numhid:numdims*numhid+numdims]).T
        self.uxyh = reshape(self.params[numdims*numhid+numdims:\
                                numdims*numhid+numdims+numhid*numhid*nummap],\
                                (numhid,numhid,nummap))
        self.uyh = asmatrix(reshape(self.params[\
                              numdims*numhid+numdims+numhid*numhid*nummap:\
                              numdims*numhid+numdims+numhid*numhid*nummap\
                              + numhid*nummap],
                              (numhid,nummap)))
        self.uxy  = asmatrix(reshape(self.params[\
                              numdims*numhid+numdims+numhid*numhid*nummap\
                              + numhid*nummap:\
                              numdims*numhid+numdims+numhid*numhid*nummap\
                              + numhid*nummap+numhid*numhid],\
                              (numhid,numhid)))
        self.uy = asmatrix(reshape(self.params[-numhid:],(numhid,1)))
        self.verbose = verbose
        self.featuresinput = random.randn(numhid,self.numcases)
        Contrastive.__init__(self,normalizeacrosscliques)

    def __str__(self):
        return "RbmGbmGbm instance with " + str(self.numdims) \
             + " input- and output-units, " + str(self.numhid) \
             + " hidden units, " + str(self.nummap) \
             + " mapping units and nu = " + str(self.nu)

    def grad(self,data,weightcosts):
        """This overrides the base-class method grad for efficiency reasons."""
        if type(data)!=type([]):
            data = [data]
        inputs = [d[0] for d in data]
        outputs = [d[1] for d in data]
        if len(inputs[0].shape) < 2: #got rank-1 arrays? stack them together
            inputs  = array(inputs).T
            outputs = array(outputs).T
        grad = zeros(0)
        mappings = self.gbm.hidprobs(outputs=outputs,inputs=inputs)
        hidbiases = self.modulatedupperweights(mappings)
        poshidprobs = self.rbm_hidprobs(outputs,hidbiases)
        hidstates = \
            (poshidprobs>random.rand(self.numhid,self.numcases)).astype(float)
        posprods = asmatrix(outputs)*poshidprobs.T
        posvisact = sum(outputs,1).reshape(self.numdims,1)
        poshidact = sum(poshidprobs,1).reshape(self.numhid,1)
        posprods_uxyh = \
            sum(reshape(self.featuresinput.T,(self.numcases,self.numhid,1,1))\
                                         *ones((1,1,self.numhid,self.nummap))\
                       *reshape(poshidprobs.T,(self.numcases,1,self.numhid,1))\
                                         *ones((1,self.numhid,1,self.nummap))\
                          *reshape(mappings.T,(self.numcases,1,1,self.nummap))\
                                        *ones((1,self.numhid,self.numhid,1)),0)
        posprods_uyh=sum(reshape(poshidprobs.T,(self.numcases,self.numhid,1))\
                                                *ones((1,1,self.nummap))\
                           *reshape(mappings.T,(self.numcases,1,self.nummap))\
                                                *ones((1,self.numhid,1)),0)
        posprods_uxy = \
              sum(reshape(self.featuresinput.T,(self.numcases,self.numhid,1))\
                                                *ones((1,1,self.numhid))\
                         *reshape(poshidprobs.T,(self.numcases,1,self.numhid))\
                                                *ones((1,self.numhid,1)),0)
        posprods_uy = sum(poshidprobs,1)    #just normal biases
        for c in range(self.cditerations):
            negdata = self.rbm_visprobs(hidstates)
            datastates = negdata + \
                       random.randn(negdata.shape[0],negdata.shape[1])*self.nu
            neghidprobs = self.rbm_hidprobs(negdata,hidbiases)
            hidstates = (neghidprobs > random.rand(self.numhid,\
                                                 self.numcases)).astype(float)
        neghidprobs = self.rbm_hidprobs(negdata,hidbiases)
        negprods    = asmatrix(negdata)*neghidprobs.T
        negvisact   = sum(negdata,1).reshape(self.numdims,1)
        neghidact = sum(neghidprobs,1).reshape(self.numhid,1)
        negprods_uxyh = sum(reshape(self.featuresinput.T,\
                                         (self.numcases,self.numhid,1,1))\
                                         *ones((1,1,self.numhid,self.nummap))\
                      *reshape(neghidprobs.T,(self.numcases,1,self.numhid,1))\
                                         *ones((1,self.numhid,1,self.nummap))\
                         *reshape(mappings.T,(self.numcases,1,1,self.nummap))\
                                       *ones((1,self.numhid,self.numhid,1)),0)
        negprods_uyh=sum(reshape(neghidprobs.T,(self.numcases,self.numhid,1))\
                                                *ones((1,1,self.nummap))\
                         *reshape(mappings.T,(self.numcases,1,self.nummap))\
                                                *ones((1,self.numhid,1)),0)
        negprods_uxy = sum(reshape(self.featuresinput.T,\
                                                (self.numcases,self.numhid,1))\
                                                *ones((1,1,self.numhid))\
                         *reshape(neghidprobs.T,(self.numcases,1,self.numhid))\
                                                *ones((1,self.numhid,1)),0)
        negprods_uy = sum(neghidprobs,1)    #just normal biases
  
        if self.sparsitygain > 0.0:
            spuxyh, spuyh, spuxy, spuy, spvyh = \
                           self.sparsityKLpenalty(poshidprobs,mappings,outputs)
        else:
            spvyh  = zeros((self.numdims,self.numhid),dtype=float)
            spuxyh = zeros((self.numhid,self.numhid,self.nummap),dtype=float)
            spuyh  = zeros((self.numhid,self.nummap),dtype=float)
            spuxy  = zeros((self.numhid,self.numhid),dtype=float)
            spuy   = zeros((self.numhid,1),dtype=float)
        gradvyh  = ((posprods-negprods)/self.numcases)-weightcosts*self.vyh\
                               -self.sparsitygain*spvyh
        gradvy   = ((posvisact-negvisact)/self.numcases)-weightcosts*self.vy
        graduxyh = ((posprods_uxyh-negprods_uxyh)/self.numcases)\
                               -weightcosts*self.uxyh\
                               -self.sparsitygain*spuxyh
        graduyh  = ((posprods_uyh-negprods_uyh)/self.numcases)\
                               -weightcosts*self.uyh\
                               -self.sparsitygain*spuyh
        graduxy  = ((posprods_uxy-negprods_uxy)/self.numcases)\
                               -weightcosts*self.uxy\
                               -self.sparsitygain*spuxy
        graduy   = ((poshidact-neghidact)/self.numcases)\
                               -weightcosts*self.uy\
                               -self.sparsitygain*spuy
        if self.verbose:
            numcases = inp.shape[1]
            print "av. squared err: %f" % \
                      (sum(sum(asarray(out-negoutput)**2))/double(numcases))
        grad = concatenate((reshape(gradvyh.A,self.numdims*self.numhid),\
                            reshape(gradvy.A,self.numdims),\
                        reshape(graduxyh,self.numhid*self.numhid*self.nummap),\
                            reshape(graduyh.A,self.numhid*self.nummap),\
                            reshape(graduxy.A,self.numhid*self.numhid),\
                            reshape(graduy.A,self.numhid)\
                          ))
        return -grad

    def sparsityKLpenalty(self,hidprobs,mappings,outputs):
        spuy = (sum(hidprobs,1)/double(hidprobs.shape[1])\
                           -self.targethidprobs)[:,newaxis]\
                            /double(hidprobs.shape[1])
        spuxyh = (sum(hidprobs,1).T/double(hidprobs.shape[1])\
                           -self.targethidprobs)[newaxis,:,newaxis,newaxis]*\
                            sum(self.featuresinput.T[:,:,newaxis,newaxis]*\
                            hidprobs.T[:,newaxis,:,newaxis]*\
                            mappings.T[:,newaxis,newaxis,:],0)\
                            /double(hidprobs.shape[1])
        spuxy = (sum(hidprobs,1).T/double(hidprobs.shape[1])\
                           -self.targethidprobs)[newaxis,:,newaxis]*\
                            sum(self.featuresinput.T[:,:,newaxis]*\
                            hidprobs.T[:,newaxis,:],0)\
                            /double(hidprobs.shape[1])
        spuyh = (sum(hidprobs,1).T/double(hidprobs.shape[1])\
                           -self.targethidprobs)[newaxis,:,newaxis]*\
                            sum(hidprobs.T[:,:,newaxis]*\
                            mappings.T[:,newaxis,:],0)\
                            /double(hidprobs.shape[1])
        spvyh = (sum(hidprobs,1)/double(hidprobs.shape[1])\
                           -self.targethidprobs)[newaxis,:]*\
                            sum(outputs,1)[:,newaxis]\
                            /double(hidprobs.shape[1])
        #return spuy, spvyh
        return spuxyh, spuyh, spuxy, spuy, spvyh

    def rbm_hidprobs(self, inputs, hidbiases):
        uxyh_ = hidbiases[0]
        uxy_ = hidbiases[1]
        return (1/(1+exp(-\
             self.vyh.T/self.nu*inputs-uxyh_/self.nu-uxy_/self.nu-self.uy))).A

    def rbm_visprobs(self, hiddens):
        return ((self.vyh/self.nu*hiddens + self.vy/self.nu)*self.nu**2).A

    def modulatedupperweights(self, mappings):
        """ Compute the modulated weights (which in this model happen to be the 
          biases for the next time-step) """
        numcases = self.featuresinput.shape[1]
        uxyh_ = matrix(sum(sum(\
                     reshape(self.featuresinput.T,(numcases,self.numhid,1,1))\
                                         *ones((1,1,self.numhid,self.nummap))\
           *reshape(self.uxyh,(1,self.numhid,self.numhid,self.nummap)),1),2)).T
        uxy_ = matrix(self.featuresinput.T*self.uxy).T
#        uxh_ = self.featuresinput.T*self.uxh 
        return uxyh_, uxy_#, uxh_

  
class Foge(Gbm):
    """Gated Boltzmann machine as a field of experts model: A field of 
    gated experts (base class)."""

    def __init__(self,height,width,nummap,\
                  sparsitygain=0.0,targethidprobs=0.2,cditerations=1,\
                  normalizeacrosscliques=True,meanfield_output=True,\
                  premap=None, postmap=None, usecliquelists=True,\
                  xy=False,xh=False,yh=True,\
                  verbose=True):
        if mod(height,2) != 1 or mod(width,2) != 1:
            raise "Height and width parameters need to be odd numbers."
        self.optlevel = True       
        self.xy = xy
        self.xh = xh
        self.yh = yh
        self.meanfield_output = meanfield_output
        self.usecliquelists = usecliquelists
        self.height  = height
        self.width   = width
        self.h       = int(floor(self.height/2.))
        self.w       = int(floor(self.width/2.))
        self.premap  = premap
        self.postmap = postmap
        if premap is not None:
            self.numin  = self.premap[1].shape[0]
            self.numout = self.postmap[0].shape[1]
        else:
            self.numin   = self.width*self.height
            self.numout  = self.width*self.height
        self.nummap  = nummap
        self.cditerations = cditerations
        self.params  = 0.0001*random.randn(self.numin*self.numout*self.nummap+\
                                     self.xy*self.numin*self.numout+\
                                     self.xh*self.numin*self.nummap+\
                                     self.yh*self.numout*self.nummap+\
                                     self.numout+\
                                     self.nummap)
        self.scorefunc=scorefunc.GbmScore(self.numin,self.numout,self.nummap,\
                               sparsitygain,targethidprobs,self.params,\
                               xy=self.xy,xh=self.xh,yh=self.yh)
        self.scorefuncs = [self.scorefunc] 
        self.verbose = verbose
        Contrastive.__init__(self,normalizeacrosscliques)

    def image2patches(self,image):
        """ Turn images into a list, or stacked matrix, of patches."""
        numcases, imheight, imwidth = image.shape
        if self.usecliquelists:
            patches = []
            if self.premap is not None:
                for i in range(self.h,imheight-self.h):
                    for j in range(self.w,imwidth-self.w):
                        patches.append(\
                          dot(self.premap[1],\
                          image[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1].\
                  reshape((numcases,self.height*self.width)).T+self.premap[0]\
                                   )
                          )
            else:
                for i in range(self.h,imheight-self.h):
                    for j in range(self.w,imwidth-self.w):
                        patches.append(\
                             image[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1].\
                                 reshape((numcases,self.height*self.width)).T)
        else:
            numpatchespercase = (imheight-2*self.h) * (imwidth-2*self.w)
            patches = zeros(\
               (self.height*self.width,numpatchespercase*numcases),dtype=float)
            cliq = 0
            if self.premap is not None:
                for i in range(self.h,imheight-self.h):
                    for j in range(self.w,imwidth-self.w):
                        patches[:,cliq*numcases:(cliq+1)*numcases] = \
                          dot(self.premap[1],\
                            image[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1].\
                              reshape((numcases,\
                                     self.height*self.width)).T+self.premap[0]\
                         )
                        cliq += 1
            else:
                for i in range(self.h,imheight-self.h):
                    for j in range(self.w,imwidth-self.w):
                        patches[:,cliq*numcases:(cliq+1)*numcases] = \
                        image[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1].\
                                reshape((numcases,self.height*self.width)).T
                        cliq += 1
            patches = [patches]
        return patches

    def posdata(self,data):
        imin = data[0]
        imout = data[1]
        if len(imin.shape) == 2:    #got only one image?
            imin = imin[newaxis,:,:]
        if len(imout.shape) == 2:
            imout = imout[newaxis,:,:]
        numcases, imheight, imwidth = imin.shape
        self.modweights = []
        self.modweights = self.globalmodweights(imin)
        self.hids = self.globalhidprobs(imout,modweights=self.modweights)
        inputcliques = self.image2patches(imin)
        outputcliques = self.image2patches(imout)
        return (zip(inputcliques,self.hids,outputcliques),)

    def negdata(self,data):
        imin  = data[0]
        imout = data[1]
        if len(imin.shape) == 2:    #got only one image?
            imin = imin[newaxis,:,:]
        if len(imout.shape) == 2:
            imout = imout[newaxis,:,:]
        numcases, imheight, imwidth = imin.shape
        if self.cditerations == 1:
            hidstates = self.sample_globalhid(self.hids)
            negoutput = self.globaloutprobs(imheight,imwidth,hidstates,\
                                                   modweights=self.modweights)
            if not self.meanfield_output:
                negoutput = self.sample_globalobs(negoutput)
        else:
            for c in range(self.cditerations):
                hidstates = self.sample_globalhid(self.hids)
                negoutput = self.globaloutprobs(imheight,imwidth,hidstates,\
                                                    modweights=self.modweights)
                negoutput = self.sample_globalobs(negoutput)
                self.hids = \
                     self.globalhidprobs(negoutput,modweights=self.modweights)
        neghidprobs = self.globalhidprobs(negoutput,modweights=self.modweights)
        #reconstruct clique-wise representation for outputs::
        inputcliques = self.image2patches(imin)
        negoutputcliques = self.image2patches(negoutput)
        if self.verbose:
            err = ((imout - negoutput)**2).sum()/double(numcases)
            print "av. squared err in the output-space: %f" % err
        return (zip(([1.0],)*len(self.hids), \
           map(lambda x: [x],zip(inputcliques,neghidprobs,negoutputcliques))),)
    

    def globaloutprobs(self,imheight,imwidth,hidcliques,imin=array([]),\
                                                               modweights=()):
        if modweights == (): 
            if len(imin.shape) == 2:    #got only one image?
                imin = imin[newaxis,:,:]
            numcases, imheight, imwidth = imin.shape
            modweights = self.globalmodweights(imin)
        else: 
            modweights = modweights
            numcases = modweights[0][0].shape[0]
        imout = zeros((numcases,imheight,imwidth),dtype=float)
        cliq = 0
        if self.usecliquelists:
            if self.postmap is not None:
                for i in range(self.h,imout.shape[1]-self.h):
                    for j in range(self.w,imout.shape[2]-self.w):
                        imout[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1] += \
                          (dot(self.postmap[0],
                               (self.outmeans(hidcliques[cliq],\
                                modweights=modweights[cliq])))\
                                   +self.postmap[1]).transpose(0,1)\
                           .reshape((numcases,self.height,self.width))
                        cliq += 1
            else:
                for i in range(self.h,imout.shape[1]-self.h):
                    for j in range(self.w,imout.shape[2]-self.w):
                        imout[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1] += \
                                           self.outmeans(hidcliques[cliq],\
                                           modweights=modweights[cliq]).\
                      			   transpose(0,1).\
                                    reshape((numcases,self.height,self.width))
                        cliq += 1
        else:
            if self.postmap is not None:
                for i in range(self.h,imout.shape[1]-self.h):
                  for j in range(self.w,imout.shape[2]-self.w):
                      imout[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1] += \
                        (dot(self.postmap[0],
                        (self.outmeans(\
                            hidcliques[0][:,cliq*numcases:(cliq+1)*numcases],\
                              modweights=modweights[cliq])))\
                                 +self.postmap[1]).transpose(0,1)\
                         .reshape((numcases,self.height,self.width))
                      cliq += 1
            else:
                for i in range(self.h,imout.shape[1]-self.h):
                    for j in range(self.w,imout.shape[2]-self.w):
                        imout[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1] += \
                          self.outmeans(\
                             hidcliques[0][:,cliq*numcases:(cliq+1)*numcases],\
                                            modweights=modweights[cliq]).\
                      			    transpose(0,1).\
                                     reshape((numcases,self.height,self.width))
                        cliq += 1
        imout = self.outprobs(imout)
        return imout

    def globalhidprobs(self,imout,imin=array([]),modweights=()):
        if len(imout.shape) == 2:    #got only one image?
            imout = imout[newaxis,:,:]
        if len(imin.shape) == 2:    #got only one image?
            imin = imin[newaxis,:,:]
        numcases, imheight, imwidth = imout.shape
        if modweights == (): 
            modweights = self.globalmodweights(imin)
        else: 
            modweights = modweights
        if self.usecliquelists:
            hidcliques = []
            cliq = 0
            if self.premap is not None:
                for i in range(self.h,imout.shape[1]-self.h):
                    for j in range(self.w,imout.shape[2]-self.w):
                        hidcliques.append(self.hidprobs(\
                          dot(\
                          self.premap[1],(\
                             imout[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1].\
                             reshape(\
                          (numcases,self.height*self.width)).T+self.premap[0])\
                             ),modweights=modweights[cliq])\
                        )
                        cliq = cliq + 1
            else:
                for i in range(self.h,imout.shape[1]-self.h):
                    for j in range(self.w,imout.shape[2]-self.w):
                        hidcliques.append(self.hidprobs(\
                           imout[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1].\
                         reshape((numcases,self.height*self.width)).T,\
                                                  modweights=modweights[cliq]))
                        cliq = cliq + 1
        else:
            numpatchespercase = (imheight-2*self.h) * (imwidth-2*self.w)
            hidcliques = zeros(\
                          (self.nummap,numpatchespercase*numcases),dtype=float)
            cliq = 0
            if self.premap is not None:
                for i in range(self.h,imout.shape[1]-self.h):
                    for j in range(self.w,imout.shape[2]-self.w):
                        hidcliques[:,cliq*numcases:(cliq+1)*numcases] = \
                                        self.hidprobs(\
                        dot(self.premap[1],\
                           (imout[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1].\
                             reshape((numcases,self.height*self.width)).T\
                                                            +self.premap[0])),\
                                               modweights=modweights[cliq])
                        cliq = cliq + 1
            else:
                for i in range(self.h,imout.shape[1]-self.h):
                    for j in range(self.w,imout.shape[2]-self.w):
                        hidcliques[:,cliq*numcases:(cliq+1)*numcases] \
                             = self.hidprobs(\
                         imout[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1].\
                            reshape((numcases,self.height*self.width)).T,\
                                                   modweights=modweights[cliq])
                        cliq = cliq + 1
            hidcliques = [hidcliques]
        return hidcliques

    def globalmodweights(self,imin):
        if len(imin.shape) == 2:    #got only one image?
            imin = imin[newaxis,:,:]
        numcases, imheight, imwidth = imin.shape
        modweights = []
        if self.premap is not None:
            for i in range(self.h,imin.shape[1]-self.h):
                for j in range(self.w,imin.shape[2]-self.w):
                    modweights.append(self.scorefunc.modulatedweights(\
                    dot(self.premap[1],\
                       (imin[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1].\
                reshape((numcases,self.height*self.width)).T+self.premap[0]))))
        else:
            for i in range(self.h,imin.shape[1]-self.h):
                for j in range(self.w,imin.shape[2]-self.w):
                    modweights.append(self.scorefunc.modulatedweights(\
                             imin[:,i-self.h:i+self.h+1,j-self.w:j+self.w+1].\
                                reshape((numcases,self.height*self.width)).T))
        return modweights

    def globalsample(self,imin,imout_init,numiter):
        """ Do numiter Gibbs iterations. If self.meanfield_output is True, 
        observables are never begin sampled, but represented by their means 
        instead."""
        if len(imin.shape) == 2:    #got only one image?
            imin = imin[newaxis,:,:]
        if len(imout_init.shape) == 2:    #got only one image?
            imout_init = imout_init[newaxis,:,:]
        numcases, imheight, imwidth = imin.shape
        imout = imout_init
        modweights = self.globalmodweights(imin)
        for i in range(numiter):
            hiddens = self.globalhidprobs(imout,modweights=modweights)
            hiddens = self.sample_globalhid(hiddens)
            imout   = self.globaloutprobs(imout.shape[1],imout.shape[2],\
                                          hiddens,modweights=modweights)
            if not self.meanfield_output:
                imout = self.sample_globalobs(imout)
        imout   = self.globaloutprobs(imout.shape[1],imout.shape[2],\
                                        hiddens,modweights=modweights)
        return imout

    def sample_globalhid(self,hidprobs):
        h = []
        for hidclique in hidprobs:
            h.append(self.sample_hid(hidclique))
        return h

    def sample_globalobs(self,outprobs):
        return self.sample_obs(outprobs)

    def showmarginalflow(self, imin, imout):
        imheight, imwidth = imin.shape
        weights = zeros((imheight,imwidth,imheight,imwidth),dtype=float)
        print " weights should be sparse matrix "
        wxy_ = [None] * imheight
        for i in range(imheight):
            wxy_[i] = [None] * imwidth
        hidcliques = self.globalhidprobs(imin,imout)
        c = 0
        if self.usecliquelists:
            for i in range(self.h,imheight-self.h):
                for j in range(self.w,imwidth-self.w):
                    wxy_[i][j] = (\
                          sum(self.scorefunc.wxyh*hidcliques[c].flatten(),2)+\
                                 self.scorefunc.wxy).A
                    c += 1
        else:
            for i in range(self.h,imheight-self.h):
                for j in range(self.w,imwidth-self.w):
                    wxy_[i][j] = \
                    (sum(self.scorefunc.wxyh*hidcliques[0][c*numcases:\
                                (c+1)*numcases].flatten(),2)+\
                                self.scorefunc.wxy).A
                    c += 1
        #for now: only 'inside', do not care about edges
        for i in range(self.h,imheight-self.h):
            for j in range(self.w,imwidth-self.w):
                for ci in range(-self.h,self.h+1):
                  for cj in range(-self.w,self.w+1):
                      weights[i-self.h:i+self.h+1,j-self.h:j+self.h+1,\
                                     i-self.h:i+self.h+1,j-self.w:j+self.w+1]\
                          += transpose(wxy_[i][j].\
                     reshape((self.height,self.width,self.height,self.width)),\
                                                                     (0,1,2,3))
        #find maxima:
        x = []
        y = []
        xTo = []
        yTo = []
        for i in range(self.h,imheight-self.h):
            for j in range(self.w,imwidth-self.w):
                t1,t2 = where(weights[i,j,:,:] == weights[i,j,:,:].max())
                xTo.append(t2-j)
                yTo.append(i-t1)
                x.append(i)
                y.append(j)
        xTo = array(xTo)
        yTo = array(yTo)
        x = array(x)
        y = array(y[::-1])
        return x,y,xTo,yTo,weights,wxy_


class FogeBinBin(Foge,GbmBinBin):

    def outprobs(self, outmeans):
        return 1./(1.+exp(-outmeans))

    def outmeans(self, hiddens, inputs=array([]), modweights=()):
        numcases = hiddens.shape[1]
        if modweights == (): 
            wxyh_, wxy_, wxh_ = self.scorefunc.modulatedweights(inputs)
        else: 
            wxyh_, wxy_, wxh_ = modweights
        return (sum(hiddens.T[:,newaxis,:]*\
                          (wxyh_+(float(self.yh) and self.scorefunc.wyh.A)),2)\
                + (float(self.xy) and wxy_.A)\
                + self.scorefunc.wy.T.A)

class FogeBinGauss(Foge,GbmBinGauss):

    def __init__(self,height,width,nummap,\
           sparsitygain=0.0,targethidprobs=0.2,nu=1.0,cditerations=1,\
           premap=None, postmap=None,\
           xy=False,xh=False,yh=True,\
           verbose=True):
        self.nu = nu
        self.meanfield_output = False
        Foge.__init__(self,height,width,nummap,\
                  sparsitygain=sparsitygain,targethidprobs=targethidprobs,\
             cditerations=cditerations,meanfield_output=self.meanfield_output,\
                  premap=premap,postmap=postmap,\
                  xy=xy,xh=xh,yh=yh,\
                  verbose=verbose)

    def outprobs(self, outmeans):
        return outmeans * self.nu**2

    def outmeans(self, hiddens, inputs=array([]), modweights=()):
        numcases = hiddens.shape[1]
        if modweights == (): 
            wxyh_, wxy_, wxh_ = self.scorefunc.modulatedweights(inputs)
        else: 
            wxyh_, wxy_, wxh_ = modweights
        return (( sum(hiddens.T[:,newaxis,:]*
                         (wxyh_+(float(self.yh) and self.scorefunc.wyh.A)),2)\
               + (float(self.xy) and wxy_.A)\
               + self.scorefunc.wy.T.A\
               ) / self.nu).T 

