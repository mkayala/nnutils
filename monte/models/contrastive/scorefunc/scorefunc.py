from numpy import double, sum, zeros, ones, asmatrix, concatenate, reshape, \
                  array, newaxis
#from pylab import *
from monte.bp import neuralnet


class LinearScore(object):
    """ Module that computes an (affine) linear score w_y.T * x + b_y for 
        given input x and output y."""

    def __init__(self,inputdims,numclasses,params):
        self.inputdims = inputdims
        self.numclasses = numclasses
        self.params = params #pointer to params that were passed in 
        self.w = asmatrix(self.params[:self.numclasses*self.inputdims]\
                                     .reshape((self.numclasses,self.inputdims)))
        self.b = asmatrix(self.params[self.numclasses*self.inputdims:])\
                                     .reshape((1,self.numclasses)).T

    def __call__(self,input,output):
        return ((self.w[output]*input.T) + self.b[output]).A

    def grad(self,input,output):
        grad_w = zeros((self.numclasses,self.inputdims),dtype=float)
        grad_w[output,:] = input[0].A
        grad_b = zeros((self.numclasses,1),dtype=float)
        grad_b[output] = 1.0
        return concatenate((reshape(grad_w,self.inputdims*self.numclasses),\
               reshape(grad_b,self.numclasses)))


class LinearTwonodebias(object):
    """ Stupid classifier who's output is the score on the joint configuration 
        of two output-nodes, the input is ignored. Useful as the 
        compatibility-potential in a linear chain crf."""

    def __init__(self,numclasses,params):
        self.numclasses = numclasses
        self.params = params

    def __call__(self,input,output):
        return self.params[output[0]*self.numclasses+output[1]]

    def grad(self,input,output):
        result = zeros((self.numclasses**2),dtype=float)
        result[output[0]*self.numclasses+output[1]]=1.0
        return result


class OnehiddenlayerbackpropScore(object):
    """ Module that computes a nonlinear score for given input x and output y
        using a one-hidden-layer-backprop-net."""

    def __init__(self,inputdims,numclasses,numhid,params):
        self.inputdims = inputdims
        self.numhid = numhid
        self.numclasses = numclasses
        self.hiddens = zeros((self.numhid,1),dtype=float) 
        self.params = params 
        self.w_inputtohidden = \
              asmatrix(self.params[:self.numclasses*self.inputdims*self.numhid]\
                .reshape((self.numhid,self.numclasses*self.inputdims)))
        self.w_hiddentooutput = \
              asmatrix(self.params[self.numclasses*self.inputdims*self.numhid:\
                                   self.numclasses*self.inputdims*self.numhid+\
                                   self.numhid])\
                                   .reshape((1,self.numhid))
        self.b_inputtohidden = \
              asmatrix(self.params[self.numclasses*self.inputdims*self.numhid\
                                   +self.numhid:\
                                   self.numclasses*self.inputdims*self.numhid+\
                                   self.numhid+\
                                   self.numhid*self.numclasses])\
                                   .reshape((self.numhid,self.numclasses))
        self.b_hiddentooutput = \
                 self.params[-self.numclasses:]\
                                      .reshape(self.numclasses)

    def __call__(self,input,output):
        if not iterable(output): output=[output]
        scores = zeros((len(output),1),dtype=float)
        index = 0
        for y in output:
            self.hiddens *= 0.0
            self.hiddens += \
              self.w_inputtohidden\
                  [:,y*self.inputdims:y*self.inputdims+self.inputdims]*input.T\
                        +self.b_inputtohidden[:,y]
            self.hiddens = 1./(1.+exp(-self.hiddens))
            scores[index,0] = \
                 (self.w_hiddentooutput*asmatrix(self.hiddens)+\
                               self.b_hiddentooutput[y]).A[0,:]
            index = index + 1
        return scores

    def grad(self,input,output,CALLFORWARDMAP=True):
        input = input[0]
        output = output[0]
        if CALLFORWARDMAP:
            self(input,output)  #fill in the hiddens
        grad_w_hiddentooutput = self.hiddens
        grad_b_hiddentooutput = zeros(self.b_hiddentooutput.shape,dtype=float)
        grad_b_hiddentooutput[output] = 1.0
        grad_w_inputtohidden = zeros(self.w_inputtohidden.shape,dtype=float)
        grad_w_inputtohidden[:,output*self.inputdims:output*self.inputdims+\
                              self.inputdims] += self.w_hiddentooutput.T.A * \
                                         ((self.hiddens*(1.0-self.hiddens))*\
                                                 input).A
        grad_b_inputtohidden = zeros(self.b_inputtohidden.shape,dtype=float)
        grad_b_inputtohidden[:,output][:,newaxis]=self.w_hiddentooutput.T.A * \
                                               (self.hiddens*(1.0-self.hiddens))
        return concatenate((reshape(grad_w_inputtohidden,\
                                  self.numhid*self.numclasses*self.inputdims),\
                            reshape(grad_w_hiddentooutput,\
                                     self.numhid),\
                            reshape(grad_b_inputtohidden,\
                                     self.numhid*self.numclasses),\
                            reshape(grad_b_hiddentooutput,\
                                     self.numclasses)))


class SigmoidhiddenLinearoutputScore(object):
    """ Backprob network with one sigmoid hidden and one linear output layer.
        To do batch-learning, rank-2 arrays can be passed in as input (where 
        cases are arranged along the second rank).
    """

    def __init__(self,numin,numhid,numout,params):
        self.numin  = numin
        self.numhid = numhid
        self.numout = numout
        self.params = params
        self.bpnet = neuralnet.SigmoidLinear(numin,numhid,numout,self.params)

    def __call__(self,input):
        return self.bpnet.fprop(input)

    def grad(self,input,output):
        modeloutput = self.bpnet.fprop(input)
        d_output = 2*(modeloutput-output)
        self.bpnet.bprop(d_output,input)
        grad = self.bpnet.grad(d_output,input)
        if len(input.shape) >= 2:
            grad = sum(grad,1)/double(input.shape[1])
        return -grad  #view it as score, not cost


class LinearRegressionScore(object):
    """ Simple linear neural network to do regression. """

    @staticmethod
    def numparams(numin,numout):
        return bp.neuralnet.Linearlayer.numparams(numin,numout)

    def __init__(self,numin,numout,params):
        self.numin  = numin
        self.numout = numout
        self.params = params
        self.bpnet = neuralnet.Linearlayer(numin,numout,self.params)

    def __call__(self,input):
        return self.bpnet.fprop(input)

    def grad(self,input,output):
        modeloutput = self.bpnet.fprop(input)
        d_output = 2*(modeloutput-output)
        self.bpnet.bprop(d_output,input)
        grad = self.bpnet.grad(d_output,input)
        if len(input.shape) >= 2:
            grad = sum(grad,1)/double(input.shape[1])
        return -grad  #view it as score, not cost


class Islsl(object):
    """ Network with the structure
        input -> sigmoid -> linear -> sigmoid -> linear 
        Useful e.g. as autoencoder. """

    @staticmethod
    def numparams(numin,numhid1,numhi2,numhid3,numout):
        return neuralnet.Islsl.numparams(numin,numhid1,numhi2,numhid3,numout)

    def __init__(self,numin,numhid1,numhid2,numhid3,numout,params):
        self.numin  = numin
        self.numhid1 = numhid1
        self.numhid2 = numhid2
        self.numhid3 = numhid3
        self.numout = numout
        self.params = params
        self.bpnet = neuralnet.Islsl(\
                             numin,numhid1,numhid2,numhid3,numout,self.params)

    def __call__(self,input):
        return self.bpnet.fprop(input)

    def grad(self,input,output):
        modeloutput = self.bpnet.fprop(input)
        d_output = 2*(modeloutput-output)
        self.bpnet.bprop(d_output,input)
        grad = self.bpnet.grad(d_output,input)
        if len(input.shape) >= 2:
            grad = sum(grad,1)/double(input.shape[1])
        return -grad  #view it as score, not cost


class SigmoidhiddenSoftmaxoutputScore(object):
    """ Backprob network with a sigmoid hidden and a softmax output layer.
        To do batch-learning, rank-2 arrays can be passed in as input (where 
        cases are arranged along the second rank).
    """

    def __init__(self,numin,numhid,numclasses,params):
        self.numin  = numin
        self.numhid = numhid
        self.numclasses = numclasses
        self.params = params
        self.bpnet = neuralnet.SigmoidLinear(\
                                          numin,numhid,numclasses,self.params)

    def __call__(self,input,output):
        if not iterable(output): output=[output]
        modeloutput = exp(self.bpnet.fprop(input))
        modeloutput = modeloutput/sum(modeloutput,0)
        index = 0
        for y in output:
            scores[index,0] = modeloutput(y)
            index = index + 1
        return scores

    def grad(self,input,output):
        modeloutput = exp(self.bpnet.fprop(input))
        modeloutput = modeloutput/sum(modeloutput,0)
        d_output = zeros((self.numclasses),dtype=float)
        d_output[y] = 1
        self.bpnet.bprop(d_output,input)
        grad = self.bpnet.grad(d_output,input)
        if len(input.shape) >= 2:
            grad = sum(grad,1)/double(input.shape[1])
        return -grad  #view it as score, not cost


class GbmScore(object):
    def __init__(self,numin,numout,nummap,sparsitygain,targethidprobs,params,
                   xy=True,xh=True,yh=True):
        self.optlevel = 1 #turn on inline-C speed-up 
        self.xy = xy
        self.xh = xh
        self.yh = yh
        self.numin  = numin
        self.numout = numout
        self.nummap = nummap
        self.sparsitygain = sparsitygain
        self.targethidprobs = targethidprobs
        self.params = params
        self.wxyh = \
               self.params[:numin*numout*nummap].reshape((numin,numout,nummap))
        self.prods_wxyh = zeros(self.wxyh.shape,dtype=float)
        if self.yh:
            self.wyh = asmatrix(self.params[numin*numout*nummap:\
                                numin*numout*nummap+numout*nummap].\
                                reshape((numout,nummap)))
            self.prods_wyh = zeros(self.wyh.shape,dtype=float)
        else: 
            self.wyh = None
            self.prods_wyh = None
        if self.xy:
            self.wxy = asmatrix(self.params[numin*numout*nummap+
                              self.yh*numout*nummap:\
                              numin*numout*nummap+self.yh*numout*nummap+\
                              numin*numout].reshape((numin,numout)))
            self.prods_wxy = zeros(self.wxy.shape,dtype=float)
        else: 
            self.wxy = None
            self.prods_wxy = None
        if self.xh:
            self.wxh = asmatrix(self.params[numin*numout*nummap+\
                                self.yh*numout*nummap+\
                                self.xy*numin*numout:\
                                numin*numout*nummap+self.yh*numout*nummap+\
                                self.xy*numin*numout+
                                numin*nummap].reshape((numin,nummap)))
            self.prods_wxh = zeros(self.wxh.shape,dtype=float)
        else: 
            self.wxh = None
            self.prods_wxh = None
        self.wy = asmatrix(self.params[numin*numout*nummap+\
                           self.yh*numout*nummap+\
                           self.xy*numin*numout+self.xh*numin*nummap:\
                           numin*numout*nummap+self.yh*numout*nummap+\
                           self.xy*numin*numout+self.xh*numin*nummap+numout].\
                           reshape((numout,1)))
        self.wh = asmatrix(self.params[numin*numout*nummap+\
                           self.yh*numout*nummap+\
                           self.xy*numin*numout+self.xh*numin*nummap+numout:\
                           numin*numout*nummap+self.yh*numout*nummap+\
                           self.xh*numin*numout+\
                           self.xy*numin*nummap+numout+nummap].\
                           reshape((nummap,1)))

    def grad(self,inputs,hidprobs,outputs):
        numcases = inputs.shape[1] 
        self.prods_wxyh *= 0.0
        if self.yh:
            self.prods_wyh  *= 0.0
        if self.xy:
            self.prods_wxy  *= 0.0
        if self.xh:
            self.prods_wxh  *= 0.0
        if self.optlevel > 0:
            from scipy import weave
            code = r"""
            for (int c=0; c<numcases; c++){
              for (int i=0; i<numin; i++){
                for (int j=0; j<numout; j++){
                  if(xy>0) prods_wxy(i,j) += inputs(i,c)*outputs(j,c);
                  for (int k=0; k<nummap; k++){
                    prods_wxyh(i,j,k)+=inputs(i,c)*outputs(j,c)*hidprobs(k,c);
                  }
                }
              }
              for (int k=0; k<nummap; k++){
                if(xh>0){
                  for (int i=0; i<numin; i++){
                    prods_wxh(i,k) += inputs(i,c)*hidprobs(k,c);
                  }
                }
                if(yh>0){
                  for (int j=0; j<numout; j++){
                    prods_wyh(j,k) += hidprobs(k,c)*outputs(j,c);
                  }
                }
              }
            }
            """
            vars = ['inputs','outputs','hidprobs','numcases','numin','nummap',\
                    'numout','prods_wxyh','prods_wxy','prods_wxh','prods_wyh',\
                    'xy','xh','yh']
            global_dict = {'prods_wxyh':self.prods_wxyh,\
                  'inputs':inputs,'outputs':outputs,'hidprobs':hidprobs,\
                  'numout':self.numout,'numin':self.numin,\
                  'nummap':self.nummap,'numcases':numcases,\
                  'prods_wxy':array(float(self.xy),ndmin=1) and self.prods_wxy,\
                  'prods_wxh':array(float(self.xh),ndmin=1) and self.prods_wxh,\
                  'prods_wyh':array(float(self.yh),ndmin=1) and self.prods_wyh,\
                  'xy':int(self.xy),'xh':int(self.xh),'yh':int(self.yh)\
                     }
            weave.inline(code,vars,global_dict=global_dict,
                                        type_converters=weave.converters.blitz)
        else:
            self.prods_wxyh += sum(   inputs.T[:,:,newaxis,newaxis]\
                                   * outputs.T[:,newaxis,:,newaxis]\
                                   *hidprobs.T[:,newaxis,newaxis,:],0)
            if self.yh:
              self.prods_wyh += \
                          sum(outputs.T[:,:,newaxis]*hidprobs.T[:,newaxis,:],0)
            if self.xh:
              self.prods_wxh += \
                          sum(inputs.T[:,:,newaxis]*hidprobs.T[:,newaxis,:],0)
            if self.xy:
              self.prods_wxy += \
                          sum(inputs.T[:,:,newaxis]*outputs.T[:,newaxis,:],0)
        visact = sum(outputs,1).flatten()
        hidact = sum(hidprobs,1).flatten()
        if self.sparsitygain > 0.0:
            spxyh, spyh, spxh, sph = self.sparsityKLpenalty(hidprobs,inputs,outputs)
        else:
            spxyh = zeros((self.numin,self.numout,self.nummap))
            spyh  = zeros((self.numout,self.nummap))
            spxh  = zeros((self.numin,self.nummap))
            sph   = zeros((self.nummap))
        gradxyh = reshape(self.prods_wxyh/numcases - self.sparsitygain*spxyh,
                                           self.numin*self.numout*self.nummap)
        if self.yh: gradyh=\
                      reshape(self.prods_wyh/numcases-self.sparsitygain*spyh,\
                                                      self.numout*self.nummap)
        else: gradyh = array([],ndmin=1)
        if self.xy: gradxy = \
                       reshape(self.prods_wxy/numcases,self.numin*self.numout)
        else: gradxy = array([],ndmin=1)
        if self.xh: gradxh=\
                     reshape(self.prods_wxh/numcases-self.sparsitygain*spxh,\
                                                       self.numin*self.nummap)
        else: gradxh = array([],ndmin=1)
        grady   = reshape(visact/numcases,self.numout) 
        gradh   = reshape(hidact/numcases - self.sparsitygain*sph,self.nummap)
        return concatenate((gradxyh,\
                            gradyh,\
                            gradxy,\
                            gradxh,\
                            grady,\
                            gradh\
                           ))

    def modulatedweights(self, inputs):
        numcases = inputs.shape[1]
        numin  = self.numin
        numout = self.numout
        nummap = self.nummap
        wxyh_ = zeros((numcases,numout,nummap),dtype=float)
        if self.optlevel>0:
            from scipy import weave
            code = r""" 
              for(int c=0;c<numcases;c++){
                for(int i=0;i<numin;i++){
                  for(int j=0;j<numout;j++){
                    for(int k=0;k<nummap;k++){
                      wxyh_(c,j,k) += inputs(i,c)*wxyh(i,j,k);
                    }
                  }
                }
              }
            """
            weave.inline(code,['wxyh','numcases','numin','inputs','wxyh_',\
                            'nummap','numout'],\
                           global_dict={'wxyh':self.wxyh,'inputs':inputs,
                           'wxyh_':wxyh_,
                           'numcases':numcases,'numin':numin,'nummap':nummap},\
                           type_converters=weave.converters.blitz)
        else:
            wxyh_=sum(inputs.T[:,:,newaxis,newaxis]*self.wxyh[newaxis,:,:,:],1)
        if self.xy: wxy_ = inputs.T*self.wxy
        else: wxy_ = None
        if self.xh: wxh_ = inputs.T*self.wxh
        else: wxh_ = None
        return wxyh_, wxy_, wxh_

    def sparsityKLpenalty(self,hidprobs,inputs,outputs):
        spxyh = (sum(hidprobs,1).T/double(hidprobs.shape[1])\
                            -self.targethidprobs)*\
                            sum(inputs.T[:,:,newaxis,newaxis]*\
                            outputs.T[:,newaxis,:,newaxis],0)\
                            /double(hidprobs.shape[1])
        if self.xh: spxh = (sum(hidprobs,1).T/double(hidprobs.shape[1])\
                           -self.targethidprobs)[newaxis,:] * \
                            sum(inputs,1)[:,newaxis]\
                            /double(hidprobs.shape[1])
        else: spxh = None
        if self.yh: spyh = (sum(hidprobs,1).T/double(hidprobs.shape[1])\
                           -self.targethidprobs)[newaxis,:] * \
                           sum(outputs,1)[:,newaxis]\
                           /double(hidprobs.shape[1])
        else: spyh = None
        sph = ((sum(hidprobs,1)/double(hidprobs.shape[1])\
                           -self.targethidprobs)[:,newaxis] \
                           /double(hidprobs.shape[1])).flatten()
        return spxyh, spyh, spxh, sph
  

class GbmChanScore(GbmScore):
    def grad(self,inputs,hidprobs,outputs):
        numcases = inputs.shape[1] 
        self.prods_wxyh *= 0.0
        if self.yh:
            self.prods_wyh *= 0.0
        if self.xy:
            self.prods_wxy *= 0.0
        if self.xh:
            self.prods_wxh *= 0.0
        if self.optlevel > 0:
            from scipy import weave
            code = r"""
            for (int c=0; c<numcases; c++){
              for (int i=0; i<numin; i++){
                for (int j=0; j<numout; j++){
                  if(xy>0) prods_wxy(i,j) += inputs(i,c)*outputs(j,c);
                  for (int k=0; k<nummap; k++){
                    prods_wxyh(i,j,k)+=inputs(i,c)*outputs(j,c)*hidprobs(k,c);
                  }
                }
              }
              for (int k=0; k<nummap; k++){
                if(xh>0){
                  for (int i=0; i<numin; i++){
                    prods_wxh(i,k) += inputs(i,c)*hidprobs(k,c);
                  }
                }
                if(yh>0){
                  for (int j=0; j<numout; j++){
                    prods_wyh(j,k) += hidprobs(k,c)*outputs(j,c);
                  }
                }
              }
            }
            """
            vars = ['inputs','outputs','hidprobs','numcases','numin','nummap',\
                    'numout','prods_wxyh','prods_wxy','prods_wxh','prods_wyh',\
                    'xy','xh','yh']
            global_dict = {'prods_wxyh':self.prods_wxyh,\
                'inputs':inputs,'outputs':outputs,'hidprobs':hidprobs,\
                'numout':self.numout,'numin':self.numin,\
                'nummap':self.nummap,'numcases':numcases,\
                'prods_wxy':array(float(self.xy),ndmin=1) and self.prods_wxy,\
                'prods_wxh':array(float(self.xh),ndmin=1) and self.prods_wxh,\
                'prods_wyh':array(float(self.yh),ndmin=1) and self.prods_wyh,\
                'xy':int(self.xy),'xh':int(self.xh),'yh':int(self.yh)\
                }
            weave.inline(code,vars,global_dict=global_dict,
                                              type_converters=weave.converters.blitz)
        else:
            self.prods_wxyh += sum(   inputs[:,:,:,newaxis]\
                                   * outputs.T[:,newaxis,:,newaxis]\
                                   *hidprobs.T[:,newaxis,newaxis,:],0)
            if self.yh:
                self.prods_wyh += \
                          sum(outputs.T[:,:,newaxis]*hidprobs.T[:,newaxis,:],0)
            if self.xh:
                self.prods_wxh += \
                   sum(inputs[:,:,:,newaxis]*hidprobs.T[:,newaxis,newaxis,:],0)
            if self.xy:
                self.prods_wxy += sum(inputs[:,:,:]*outputs.T[:,newaxis,:],0)
        visact = sum(outputs,1).flatten()
        hidact = sum(hidprobs,1).flatten()
        if self.sparsitygain > 0.0:
            spxyh, spyh, spxh, sph = \
                                self.sparsityKLpenalty(hidprobs,inputs,outputs)
        else:
            spxyh = zeros((self.numin,self.numout,self.nummap))
            spyh  = zeros((self.numout,self.nummap))
            spxh  = zeros((self.numin,self.nummap))
            sph   = zeros((self.nummap))
        gradxyh = reshape(self.prods_wxyh/numcases - self.sparsitygain*spxyh,
                                                self.numin*self.numout*self.nummap)
        if self.yh: 
            gradyh=reshape(self.prods_wyh/numcases-self.sparsitygain*spyh,\
                                                     self.numout*self.nummap)
        else: 
            gradyh = array([],ndmin=1)
        if self.xy: 
            gradxy = reshape(self.prods_wxy/numcases,self.numin*self.numout)
        else: 
            gradxy = array([],ndmin=1)
        if self.xh: 
            gradxh=reshape(self.prods_wxh/numcases-self.sparsitygain*spxh,\
                                                       self.numin*self.nummap)
        else: 
            gradxh = array([],ndmin=1)
        grady   = reshape(visact/numcases,self.numout) 
        gradh   = reshape(hidact/numcases - self.sparsitygain*sph,self.nummap)
        return concatenate((gradxyh,\
                            gradyh,\
                            gradxy,\
                            gradxh,\
                            grady,\
                            gradh\
                           ))

    def modulatedweights(self, inputs):
        numcases = inputs.shape[1]
        numin  = self.numin
        numout = self.numout
        nummap = self.nummap
        wxyh_ = zeros((numcases,numout,nummap),dtype=float)
        if self.optlevel>0:
            from scipy import weave
            code = r""" 
              for(int c=0;c<numcases;c++){
                for(int i=0;i<numin;i++){
                  for(int j=0;j<numout;j++){
                    for(int k=0;k<nummap;k++){
                      wxyh_(c,j,k) += inputs(i,c)*wxyh(i,j,k);
                    }
                  }
                }
              }
            """
            weave.inline(code,['wxyh','numcases','numin','inputs','wxyh_',\
                          'nummap','numout'],\
                          global_dict={'wxyh':self.wxyh,'inputs':inputs,
                          'wxyh_':wxyh_,
                          'numcases':numcases,'numin':numin,'nummap':nummap},\
                              type_converters=weave.converters.blitz)
        else:
            wxyh_ = sum(inputs[:,:,:,newaxis]*self.wxyh[newaxis,:,:,:],1)
        if self.xy: wxy_ = sum(inputs[:,:,:]*self.wxy[newaxis,:,:].A,1)
        else: wxy_ = None
        if self.xh: wxh_ = \
          sum(sum(inputs[:,:,:,newaxis]*self.wxh[newaxis,:,newaxis,:].A,1),2)
        else: wxh_ = None
        return wxyh_, wxy_, wxh_

    def sparsityKLpenalty(self,hidprobs,inputs,outputs):
        spxyh = (sum(hidprobs,1).T/double(hidprobs.shape[1])\
                            -self.targethidprobs)*\
                            sum(inputs.T[:,:,newaxis,newaxis]*\
                            outputs.T[:,newaxis,:,newaxis],0)\
                            /double(hidprobs.shape[1])
        if self.xh: spxh = (sum(hidprobs,1).T/double(hidprobs.shape[1])\
                           -self.targethidprobs)[newaxis,:] * \
                            sum(inputs,1)[:,newaxis]\
                            /double(hidprobs.shape[1])
        else: spxh = None
        if self.yh: spyh = (sum(hidprobs,1).T/double(hidprobs.shape[1])\
                           -self.targethidprobs)[newaxis,:] * \
                           sum(outputs,1)[:,newaxis]\
                           /double(hidprobs.shape[1])
        else: spyh = None
        sph = ((sum(hidprobs,1)/double(hidprobs.shape[1])\
                           -self.targethidprobs)[:,newaxis] \
                           /double(hidprobs.shape[1])).flatten()
        return spxyh, spyh, spxh, sph


class BpGbmScore(GbmScore):
    def __init__(self,numrawin,numin,nummap,numout,prefuncclass,\
                          sparsitygain,targethidprobs,params,\
                          xy=True,xh=True,yh=True):
        self.xy = xy
        self.xh = xh
        self.yh = yh
        self.numrawin = numrawin
        self.params = params
        self.prefunc = prefuncclass(numrawin,numin,\
                                    params[numin*numout*nummap+\
                                    self.yh*numout*nummap+\
                                    self.xy*numin*numout+
                                    self.xh*numin*nummap+\
                                    numout+nummap:])
        self.prefunc.params += params[numin*numout*nummap+
                                 self.yh*numout*nummap+
                                 self.xy*numin*numout+
                                 self.xh*numin*nummap+
                                 numout+nummap:]
        self.optlevel = 1 
        self.numin  = numin
        self.numout = numout
        self.nummap = nummap
        self.sparsitygain = sparsitygain
        self.targethidprobs = targethidprobs
        self.wxyh = self.params[:numin*numout*nummap].reshape((numin,numout,nummap))
        self.prods_wxyh = zeros(self.wxyh.shape,dtype=float)
        if self.yh:
            self.wyh = asmatrix(self.params[numin*numout*nummap:\
                                numin*numout*nummap+numout*nummap].\
                                reshape((numout,nummap)))
            self.prods_wyh = zeros(self.wyh.shape,dtype=float)
        else: 
            self.wyh = None
            self.prods_wyh = None
        if self.xy:
            self.wxy = asmatrix(self.params[numin*numout*nummap+
                                self.yh*numout*nummap:\
                                numin*numout*nummap+self.yh*numout*nummap+\
                                numin*numout].reshape((numin,numout)))
            self.prods_wxy = zeros(self.wxy.shape,dtype=float)
        else: 
            self.wxy = None
            self.prods_wxy = None
        if self.xh:
            self.wxh = asmatrix(self.params[numin*numout*nummap+\
                                self.yh*numout*nummap+\
                                self.xy*numin*numout:\
                                numin*numout*nummap+self.yh*numout*nummap+\
                                self.xy*numin*numout+
                                numin*nummap].reshape((numin,nummap)))
            self.prods_wxh = zeros(self.wxh.shape,dtype=float)
        else: 
            self.wxh = None
            self.prods_wxh = None
        self.wy = asmatrix(self.params[numin*numout*nummap+\
                           self.yh*numout*nummap+\
                           self.xy*numin*numout+self.xh*numin*nummap:\
                           numin*numout*nummap+self.yh*numout*nummap+\
                           self.xy*numin*numout+self.xh*numin*nummap+numout].\
                           reshape((numout,1)))
        self.wh = asmatrix(self.params[numin*numout*nummap+\
                           self.yh*numout*nummap+\
                           self.xy*numin*numout+self.xh*numin*nummap+numout:\
                           numin*numout*nummap+self.yh*numout*nummap+\
                           self.xy*numin*numout+self.xh*numin*nummap+\
                           numout+nummap].reshape((nummap,1)))

    def grad(self,inputs,hidprobs,outputs):
        numcases   = inputs.shape[1]
        prefuncoutputs  = self.prefunc.fprop(inputs)
        gbmgrad    = GbmScore.grad(self,prefuncoutputs,hidprobs,outputs)
        if self.optlevel>0:
            d_prefuncoutputs = zeros((self.numin,numcases),dtype=float)
            from scipy import weave
            code = r""" 
              for(int c=0;c<numcases;c++){
                for(int i=0;i<numin;i++){
                  for(int j=0;j<numout;j++){
                    for(int k=0;k<nummap;k++){
                  d_prefuncoutputs(i,c) += inputs(i,c)*outputs(j,c)*wxyh(i,j,k);
                      if(xy>0) d_prefuncoutputs(i,c) += outputs(j,c)*wxy(i,j); 
                      if(xh>0) d_prefuncoutputs(i,c) += hidprobs(k,c)*wxh(i,k);
                    }
                  }
                }
              }
            """
            vars = ['wxyh','wxy','wxh','numcases','numin',\
                    'hidprobs','outputs','inputs',\
                    'nummap','numout','d_prefuncoutputs',\
                    'xy','xh']
            global_dict = {'wxyh':self.wxyh,\
                         'wxy':array(float(self.xy),ndmin=1) and self.wxy.A,\
                         'wxh':array(float(self.xh),ndmin=1) and self.wxh.A,\
                         'numcases':numcases,\
                         'numin':self.numin,'nummap':self.nummap,\
                         'numout':self.numout,\
                         'outputs':outputs,'hidprobs':hidprobs,'inputs':inputs,\
                           'd_prefuncoutputs':d_prefuncoutputs,\
                           'xy':int(self.xy),\
                           'xh':int(self.xh)}
            weave.inline(code,vars,global_dict=global_dict,\
                         type_converters=weave.converters.blitz)
        else:
            d_prefuncoutputs = sum(sum(\
                            self.wxyh[newaxis,:,:,:]\
                            *outputs.T[:,newaxis,:,newaxis]\
                            *hidprobs.T[:,newaxis,newaxis,:]\
                            +\
                            float(self.xy) and self.wxy.A[newaxis,:,:,newaxis]\
                            *outputs.T[:,newaxis,:,newaxis]\
                            +\
                            float(self.xh) and self.wxh.A[newaxis,:,newaxis,:]\
                            *hidprobs.T[:,newaxis,newaxis,:]\
                        ,3),2).T
        self.prefunc.bprop(d_prefuncoutputs,inputs)
        prefuncgrad = sum(self.prefunc.grad(d_prefuncoutputs,inputs),1)/numcases
        return concatenate((gbmgrad,prefuncgrad))

    def modulatedweights(self, inputs):
        numcases = inputs.shape[1]
        prefuncoutputs = self.prefunc.fprop(inputs)
        if self.optlevel>0:
            wxyh_ = zeros((numcases,self.numout,self.nummap),dtype=float)
            from scipy import weave
            code = r""" 
              for(int c=0;c<numcases;c++){
                for(int i=0;i<numin;i++){
                  for(int j=0;j<numout;j++){
                    for(int k=0;k<nummap;k++){
                      wxyh_(c,j,k) += prefuncoutputs(i,c)*wxyh(i,j,k);
                    }
                  }
                }
              }
            """
            weave.inline(code,['wxyh','numcases','numin','prefuncoutputs',\
                               'wxyh_','nummap','numout'],\
                              global_dict={'wxyh':self.wxyh,\
                              'prefuncoutputs':prefuncoutputs,\
                              'wxyh_':wxyh_, 'numcases':numcases,\
                              'numin':self.numin,'nummap':self.nummap,
                              'numout':self.numout},\
                              type_converters=weave.converters.blitz)
        else:
            wxyh_=sum(prefuncoutputs.T[:,:,newaxis,newaxis]*\
                                                     self.wxyh[newaxis,:,:,:],1)
        if self.xy: wxy_ = prefuncoutputs.T*self.wxy
        else: wxy_ = None
        if self.xh: wxh_ = prefuncoutputs.T*self.wxh
        else: wxh_ = None
        return wxyh_, wxy_, wxh_

    def sparsityKLpenalty(self,hidprobs,inputs,outputs):
        raise "sparsity not yet implemented"

