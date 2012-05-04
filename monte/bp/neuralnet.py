#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from numpy import zeros, asmatrix, concatenate, outer, multiply, prod, sum, exp


class Linearlayer(object):
    """Simple (affine) linear layer."""

    @staticmethod
    def numparams(numin,numout):
        return numin*numout+numout

    def __init__(self,numin,numout,params):
        self.numin  = numin
        self.numout = numout
        self.params = params #pointer to params that were passed in 
        self.w = asmatrix(self.params[:self.numin*self.numout])\
                              .reshape((self.numout,self.numin))
        self.b = asmatrix(self.params[self.numin*self.numout:])\
                              .reshape((self.numout,1))
        self.gradw = zeros(prod(self.w.shape),dtype=float)

    def fprop(self,input):
        if len(input.shape) >= 2:
            return (self.w * asmatrix(input) + self.b).A
        else:
            return (self.w * asmatrix(input).T + self.b).A.flatten()

    def bprop(self,d_output,input):
        if len(input.shape) >= 2:
            return (self.w.T * d_output).A
        else:
            return (self.w.T * d_output).A.flatten()

    def grad(self,d_output,input):
        gradw = zeros((prod(self.w.shape),input.shape[1]),dtype=float)
        if len(input.shape) >= 2:
            for i in range(input.shape[1]):
                gradw[:,i] = outer(d_output[:,i],input[:,i]).flatten()
                gradb = d_output
        else:
            gradw += outer(d_output,input).flatten()
            gradb = d_output 
        return concatenate((gradw,gradb))


class LinearlayerNobias(object):
    """Purely linear layer, without bias. Useful in modules where the bias is 
    supposed to be provided from the outside to be learned separately.
    """

    @staticmethod
    def numparams(numin,numout):
        return numin*numout

    def __init__(self,numin,numout,params):
        self.numin  = numin
        self.numout = numout
        self.params = params #pointer to params that were passed in 
        self.w = asmatrix(self.params[:self.numin*self.numout])\
                              .reshape((self.numout,self.numin))
        self.gradw = zeros(prod(self.w.shape),dtype=float)

    def fprop(self,input):
        if len(input.shape) >= 2:
            return (self.w * asmatrix(input)).A
        else:
            return (self.w * asmatrix(input).T).A.flatten()

    def bprop(self,d_output,input):
        if len(input.shape) >= 2:
            return (self.w.T * d_output).A
        else:
            return (self.w.T * d_output).A.flatten()

    def grad(self,d_output,input):
        #self.gradw *= 0.0
        gradw = zeros((prod(self.w.shape),input.shape[1]),dtype=float)
        if len(input.shape) >= 2:
            #self.gradw += (asmatrix(d_output) * asmatrix(input).T).A.flatten()
            for i in range(input.shape[1]):
                gradw[:,i] = outer(d_output[:,i],input[:,i]).flatten()
        else:
            gradw += outer(d_output,input).flatten()
        #return concatenate((self.gradw,gradb))
        return gradw


class Softmax(object):
    """ Simple elementwise softmax."""

    def __init__(self,numunits):
        self.numunits = numunits

    def fprop(self,input):
        expinput = exp(input)
        self.output = expinput/sum(expinput,0)
        return self.output

    def bprop(self,d_output,input):
        return self.output*eye(self.numunits)-outer(self.output,self.output)

    def grad(self,d_output,input):
        if len(input.shape) >= 2:
            return zeros((0,input.shape[1]),dtype=float)
        return zeros(0,dtype=float) #empty: the softmax has no parameters


class LinearSoftmaxlayer(object):
    """ Linear layer followed by a softmax. (AKA multinomial logit model). """

    def __init__(self,numin,numout,params):
        self.numin  = numin
        self.numout = numout
        self.params = params 
        self.linearlayer = Linearlayer(self.numin,self.numout,self.params)
        self.softmax = Softmax(self.numout)

    def fprop(self,input):
        self.activities = self.linearlayer.fprop(input)
        return self.softmax.fprop(self.activities)

    def bprop(self,d_output,input):
        self.d_activities = self.softmax.bprop(d_output,self.activities)
        return self.linearlayer.bprop(self.d_activities,input)

    def grad(self,d_output,input):
        grad2 = self.softmax.grad(d_output,self.activities) #is empty
        grad1 = self.linearlayer.grad(self.d_activities,input)
        return concatenate((grad1,grad2))


class Sigmoidelementwise(object):
    """ Neural network layer that computes sigmoids (elementwise) for 
        all its inputs. Useful in combination with linearlayer to build
        a standard neural network layer. """
    def __init__(self,numunits):
        self.numunits = numunits

    def fprop(self,input):
        return 1./(1.+exp(-input))

    def bprop(self,d_output,input):
        a = 1./(1.+exp(-input))
        if len(input.shape) >= 2:
            return multiply(d_output,multiply(a,(1-a)))
        else:
            return multiply(d_output,multiply(a,(1-a)))

    def grad(self,d_output,input):
        if len(input.shape) >= 2:
            return zeros((0,input.shape[1]),dtype=float) 
        else:
            return zeros(0,dtype=float)  #empty: sigmoid has no parameters


class NNlayer(object):
    """A single layer of a neural network: Linear layer followed by elementwise 
    sigmoid.
    """

    @staticmethod
    def numparams(numin,numout):
        return Linearlayer.numparams(numin,numout)

    def __init__(self,numin,numout,params):
        self.numin  = numin
        self.numout = numout
        self.params = params 
        self.linearlayer  = Linearlayer(self.numin,self.numout,self.params)
        self.sigmoidlayer = Sigmoidelementwise(self.numout)

    def fprop(self,input):
        self.activities = self.linearlayer.fprop(input)
        return self.sigmoidlayer.fprop(self.activities)

    def bprop(self,d_output,input):
        self.d_activities = self.sigmoidlayer.bprop(d_output,self.activities)
        return self.linearlayer.bprop(self.d_activities,input)

    def grad(self,d_output,input):
        grad2 = self.sigmoidlayer.grad(d_output,self.activities) #is empty
        grad1 = self.linearlayer.grad(self.d_activities,input)
        return concatenate((grad1,grad2))


class SigmoidLinear(object):
    """ Neural network that consists of a sigmoid layer, followed by a 
    linear layer. """

    @staticmethod
    def numparams(numin,numhid,numout):
        """Computes the number of parameters that would be required for the 
           given model dimensions."""
        return numin*numhid+numhid+numhid*numout+numout

    def __init__(self,numin,numhid,numout,params):
        self.numin  = numin
        self.numhid = numhid
        self.numout = numout
        self.params = params
        self.hiddenlayer = NNlayer(self.numin,self.numhid,\
                              self.params[:self.numin*self.numhid+self.numhid])
        self.outputlayer = Linearlayer(self.numhid,self.numout,\
                              self.params[self.numin*self.numhid+self.numhid:])

    def fprop(self,input):
        self.hiddens = self.hiddenlayer.fprop(input)
        return self.outputlayer.fprop(self.hiddens)

    def bprop(self,d_output,input):
        self.d_hiddens = self.outputlayer.bprop(d_output,self.hiddens)
        return self.hiddenlayer.bprop(self.d_hiddens,input)

    def grad(self,d_output,input):
        grad1 = self.hiddenlayer.grad(self.d_hiddens,input)
        grad2 = self.outputlayer.grad(d_output,self.hiddens)
        return concatenate((grad1,grad2))


class SigmoidLinearNobias(object):
    """ Neural network that consists of a sigmoid layer, followed by a
    linear layer without biases. """
    @staticmethod
    def numparams(numin,numhid,numout):
        """Computes the number of parameters that would be required for the 
           given model dimensions."""
        return numin*numhid+numhid+numhid*numout
    def __init__(self,numin,numhid,numout,params):
        self.numin  = numin
        self.numhid = numhid
        self.numout = numout
        self.params = params
        self.hiddenlayer = NNlayer(self.numin,self.numhid,\
                              self.params[:self.numin*self.numhid+self.numhid])
        self.outputlayer = LinearlayerNobias(self.numhid,self.numout,\
                              self.params[self.numin*self.numhid+self.numhid:])
    def fprop(self,input):
        self.hiddens = self.hiddenlayer.fprop(input)
        return self.outputlayer.fprop(self.hiddens)
    def bprop(self,d_output,input):
        self.d_hiddens = self.outputlayer.bprop(d_output,self.hiddens)
        return self.hiddenlayer.bprop(self.d_hiddens,input)
    def grad(self,d_output,input):
        grad1 = self.hiddenlayer.grad(self.d_hiddens,input)
        grad2 = self.outputlayer.grad(d_output,self.hiddens)
        return concatenate((grad1,grad2))


class Islsl(object):
    """ Network with the structure
        input -> sigmoid -> linear -> sigmoid -> linear 
        Useful e.g. as autoencoder. """

    @staticmethod
    def numparams(numin,numhid1,numhid2,numhid3,numout):
        """ Computes the number of parameters that would be required for the 
           given model dimensions. """
        return numin*numhid1+numhid1 + numhid1*numhid2+numhid2 +\
                    numhid2*numhid3+numhid3 + numhid3*numout+numout

    def __init__(self,numin,numhid1,numhid2,numhid3,numout,params):
        self.numin  = numin
        self.numhid1 = numhid1
        self.numhid2 = numhid2
        self.numhid3 = numhid3
        self.numout = numout
        self.params = params
        self.layer1 = SigmoidLinear(self.numin,self.numhid1,self.numhid2,\
                self.params[:SigmoidLinear.numparams(numin,numhid1,numhid2)])
        self.layer2 = SigmoidLinear(self.numhid2,self.numhid3,self.numout,\
                self.params[-SigmoidLinear.numparams(\
                               self.numhid2,self.numhid3,self.numout):])

    def fprop(self,input):
        self.hiddens = self.layer1.fprop(input)
        return self.layer2.fprop(self.hiddens)

    def bprop(self,d_output,input):
        self.d_hiddens = self.layer2.bprop(d_output,self.hiddens)
        return self.layer1.bprop(self.d_hiddens,input)

    def grad(self,d_output,input):
        grad1 = self.layer1.grad(self.d_hiddens,input)
        grad2 = self.layer2.grad(d_output,self.hiddens)
        return concatenate((grad1,grad2))


class IslslNobias(object):
    """ Network with the structure
        input -> sigmoid -> linear -> sigmoid -> linear 
        but without output-biases.
        Useful e.g. as autoencoder within as structured 
        prediction model. """

    @staticmethod
    def numparams(numin,numhid1,numhid2,numhid3,numout):
        """ Computes the number of parameters that would be required for the 
            given model dimensions. """
        return numin*numhid1+numhid1 + numhid1*numhid2+numhid2 +\
                    numhid2*numhid3+numhid3 + numhid3*numout

    def __init__(self,numin,numhid1,numhid2,numhid3,numout,params):
        self.numin  = numin
        self.numhid1 = numhid1
        self.numhid2 = numhid2
        self.numhid3 = numhid3
        self.numout = numout
        self.params = params
        self.layer1 = SigmoidLinear(self.numin,self.numhid1,self.numhid2,\
                self.params[:SigmoidLinear.numparams(numin,numhid1,numhid2)])
        self.layer2 = SigmoidLinearNobias(self.numhid2,self.numhid3,
                             self.numout,self.params[
                             SigmoidLinear.numparams(numin,numhid1,numhid2):])

    def fprop(self,input):
        self.hiddens = self.layer1.fprop(input)
        return self.layer2.fprop(self.hiddens)

    def bprop(self,d_output,input):
        self.d_hiddens = self.layer2.bprop(d_output,self.hiddens)
        return self.layer1.bprop(self.d_hiddens,input)

    def grad(self,d_output,input):
        grad1 = self.layer1.grad(self.d_hiddens,input)
        grad2 = self.layer2.grad(d_output,self.hiddens)
        return concatenate((grad1,grad2))


class GatedLinear(object):
    """(Affine) linear layer, whose connections are modulated by extra 
        inputs.
    """
    @staticmethod
    def numparams(numin,numout,numgate):
        return numin*numout*numgate+numout
    def __init__(self,numin,numout,numgate,params,maxcases=None):
        self.optlevel = 1   #set optimization level to 1 (ie., use inline C)
        self.maxcases = maxcases #the maximum number of cases to preallocate mem
        self.numin  = numin
        self.numout = numout
        self.numgate = numgate
        self.params = params #pointer to params that were passed in 
        self.w = self.params[:self.numin*self.numout*self.numgate]\
                              .reshape((self.numgate,self.numin,self.numout))
        self.b = self.params[self.numin*self.numout*self.numgate:]\
                              .reshape((self.numout,1))
        #self.gradw = zeros(prod(self.w.shape),dtype=float)
        if self.maxcases:
            self.gradw = \
               zeros((self.numin*self.numout*self.numgate,self.maxcases),\
                     dtype=float)
            self.d_inputs = zeros((self.numin,self.maxcases),dtype=float)
            self.outputs = zeros((self.numout,maxcases),dtype=float)
    def fprop(self,inputs,gates):
        if len(inputs.shape) < 2:
            inputs = inputs[:,newaxis]
            gates = gates[:,newaxis]
        if self.optlevel > 0:
            numcases = inputs.shape[1]
            if self.maxcases:
                outputs = self.outputs[:,:numcases]
                outputs *= 0.0
            else: 
                outputs = zeros((self.numout,numcases),dtype=float)
            from scipy import weave
            code = r"""
            for (int c=0; c<numcases; c++){
              for (int j=0; j<numout; j++){
                for (int i=0; i<numin; i++){
                  for (int k=0; k<numgate; k++)
                    outputs(j,c) += inputs(i,c)*gates(k,c)*weights(k,i,j);
                }
                outputs(j,c) += biases(j);
              }
            }"""
            global_dict = {'inputs':inputs,'outputs':outputs,'gates':gates,\
                           'weights':self.w,'biases':self.b,\
                           'numout':self.numout,'numin':self.numin,\
                           'numgate':self.numgate,'numcases':numcases}
            weave.inline(code,global_dict.keys(),global_dict=global_dict,
                                       type_converters=weave.converters.blitz)
            return outputs
        else:
            return sum(sum(self.w[newaxis,:,:,:] * \
                       gates.T[:,:,newaxis,newaxis] * \
                       inputs.T[:,newaxis,:,newaxis],1),1).transpose(1,0) \
                       + self.b
    def bprop(self,d_outputs,inputs,gates):
        if len(inputs.shape) < 2:
            inputs = inputs[:,newaxis]
            gates = gates[:,newaxis]
        if self.optlevel > 0:
            numcases = inputs.shape[1]
            if self.maxcases:
                d_inputs = self.d_inputs[:,:numcases]
                d_inputs *= 0.0
            else: 
                d_inputs = zeros((self.numin,numcases),dtype=float)
            from scipy import weave
            code = r"""
            for (int c=0; c<numcases; c++)
              for (int i=0; i<numin; i++)
                for (int j=0; j<numout; j++)
                  for (int k=0; k<numgate; k++)
                    d_inputs(i,c) += d_outputs(j,c)*gates(k,c)*weights(k,i,j); 
            """
            global_dict = \
                     {'d_inputs':d_inputs,'d_outputs':d_outputs,'gates':gates,\
                      'weights':self.w,\
                      'numout':self.numout,'numin':self.numin,\
                      'numgate':self.numgate,'numcases':numcases}
            weave.inline(code,global_dict.keys(),global_dict=global_dict,
                                        type_converters=weave.converters.blitz)
            return d_inputs
        else:
            return sum(sum(self.w[newaxis,:,:,:]*\
                         gates.T[:,:,newaxis,newaxis]*\
                         d_outputs.T[:,newaxis,newaxis,:]\
                        ,1),2).transpose(1,0)
    def grad(self,d_outputs,inputs,gates):
        if len(inputs.shape) < 2:
            inputs = inputs[:,newaxis]
            gates = gates[:,newaxis]
        numcases = inputs.shape[1]
        if self.maxcases:
            gradw = self.gradw[:,:numcases]
            gradw *= 0.0
        else:
            gradw = \
              zeros((self.numin*self.numout*self.numgate,numcases),dtype=float)
        if self.optlevel > 0:
            from scipy import weave
            code = r"""
            for (int c=0; c<numcases; c++)
              for (int k=0; k<numgate; k++)
                for (int i=0; i<numin; i++)
                  for (int j=0; j<numout; j++)
                    gradw(k*(numin*numout)+i*numout+j,c)=inputs(i,c)*gates(k,c)*d_outputs(j,c);
            """
            global_dict = {'inputs':inputs,'d_outputs':d_outputs,'gates':gates,\
                           'gradw':gradw,\
                           'numout':self.numout,'numin':self.numin,\
                           'numgate':self.numgate,'numcases':numcases}
            weave.inline(code,global_dict.keys(),global_dict=global_dict,
                                        type_converters=weave.converters.blitz)
        else:
            gradw += (gates.T[:,:,newaxis,newaxis]\
                           *inputs.T[:,newaxis,:,newaxis]\
                           *d_outputs.T[:,newaxis,newaxis,:]).\
                 reshape(inputs.shape[1],self.numin*self.numout*self.numgate).\
                    transpose(1,0)
        gradb = d_outputs
        return concatenate((gradw,gradb))


class GatedLinearNobias(object):
    """ Linear layer, whose connections are modulated by extra inputs.
        There are no biases on the outputs. 
    """
    @staticmethod
    def numparams(numin,numout,numgate):
        return numin*numout*numgate
    def __init__(self,numin,numout,numgate,params,maxcases=None):
        self.optlevel = 1   #set optimization level to 1 (ie., use inline C)
        self.maxcases = maxcases #the maximum number of cases to preallocate mem
        self.numin  = numin
        self.numout = numout
        self.numgate = numgate
        self.params = params #pointer to params that were passed in 
        self.w = self.params[:self.numin*self.numout*self.numgate]\
                              .reshape((self.numgate,self.numin,self.numout))
        #self.gradw = zeros(prod(self.w.shape),dtype=float)
        if self.maxcases:
            self.gradw = \
               zeros((self.numin*self.numout*self.numgate,self.maxcases),\
                     dtype=float)
            self.d_inputs = zeros((self.numin,self.maxcases),dtype=float)
            self.outputs = zeros((self.numout,maxcases),dtype=float)
    def fprop(self,inputs,gates):
        if len(inputs.shape) < 2:
            inputs = inputs[:,newaxis]
            gates = gates[:,newaxis]
        if self.optlevel > 0:
            numcases = inputs.shape[1]
            if self.maxcases:
                outputs = self.outputs[:,:numcases]
                outputs *= 0.0
            else: 
                outputs = zeros((self.numout,numcases),dtype=float)
            from scipy import weave
            code = r"""
            for (int c=0; c<numcases; c++){
              for (int j=0; j<numout; j++){
                for (int i=0; i<numin; i++){
                  for (int k=0; k<numgate; k++){
                    outputs(j,c) += inputs(i,c)*gates(k,c)*weights(k,i,j);
                  } 
                }
              }
            }"""
            global_dict = {'inputs':inputs,'outputs':outputs,'gates':gates,\
                           'weights':self.w,\
                           'numout':self.numout,'numin':self.numin,\
                           'numgate':self.numgate,'numcases':numcases}
            weave.inline(code,global_dict.keys(),global_dict=global_dict,
                                       type_converters=weave.converters.blitz)
            return outputs
        else:
            return sum(sum(self.w[newaxis,:,:,:] * \
                       gates.T[:,:,newaxis,newaxis] * \
                       inputs.T[:,newaxis,:,newaxis],1),1).transpose(1,0)
    def bprop(self,d_outputs,inputs,gates):
        if len(inputs.shape) < 2:
            inputs = inputs[:,newaxis]
            gates = gates[:,newaxis]
        if self.optlevel > 0:
            numcases = inputs.shape[1]
            if self.maxcases:
                d_inputs = self.d_inputs[:,:numcases]
                d_inputs *= 0.0
            else: 
                d_inputs = zeros((self.numin,numcases),dtype=float)
            from scipy import weave
            code = r"""
            for (int c=0; c<numcases; c++)
              for (int i=0; i<numin; i++)
                for (int j=0; j<numout; j++)
                  for (int k=0; k<numgate; k++)
                    d_inputs(i,c) += d_outputs(j,c)*gates(k,c)*weights(k,i,j); 
            """
            global_dict = \
                     {'d_inputs':d_inputs,'d_outputs':d_outputs,'gates':gates,\
                      'weights':self.w,\
                      'numout':self.numout,'numin':self.numin,\
                      'numgate':self.numgate,'numcases':numcases}
            weave.inline(code,global_dict.keys(),global_dict=global_dict,
                                        type_converters=weave.converters.blitz)
            return d_inputs
        else:
            return sum(sum(self.w[newaxis,:,:,:]*\
                         gates.T[:,:,newaxis,newaxis]*\
                         d_outputs.T[:,newaxis,newaxis,:]\
                        ,1),2).transpose(1,0)
    def grad(self,d_outputs,inputs,gates):
        if len(inputs.shape) < 2:
            inputs = inputs[:,newaxis]
            gates = gates[:,newaxis]
        numcases = inputs.shape[1]
        if self.maxcases:
            gradw = self.gradw[:,:numcases]
            gradw *= 0.0
        else:
            gradw = \
              zeros((self.numin*self.numout*self.numgate,numcases),dtype=float)
        if self.optlevel > 0:
            from scipy import weave
            code = r"""
            for (int c=0; c<numcases; c++)
              for (int k=0; k<numgate; k++)
                for (int i=0; i<numin; i++)
                  for (int j=0; j<numout; j++)
                    gradw(k*(numin*numout)+i*numout+j,c)=inputs(i,c)*gates(k,c)*d_outputs(j,c);
            """
            global_dict = {'inputs':inputs,'d_outputs':d_outputs,'gates':gates,\
                           'gradw':gradw,\
                           'numout':self.numout,'numin':self.numin,\
                           'numgate':self.numgate,'numcases':numcases}
            weave.inline(code,global_dict.keys(),global_dict=global_dict,
                                        type_converters=weave.converters.blitz)
        else:
            gradw += (gates.T[:,:,newaxis,newaxis]\
                           *inputs.T[:,newaxis,:,newaxis]\
                           *d_outputs.T[:,newaxis,newaxis,:]).\
                 reshape(inputs.shape[1],self.numin*self.numout*self.numgate).\
                    transpose(1,0)
        return gradw


class GatedLsl(object):
    """ Gated neural network of the form inputs->linear->sigmoid->linear. The 
        parameters (ie. the two linear layers) are modulated by extra inputs.
    """
    @staticmethod
    def numparams(numin,numhid,numout,numgate):
        return numin*numhid*numgate+numhid + numhid*numout*numgate + numout
    def __init__(self,numin,numhid,numout,numgate,params,maxcases=None):
        self.numin   = numin
        self.numhid  = numhid
        self.maxcases= maxcases
        self.params  = params 
        self.numout  = numout
        self.numgate = numgate
        numparamsfirstlayer = GatedLinear.numparams(numin,numhid,numgate)
        self.layer1  = GatedLinear(self.numin,self.numhid,self.numgate,\
                               self.params[:numparamsfirstlayer],self.maxcases)
        self.sigmoid = Sigmoidelementwise(self.numhid)
        self.layer2  = GatedLinear(self.numhid,self.numout,self.numgate,\
                              self.params[numparamsfirstlayer:],self.maxcases)
    def fprop(self,inputs,gates):
        self.hidactivities = self.layer1.fprop(inputs,gates)
        self.hiddens = self.sigmoid.fprop(self.hidactivities)
        return self.layer2.fprop(self.hiddens,gates)
    def bprop(self,d_outputs,inputs,gates):
        self.d_hiddens = self.layer2.bprop(d_outputs,inputs,gates)
        self.d_hidactivities = \
                          self.sigmoid.bprop(self.d_hiddens,self.hidactivities)
        return self.layer1.bprop(self.d_hidactivities,inputs,gates)
    def mappings(self,inputs,gates):
        hidactivities = self.layer1.fprop(inputs,gates)
        return self.sigmoid.fprop(hidactivities)
    def outputs(self,hiddens,gates):
        return self.layer2.fprop(hiddens,gates)
    def grad(self,d_outputs,inputs,gates):
        grad2 = self.layer2.grad(d_outputs,self.hiddens,gates)
        grad1 = self.layer1.grad(self.d_hidactivities,inputs,gates)
        return concatenate((grad1,grad2))


class GatedLslNobias(object):
    """ Gated neural network of the form inputs->linear->sigmoid->linear. The 
        parameters (ie. the two linear layers) are modulated by extra inputs.
    """
    @staticmethod
    def numparams(numin,numhid,numout,numgate):
        return numin*numhid*numgate+numhid + numhid*numout*numgate 
    def __init__(self,numin,numhid,numout,numgate,params,maxcases=None):
        self.numin   = numin
        self.numhid  = numhid
        self.maxcases= maxcases
        self.params  = params 
        self.numout  = numout
        self.numgate = numgate
        numparamsfirstlayer = GatedLinear.numparams(numin,numhid,numgate)
        self.layer1  = GatedLinear(self.numin,self.numhid,self.numgate,\
                               self.params[:numparamsfirstlayer],self.maxcases)
        self.sigmoid = Sigmoidelementwise(self.numhid)
        self.layer2  = GatedLinearNobias(self.numhid,self.numout,self.numgate,\
                              self.params[numparamsfirstlayer:],self.maxcases)
    def fprop(self,inputs,gates):
        self.hidactivities = self.layer1.fprop(inputs,gates)
        self.hiddens = self.sigmoid.fprop(self.hidactivities)
        return self.layer2.fprop(self.hiddens,gates)
    def bprop(self,d_outputs,inputs,gates):
        self.d_hiddens = self.layer2.bprop(d_outputs,inputs,gates)
        self.d_hidactivities = \
                          self.sigmoid.bprop(self.d_hiddens,self.hidactivities)
        return self.layer1.bprop(self.d_hidactivities,inputs,gates)
    def mappings(self,inputs,gates):
        hidactivities = self.layer1.fprop(inputs,gates)
        return self.sigmoid.fprop(hidactivities)
    def outputs(self,hiddens,gates):
        return self.layer2.fprop(hiddens,gates)
    def grad(self,d_outputs,inputs,gates):
        grad2 = self.layer2.grad(d_outputs,self.hiddens,gates)
        grad1 = self.layer1.grad(self.d_hidactivities,inputs,gates)
        return concatenate((grad1,grad2))

