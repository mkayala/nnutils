#Copyright (C) 2007-2008 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from numpy import hstack
from numpy.random import randn

from monte import bp

class LinearRegression(object):
    """Linear regression with a backprop-network."""

    def __init__(self,numin, numout):
        self.numin  = numin
        self.numout = numout
        self.params = 0.01 * \
                      randn(bp.neuralnet.Linearlayer.numparams(numin, numout))
        self.bpmodule = bp.neuralnet.Linearlayer(numin, numout, self.params)

    def cost(self, data, weightcost):
        inputs, outputs = data
        modeloutputs = self.bpmodule.fprop(inputs)
        return ((modeloutputs - outputs)**2).sum() + \
                weightcost * (self.params**2).sum()

    def grad(self, data, weightcost):
        inputs, outputs = data
        modeloutputs = self.bpmodule.fprop(inputs)
        d_outputs = 2 * (modeloutputs-outputs) 
        self.bpmodule.bprop(d_outputs, inputs)
        return self.bpmodule.grad(d_outputs, inputs).sum(1) + \
               2 * weightcost * self.params

    def apply(self, inputs):
        return self.bpmodule.fprop(inputs)


class LinearLinear(object):
    """ Neural network consisting of two linear layers. 
    """

    def __init__(self,numin, numhid, numout):
        self.numin  = numin
        self.numhid = numhid
        self.numout = numout
        self.numparamslayer1 = bp.neuralnet.Linearlayer.numparams(
                                                                numin, numhid)
        self.numparamslayer2 = bp.neuralnet.Linearlayer.numparams(
                                                                numhid, numout)
        self.params = 0.01 * randn(self.numparamslayer1 + self.numparamslayer2)
        self.layer1 = bp.neuralnet.Linearlayer(numin, numhid, 
                           self.params[:self.numparamslayer1])
        self.layer2 = bp.neuralnet.Linearlayer(numhid, numout, \
                           self.params[self.numparamslayer1:])
        self.costmodule = bp.cost.Squarederror()

    def cost(self, data, weightcost):
        inputs, outputs = data
        return self.costmodule.fprop(self.apply(inputs), outputs)
        #modeloutputs = self.apply(inputs)
        #return ((modeloutputs - outputs)**2).sum() + \
        #        weightcost * (self.params**2).sum()

    def grad(self, data, weightcost):
        inputs, outputs = data
        modeloutputs = self.apply(inputs)
        d_outputs = self.costmodule.bprop(modeloutputs, outputs)
        #d_outputs = 2 * (modeloutputs-outputs) 
        d_hiddens = self.layer2.bprop(d_outputs, self.hiddens)
        self.layer1.bprop(d_hiddens, inputs)
        grad2 = self.layer2.grad(d_outputs, self.hiddens).sum(1)
        grad1 = self.layer1.grad(d_hiddens, inputs).sum(1)
        return hstack((grad1, grad2)) + 2 * weightcost * self.params

    def apply(self, inputs):
        self.hiddens = self.layer1.fprop(inputs)
        return self.layer2.fprop(self.hiddens)


class NNIsl(object):
    """Sigmoid layer followed by a linear layer."""

    def __init__(self,numin, numhid, numout):
        self.numin  = numin
        self.numhid = numhid
        self.numout = numout
        self.params = 0.01 * \
            randn(bp.neuralnet.SigmoidLinear.numparams(numin, numhid, numout))
        self.bpmodule = bp.neuralnet.SigmoidLinear(numin, numhid, numout, 
                                                   self.params)

    def cost(self, data, weightcost):
        inputs, outputs = data
        modeloutputs = self.bpmodule.fprop(inputs)
        return ((modeloutputs - outputs)**2).sum() + \
                weightcost * (self.params**2).sum()

    def grad(self, data, weightcost):
        inputs, outputs = data
        modeloutputs = self.bpmodule.fprop(inputs)
        d_outputs = 2 * (modeloutputs-outputs) 
        self.bpmodule.bprop(d_outputs, inputs)
        return self.bpmodule.grad(d_outputs, inputs).sum(1) + \
               2 * weightcost * self.params

    def apply(self, inputs):
        return self.bpmodule.fprop(inputs)


class NNIslsl(object):
    """ Neural network with the structure:
        input -> sigmoid layer -> linear layer -> sigmoid layer -> linear layer 
    """

    def __init__(self,numin, numhid1, numhid2, numhid3, numout):
        self.numin  = numin
        self.numhid1 = numhid1
        self.numhid2 = numhid2
        self.numhid3 = numhid3
        self.numout = numout
        self.params = 0.01 * \
            randn(bp.neuralnet.Islsl.numparams(numin, numhid1, 
                                                   numhid2, numhid3, numout))
        self.bpmodule = bp.neuralnet.Islsl(numin, numhid1, numhid2, 
                                                numhid3, numout, self.params)

    def cost(self, data, weightcost):
        inputs, outputs = data
        modeloutputs = self.bpmodule.fprop(inputs)
        return ((modeloutputs - outputs)**2).sum() + \
                weightcost * (self.params**2).sum()

    def grad(self, data, weightcost):
        inputs, outputs = data
        modeloutputs = self.bpmodule.fprop(inputs)
        d_outputs = 2 * (modeloutputs-outputs) 
        self.bpmodule.bprop(d_outputs, inputs)
        return self.bpmodule.grad(d_outputs, inputs).sum(1) + \
               2 * weightcost * self.params

    def apply(self, inputs):
        return self.bpmodule.fprop(inputs)


