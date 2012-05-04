#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.


from numpy import double, sum, zeros, argmax, newaxis, exp
from pylab import randn, find
from monte.models.contrastive.contrastive import Contrastive
from monte.models.contrastive.scorefunc import scorefunc
from monte.util.util import logsumexp, onehot
from monte import bp

class Isl(Contrastive):
    """Backprop network with one sigmoid hidden and one linear output layer."""
    
    def __init__(self,numin,numhid,numout):
        self.numin  = numin
        self.numhid = numhid
        self.numout = numout
        self.params = 0.01 * randn(self.numin*self.numhid+self.numhid+\
                                   self.numhid*self.numout+self.numout)
        self.scorefuncs = [scorefunc.SigmoidhiddenLinearoutputScore\
                                       (numin,numhid,numout,self.params)]
        Contrastive.__init__(self,normalizeacrosscliques=False)
    
    def posdata(self,data):
        return (((data[0],data[1]),),)

    def negdata(self,data):
        return ([],)   #no negdata: scorefunc takes care of it

    def cost(self,data,weightcost):
        if type(data)!=type([]):
            data = [data]
        numcases = len(data)
        cost = 0.0
        for i in range(numcases):
            input = data[i][0]
            desiredoutput = data[i][1]
            output = self.scorefuncs[0](input)
            if len(input.shape) >= 2:
                cost += sum((output-desiredoutput)**2)/\
                                                double(numcases*input.shape[1])
            else:
                cost += sum((output-desiredoutput)**2)/double(numcases)
        cost += 0.5 * weightcost * sum(self.params**2)
        return cost

    def hiddens(self,inputs):
        if type(inputs) != type([]):
            inputs = [inputs]
        numcases = len(inputs)
        result = []
        for i in range(numcases):
           result.append(self.scorefuncs[0].bpnet.hiddenlayer.fprop(inputs[i]))
        return result

    def apply(self,inputs):
        if type(inputs) != type([]):
            inputs = [inputs]
        numcases = len(inputs)
        result = []
        for i in range(numcases):
            result.append(self.scorefuncs[0].bpnet.fprop(inputs[i]))
        return result


class Islsl(Contrastive):
    """ Network with the structure
        input -> sigmoid -> linear -> sigmoid -> linear 
        Useful e.g. as an autoencoder. """

    def __init__(self,numin,numhid1,numhid2,numhid3,numout):
        self.numin  = numin
        self.numhid1 = numhid1
        self.numhid2 = numhid2
        self.numhid3 = numhid3
        self.numout = numout
        self.params = 0.1 * randn(scorefunc.Islsl.numparams(\
                                         numin,numhid1,numhid2,numhid3,numout))
        self.scorefuncs = [scorefunc.Islsl(\
                             numin,numhid1,numhid2,numhid3,numout,self.params)]
        Contrastive.__init__(self,normalizeacrosscliques=False)

    def posdata(self,data):
        return (((data[0],data[1]),),)

    def negdata(self,data):
        return ([],)   #no negdata: scorefunc takes care of it

    def cost(self,data,weightcost):
        if type(data)!=type([]):
            data = [data]
        numcases = len(data)
        cost = 0.0
        for i in range(numcases):
            input = data[i][0]
            desiredoutput = data[i][1]
            output = self.scorefuncs[0](input)
            if len(input.shape) >= 2:
                cost += sum((output-desiredoutput)**2)/\
                                                double(numcases*input.shape[1])
            else:
                cost += sum((output-desiredoutput)**2)/double(numcases)
        cost += 0.5 * weightcost * sum(self.params**2)
        return cost

    def hiddens(self,inputs):
        if type(inputs) != type([]):
            inputs = [inputs]
        numcases = len(inputs)
        result = []
        for i in range(numcases):
            result.append(self.scorefuncs[0].bpnet.layer1.fprop(inputs[i]))
        return result

    def apply(self,inputs):
        if type(inputs) != type([]):
            inputs = [inputs]
        numcases = len(inputs)
        result = []
        for i in range(numcases):
            result.append(self.scorefuncs[0].bpnet.fprop(inputs[i]))
        return result
  

class LinearRegression(Contrastive):
    """ Simple linear neural network that can be used, for example, to perform 
        linear regression. 
    """

    def __init__(self,numin,numout):
        self.numin  = numin
        self.numout = numout
        self.params = 0.01 * randn(numin*numout+numout)
        self.scorefunc = scorefunc.LinearRegressionScore(numin,numout,self.params)
        self.scorefuncs = [self.scorefunc]
        Contrastive.__init__(self,normalizeacrosscliques=False)

    def posdata(self,data):
        return (((data[0],data[1]),),)

    def negdata(self,data):
        return ([],)   

    def cost(self,data,weightcost):
        if type(data)!=type([]):
            data = [data]
        numcases = len(data)
        cost = 0.0
        for i in range(numcases):
            input = data[i][0]
            desiredoutput = data[i][1]
            output = self.scorefuncs[0](input)
            if len(input.shape) >= 2:
                cost += sum((output-desiredoutput)**2)/\
                                          double(numcases*input.shape[1])
            else:
                cost += sum((output-desiredoutput)**2)/double(numcases)
        cost += 0.5 * weightcost * sum(self.params**2)
        return cost

    def apply(self,inputs):
        if type(inputs) != type([]):
            inputs = [inputs]
        numcases = len(inputs)
        result = []
        for i in range(numcases):
            result.append(self.scorefuncs[0].bpnet.fprop(inputs[i]))
        return result
  

class LogisticRegression(Contrastive):

    def __init__(self,numin,numclasses):
        self.numin  = numin
        self.numclasses = numclasses
        self.params = 0.01 * randn(self.numin*self.numclasses+self.numclasses)
        self.scorefunc = logreg_score(self.numin,self.numclasses,self.params)
        self.scorefuncs = [scorefunc]
        Contrastive.__init__(self,normalizeacrosscliques=False)

    def posdata(self,data):
        return (((data[0],data[1]),),)

    def negdata(self,data):
        return ([],)   #no negdata: scorefunc takes care of it

    def cost(self,data,weightcost):
        if type(data)!=type([]):
            data = [data]
        numcases = len(data)
        cost = 0.0
        for i in range(numcases):
            input = data[i][0]
            output = data[i][1]
            modeloutput = self.scorefuncs[0](input)
            cost += sum(modeloutput[output]-logsumexp(modeloutput,0))/\
                                                double(numcases*input.shape[1])
        cost += 0.5 * weightcost * sum(self.params**2)
        return cost

    def apply(self,inputs):
        if type(inputs) != type([]):
            inputs = [inputs]
        numcases = len(inputs)
        result = []
        for i in range(numcases):
            result.append(self.scorefuncs[0].bpnet.fprop(inputs[i]))
        return result


