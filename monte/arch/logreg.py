#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

"""Logistic regression.

This module contains two classes:
1) Logreg defines a (linear) logistic regression model.
2) LogregNNIsl defines a nonlinear classifier based on a backprop-network.

"""

from numpy import zeros, ones, sum, dot, newaxis, exp, log, inf, \
                  hstack, argmax, double
from pylab import randn, find
from monte.util.util import logsumexp,unhot,onehot
from monte import bp


class Logreg(object):
    """ Logistic regression. 

        Simple (linear) logistic regression. Labels are always expected 
        to be in 'onehot'-encoding. 
  
        Models can be trained using either their own train-method (which is 
        not very fast), or can make use of external optimization functions. 
        In the latter case, use the method updateparams to update the model-
        parameters.
    """

    def __init__(self,numclasses,d,regularizebiases=False):
        self.regularizebiases = regularizebiases
        self.numclasses = numclasses #number of classes
        self.numdimensions = d       #number of input dimensions
        self.params = \
                 0.01*randn(self.numclasses*self.numdimensions+self.numclasses)
        self.weights = self.params[:self.numclasses*self.numdimensions].\
                                  reshape((self.numclasses,self.numdimensions))
        self.biases = self.params[self.numclasses*self.numdimensions:]
    
    def cost(self, features, labels, weightcost):
        #labels = onehot(labels)
        scores = dot(self.weights,features) + self.biases[:,newaxis]
        if self.regularizebiases:
            negloglik = (-sum(sum(labels*scores)) + \
                       sum(logsumexp(scores,0)))/double(features.shape[1]) + \
                       weightcost * sum(sum(self.params**2))
        else:
            negloglik = (-sum(sum(labels*scores)) + \
                       sum(logsumexp(scores,0)))/double(features.shape[1])+ \
                       weightcost * sum(sum(self.weights**2))
        return negloglik 

    def grad(self, features, labels, weightcost):
        #labels = onehot(labels)
        gradw = zeros((self.numclasses,self.numdimensions), dtype=float)
        gradb = zeros(self.numclasses, dtype=float)
        scores = dot(self.weights,features) + self.biases[:,newaxis]
        #mus = exp(scores)/sum(exp(scores),0)
        mus = zeros((self.numclasses,features.shape[1]),dtype=float)
        for i in range(self.numclasses):
            m = scores-scores[i,:][newaxis,:]
            mus[i,:] = 1./sum(exp(m),0)
        for c in range(self.numclasses):
            gradw[c,:] = -sum((labels[c,:]-mus[c,:])*features,1) 
            gradb[c] = -sum(labels[c,:]-mus[c,:])
        gradw /= double(features.shape[1])
        gradb /= double(features.shape[1])
        gradw = gradw + 2*weightcost * self.weights
        if self.regularizebiases:
            gradb += 2*weightcost * self.biases
        return hstack((gradw.flatten(),gradb))

    def f(self,x,features,labels,weightcost):
        """Wrapper function around cost function to check grads, etc."""
        xold = self.params.copy()
        self.updateparams(x.copy())
        result = self.cost(features,labels,weightcost) 
        self.updateparams(xold.copy())
        return result

    def g(self,x,features,labels,weightcost):
        """Wrapper function around gradient to check grads, etc."""
        xold = self.params.copy()
        self.updateparams(x.copy())
        result = self.grad(features,labels,weightcost).flatten()
        self.updateparams(xold.copy())
        return result

    def updateparams(self,newparams):
        """ Update model parameters."""
        self.params *= 0.0
        self.params += newparams.copy()

    def probabilities(self, features):
        scores = dot(self.weights,features) + self.biases[:,newaxis]
        return exp(scores - logsumexp(scores,0))

    def classify(self, features): 
        """Use input weights to classify instances (provided columnwise 
           in matrix features.)"""
        if len(features.shape)<2:
            features = features[:,newaxis]
        N = features.shape[1]
        confidences = dot(self.weights,features)
        confidences += self.biases[:,newaxis]
        labels = zeros((self.numclasses,N))
        for i in range(N):
            labels[argmax(confidences[:,i]),i] = 1
        return labels

    def zeroone(self, features, labels):
        """ Computes the average classification error (aka. zero-one-loss) 
            for the given instances and their labels. 
        """
        return 1.0 - (self.classify(features)*labels).sum().sum()/\
                      double(features.shape[1])

    def train(self, features, labels, weightcost):
        """Train the model using gradient descent.
  
           Inputs:
           -Instances (column-wise),
           -'One-hot'-encoded labels, 
           -Scalar weightcost, specifying the amount of regularization"""
      
        numcases = features.shape[1]
        stepsize = 0.001
        gradw = zeros((self.numclasses,self.numdimensions), dtype=float)
        gradb = zeros((self.numclasses), dtype=float)
      
        confidences = dot(self.weights,features)
        #mus=exp(confidences)/repmat(sum(exp(confidences),0),self.numclasses,1)
        mus = zeros((self.numclasses,numcases),dtype=float)
        for i in range(self.numclasses):
            m = confidences-confidences[i,:][newaxis,:]
            mus[i,:] = 1./sum(exp(m),0)
        if self.regularizebiases:
            likelihood=sum(sum(labels*log(mus)))/double(numcases)-\
                                            weightcost*sum(sum(self.params**2))
        else:
            likelihood=sum(sum(labels*log(mus)))/double(numcases)-\
                                           weightcost*sum(sum(self.weights**2))
        likelihood_new = -inf
        while stepsize > 10**-6:
            print 'stepsize:' + str(stepsize)
            print likelihood
            # compute gradient:
            for c in range(self.numclasses):
                gradw[c,:] = sum((labels[c,:]-mus[c,:])*features,1) 
                gradb[c] = -sum(labels[c,:]-mus[c,:])
            gradw /= double(numcases)
            gradb /= double(numcases)
            gradw = gradw - 2*weightcost * self.weights
            if self.regularizebiases:
                gradb = gradb - 2*weightcost * self.biases
            # do a gradient step:
            weights_new = self.weights + stepsize * gradw
            b_new = self.biases + stepsize * gradb
            # re-compute log-likelihood:
            confidences = dot(weights_new,features) + b_new[:,newaxis]
            mus_new = zeros((self.numclasses,numcases),dtype=float)
            for i in range(self.numclasses):
                m = confidences-confidences[i,:][newaxis,:]
                mus_new[i,:] = 1./sum(exp(m),0)
            if self.regularizebiases:
                likelihood_new = sum(sum(labels*log(mus_new)))\
                                               /double(numcases) \
                         - weightcost*(sum(sum(weights_new**2))+sum(b_new**2))
            else:
                likelihood_new = sum(sum(labels*log(mus_new)))\
                                               /double(numcases) \
                                          - weightcost*sum(sum(weights_new**2))
            if likelihood_new > likelihood:
                stepsize = stepsize * 1.1
                likelihood = likelihood_new
                self.weights = weights_new
                self.biases = b_new
                mus = mus_new
            else:
                stepsize = stepsize*0.5


class LogregNNIsl(object):
    """ Nonlinear classifier that uses a backprop network with one sigmoid 
        hidden layer. """
    
    def __init__(self,numclasses,numin,numhid,regularizebiases=False):
        self.numin  = numin
        self.numhid = numhid
        self.regularizebiases = regularizebiases
        self.numclasses = numclasses
        self.params = 0.01 * randn(self.numin*self.numhid+self.numhid+\
                                   self.numhid*self.numclasses+self.numclasses)
        self.weightsbeforebiases = self.params[:-self.numclasses]
        self.scorefunc = bp.neuralnet.SigmoidLinear(
                                           numin,numhid,numclasses,self.params)
    
    def cost(self,features,labels,weightcost):
        labels = labels.astype(float)
        numcases = features.shape[1]
        scores = self.scorefunc.fprop(features)
        if self.regularizebiases:
            cost = (-sum(sum(labels*scores)) + \
                 sum(logsumexp(scores,0)))/double(numcases) + \
                 weightcost * sum(sum(self.params**2))
        else:
            cost = (-sum(sum(labels*scores)) + \
                 sum(logsumexp(scores,0)))/double(numcases) + \
                 weightcost * sum(sum(self.weightsbeforebiases**2))
        return cost

    def grad(self, features, labels, weightcost):
        scores = self.scorefunc.fprop(features)
        modeloutput = zeros((self.numclasses,features.shape[1]),dtype=float)
        for i in range(self.numclasses):
            m = scores-scores[i,:][newaxis,:]
            modeloutput[i,:] = 1./sum(exp(m),0)
        d_outputs = modeloutput - labels
        self.scorefunc.bprop(d_outputs,features)
        grad = self.scorefunc.grad(d_outputs,features)
        grad = sum(grad,1) / double(features.shape[1])
        if self.regularizebiases:
            grad += 2*weightcost * self.params
        else:
            grad[:-self.numclasses] += 2*weightcost * self.weightsbeforebiases
        return grad

    def probabilities(self, features):
        scores = self.scorefunc.fprop(features)
        return exp(scores - logsumexp(scores,0))

    def classify(self, features): 
        if len(features.shape)<2:
            features = features[:,newaxis]
        numcases = features.shape[1]
        confidences = self.scorefunc.fprop(features)
        labels = zeros((self.numclasses,numcases),dtype=int)
        for i in range(numcases):
            labels[argmax(confidences[:,i]),i] = 1
        return labels

    def zeroone(self, features, labels):
        """ Computes the average classification error (aka. zero-one-loss) 
            for the given instances and their labels. 
        """
        return 1.0 - (self.classify(features)*labels).sum().sum()/\
                      double(features.shape[1])

    def f(self,x,features,labels,weightcost):
        """Wrapper function around cost function to check grads, etc."""
        xold = self.params.copy()
        self.updateparams(x.copy())
        result = self.cost(features,labels,weightcost) 
        self.updateparams(xold.copy())
        return result

    def g(self,x,features,labels,weightcost):
        """Wrapper function around gradient to check grads, etc."""
        xold = self.params.copy()
        self.updateparams(x.copy())
        result = self.grad(features,labels,weightcost).flatten()
        self.updateparams(xold.copy())
        return result

    def updateparams(self,newparams):
        """ Update model parameters."""
        self.params *= 0.0
        self.params += newparams.copy()


if __name__ == "__main__":
    #make some random toy-data:
    traininputs = hstack((randn(2,100)-1.0,randn(2,100)+1.0))
    trainlabels = onehot(hstack((ones((100))*0,ones((100))*1))).T
    testinputs = hstack((randn(2,100)-1.0,randn(2,100)+1.0))
    testlabels = onehot(hstack((ones((100))*0,ones((100))*1))).T
    testinputs = hstack((randn(2,100)-1.0,randn(2,100)+1.0))
    #build and train a model:
    model = Logreg(2,2)
    model.train(traininputs,trainlabels,0.001)
    #or use a trainer object:
    #from monte.gym import trainer 
    #trainer = trainer.Conjugategradients(model,20)
    #trainer.step(traininputs,trainlabels,0.001)
    #try model on test data:
    predictedlabels = model.classify(testinputs) 
    print 'true labels: '
    print unhot(testlabels.T)
    print 'predicted labels: '
    print unhot(predictedlabels.T)
    print 'error rate: '
    print sum(unhot(testlabels.T)!=unhot(predictedlabels.T))/\
                                                     float(testinputs.shape[1])

