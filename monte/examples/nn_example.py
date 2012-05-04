#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.
#
#
#
from numpy import arange, newaxis, sin
from pylab import randn, plot, scatter, hold
from monte.arch.neuralnet import NeuralnetIsl
from monte.gym import trainer

nn       = NeuralnetIsl(1,10,1)     #neural network with one hidden layer
#nntrainer  = trainer.Bolddriver(nn)
nntrainer  = trainer.Conjugategradients(nn,10)

inputs = arange(-10.0,10.0,0.1)[newaxis,:]
outputs = sin(inputs) + randn(1,inputs.shape[1])
testinputs  = arange(-10.5,10.5,0.05)[newaxis,:]
testoutputs = sin(testinputs)

for i in range(50):
    hold(False)
    scatter(inputs[0],outputs[0])
    hold(True)
    plot(testinputs[0],nn.apply(testinputs)[0][0])
    nntrainer.step((inputs,outputs),0.0001)
    #savefig('%02d' % i)
    print nn.cost((inputs,outputs),0.0001)

