#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.
#
#
#
#This example file shows how to build an autoencoder network using Monte.
#The network is trained using one of the trainer objects that come with Monte.
#For training, a toy-dataset is used which consists of a two-dimensional 
#manifold embedded in three dimensions.

from numpy import *
from pylab import *
from monte.util.toydata import Scurve3d
from monte.arch.autoencoder import Autoencoder
from monte.gym import trainer
import random

#use 1000 points for training and 1000 points for testing:
N = 1000; Ntest = 1000

#generate some toy-data:
Y, Ytest, X, Xtest = Scurve3d(N,Ntest,sigma=0.1)

#build a 3-5-2-5-3-autoencoder and register it with a trainer:
model = Autoencoder(numin=3,numhid1=10,numhid2=2,numhid3=10)
mytrainer = trainer.Conjugategradients(model,20)

##plot 3d-data, if you dare:
#from matplotlib import axes3d
#fig1 = figure()
#ax = axes3d.Axes3D(fig1)
#ax.scatter3D(Y[0,:],Y[1,:],Y[2,:])

#set initial weightcost:
weightcost = 0.001

#prepare figure:
fig2 = figure(); hold(False)

#train and watch the embedding unfold:
for i in range(100):
    print i
    print 'iteration: ' + str(i) + '\n' + \
          ' current weightcost: ' +  str(weightcost) + \
          '  error on test-data: ' + \
                             str(norm(Ytest-model.apply(Ytest)[0],'fro')/Ntest)
    Xtest_ = model.hiddens(Ytest)[0]
    scatter(Xtest_[0], Xtest_[1], c=Xtest.flatten())
    batch = range(N); random.shuffle(batch)
    mytrainer.step((Y[:,batch[:300]],Y[:,batch[:300]]),weightcost)
    weightcost *= 0.9

print 'final error: ', str(norm(Ytest-model.apply(Ytest)[0], 'fro'))

