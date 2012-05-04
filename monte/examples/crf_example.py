#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.
#
#
#
#This example file shows how to build a simple model using monte, how to 
#train the model using the provided trainers, and how to apply the trained 
#model to some data.

from numpy import array
from pylab import randn
from monte.arch.crf import ChainCrfLinear #all ready-to-use models reside in monte.arch
from monte.gym import trainer #everything to do with training resides in 
                              #monte.gym

#Make a linear-chain CRF:
mycrf = ChainCrfLinear(3,2)

#Make a trainer (that does 5 steps per call), and register mycrf with it:
mytrainer = trainer.Conjugategradients(mycrf,5)

#Alternatively, we could have used one of these, for example:
#mytrainer = trainer.OnlinegradientNocost(mycrf,0.95,0.01)
#mytrainer = trainer.Bolddriver(mycrf,0.01)
#mytrainer = trainer.GradientdescentMomentum(mycrf,0.95,0.01)

#Produce some stupid toy data for training:
inputs = randn(10,3)
outputs = array([0,1,1,0,0,0,1,0,1,0])

#Train the model. Since we have registered our model with this trainer, 
#calling the trainers step-method trains our model (for a number of steps):
for i in range(20):
    mytrainer.step((inputs,outputs),0.001)
    print mycrf.cost((inputs,outputs),0.001)

#Apply to some stupid test data:
testinputs = randn(15,3)
predictions = mycrf.viterbi(testinputs)
print predictions

