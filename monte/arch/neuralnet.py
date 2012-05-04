#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from monte.models.contrastive import nn


class NeuralnetIsl(nn.Isl):
    """Simple neural network with a sigmoid hidden layer and a linear output 
    layer.
  
    To construct a network, you need to specify number of inputs, number of 
    hiddens units and number of outputs.
    """

    def __init__(self,numin,numhid,numout):
        nn.Isl.__init__(self,numin,numhid,numout)


class NeuralnetIslsl(nn.Islsl):
    """Simple neural network with three hidden layers. The innermost hidden 
       layer is linear, the other two hidden layers are sigmoid layers. 
  
    To construct a network, you need to specify number of inputs, the 
    number of units for each hidden layer and the number of outputs.
    """

    def __init__(self,numin,numhid1,numhid2,numhid3,numout):
        nn.Islsl.__init__(self,numin,numhid1,numhid2,numhid3,numout)

