#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from monte.models.contrastive import chainmodel


class ChainCrfLinear(chainmodel.ChainmodelLinear):
    """ A linear chain crf. To construct a model, specify the number of input 
    dimensions and the number of classes."""
    pass


class ChainCrfNNIsl(chainmodel.ChainmodelBackprop):
    """ A simple chain crf with nonlinear observation potentials. Observation 
    potentials are defined as a backprop-network with one hidden layer. 
    To construct a model, specify the number of inputs, the number of hidden 
    units to use in the backprop-net, and the number of classes."""
    pass

