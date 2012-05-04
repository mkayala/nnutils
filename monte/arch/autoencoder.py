#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from monte.models.contrastive import nn

class Autoencoder(nn.Islsl):
    def __init__(self,numin,numhid1,numhid2,numhid3):
        nn.Islsl.__init__(self,numin,numhid1,numhid2,numhid3,numin)
  
