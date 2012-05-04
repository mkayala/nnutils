#Copyright (C) 2007-2008 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from pylab import axis, plot, hold

def showcost(model,*args):
    print model.cost(*args)

class plotcost(object):
    def __init__(self,maxticks = 50000):
        self.maxticks = maxticks
        self.costs = []
        axis()
        hold(False)
    def __call__(self,model, *args):
        self.costs.append(model.cost(*args))
        plot(self.costs)
        if len(self.costs) > self.maxticks:
            self.costs = []

