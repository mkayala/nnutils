#Copyright (C) 2007-2008 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from numpy import sum


class Squarederror(object):
    """Standard squared error cost."""

    @staticmethod
    def numparams():
        return 0

    def __init__(self):
        pass

    def fprop(self, actual, desired):
        return ((actual-desired)**2).sum()

    def bprop(self, actual, desired):
        return 2 * (actual - desired)

    def grad(self, d_output, desired):
        return zeros(0, dtype=float) 

