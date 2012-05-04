#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.
#
#
#
#This example file shows how to build and train a sparse linear regression 
#model (Tibshirani's Lasso).


from numpy import array, dot, newaxis
from pylab import randn, plot, clf, show, xlabel, ylabel, legend
from monte.arch import lasso 

#Number of data cases:
N = 100

#Some random 10-dimensional inputs:
x = randn(10,N)

#Define a sparse linear regression model
M = array([2.0,-2.0,0.0,0.0,3.0,1.0,0.0,-5.0,-3.0,1.0])[newaxis,:]
y = dot(M,x) + randn(1,N)

#Make and train a (10-dimensional) Lasso model:
l = lasso.Lasso(10)
l.train(x,y,15.0)

#Plot:
clf()
plot(abs(M), label='truth')
plot(abs(l.getparams()[0]), label='model')
xlabel('input dimension')
ylabel('absolut value of regression coefficient')
legend()
show()

