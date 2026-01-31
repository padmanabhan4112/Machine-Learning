# Python code to fit data points using a straight line

import numpy as np
import matplotlib.pyplot as plt

N = 50
x = np.random.rand(N)                           #array size = 50 , values = 0 to 1 , Uniform distribution
a = 2.5                                         # true parameter
b = 1.3                                         # true parameter
y = a*x + b + 0.2*np.random.randn(N)            # Synthesize training data , Standard Normal (Gaussian)
X = np.column_stack((x, np.ones(N)))            # construct the X matrix
theta = np.linalg.lstsq(X, y, rcond=None)[0]    # solve y = X theta , rcond handles multi-collinearity problem
t = np.linspace(0,1,200)                        # interpolate and plot
yhat = theta[0]*t + theta[1]
plt.plot(x,y,'o')
plt.plot(t,yhat,'r',linewidth=4)
plt.show()


