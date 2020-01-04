import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import math

from scipy.integrate import quad, simps
from scipy.optimize import minimize

from distributions_probability import func_gaussian_pdf
from plotify import Plotify

plotify = Plotify()
r = np.random
r.seed(42)

def pdf(x, C, a):
  return C * (1 - math.exp(-a * x))

def pdf_to_plot(x, C, xmin, xmax, n_bins, n_points):
  k = (xmax - xmin) / n_bins
  N = n_points * k
  return N * C * (1 - math.exp(-a * x))


a = 2
xvals = np.linspace(0, 2, 1000)
yvals = np.zeros(len(xvals))

def integrate_pdf(C):
  integral_C = quad(pdf, 0, 2, args=(C, a), epsabs=200) 
  return np.abs(integral_C[0] - 1)

x0 = 0.66
result = minimize(integrate_pdf, x0, method='Nelder-Mead', tol=1e-10)
C = result.x

for i in range(len(xvals)):
  yvals[i] = pdf(xvals[i], C, a)

value_squares = yvals ** 2
rms = np.sqrt(np.sum(value_squares) / len(value_squares))

n_points = 20000
n_bins = 100
xmin = 0
xmax = 2
ymin = 0
ymax = pdf(2, C, a)

def von_neumann(f, xmin, xmax, ymin, ymax, N_points, f_arg=()):
  x_accepted = []
  hello_x = []
  hello_y = []
  while len(x_accepted) < n_points:
    x = r.uniform(xmin, xmax)  
    y = r.uniform(ymin, ymax)

    hello_x.append(x)
    hello_y.append(y)

    if f(x, C) > y:
      x_accepted.append(x)
    
  return x_accepted, hello_x, hello_y



accepted_x_vals, hello_x, hello_y = von_neumann(pdf, xmin, xmax, ymin, ymax, N_points=n_points, f_arg=C)

xvals = np.linspace(0, 2, n_points)
yvals = np.zeros(len(xvals))

for i in range(len(xvals)):
  yvals[i] = pdf_to_plot(xvals[i], C, xmin, xmax, n_bins, n_points)

fig, ax = plotify.get_figax()

ax.plot(xvals, yvals, c=plotify.c_blue)
ax.hist(accepted_x_vals, bins=n_bins, range=(0, 2), histtype='step', label='histogram', color=plotify.c_orange)


plt.show()

print(f'rms = {rms}')




# plotify.plot(xvals, yvals)
