import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import math

from scipy.integrate import quad, simps
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr, chisquare, norm

from distributions_probability import func_gaussian_pdf
from plotify import Plotify

plotify = Plotify()
r = np.random


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
print(f'C = {C}')

for i in range(len(xvals)):
  yvals[i] = pdf(xvals[i], C, a)

value_squares = yvals ** 2
rms = np.sqrt(np.sum(value_squares) / len(value_squares))
y_mean = np.mean(yvals)
print(f'y_mean = {y_mean}')

n_points = 1000
n_bins = 10
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

    if f(x, C, a) > y:
      x_accepted.append(x)
    
  return x_accepted, hello_x, hello_y

accepted_x_vals, hello_x, hello_y = von_neumann(pdf, xmin, xmax, ymin, ymax, N_points=n_points, f_arg=C)

xvals = np.linspace(0, 2, n_points)
yvals = np.zeros(len(xvals))

for i in range(len(xvals)):
  yvals[i] = pdf_to_plot(xvals[i], C, xmin, xmax, n_bins, n_points)

fig, ax = plotify.get_figax()

ax.plot(xvals, yvals, c=plotify.c_blue)
ax.hist(accepted_x_vals, bins=n_bins, range=(0, 2), histtype='step', label='histogram', color=plotify.c_orange, linewidth=2)
ax.set_xlabel("Randomly Sampled Value")
ax.set_ylabel("Number Of Sampled Values")
ax.set_title("Sampling Values According f(x)")
ax.legend({'Number of Sampled Values', 'Probability Distribution'}, facecolor="#282D33", loc="upper left")


plt.savefig(('plots/' + 'monte_carlo'), facecolor=plotify.background_color, dpi=180)

# plt.show()


def pdf_to_minimize(a):
  sample_values = accepted_x_vals
  expected_values = np.zeros(len(sample_values))
  for i in range(len(sample_values)):
    expected_values[i] = pdf(sample_values[i], C, a)

  chi2_value, chi2_pval = chisquare(sample_values, expected_values)

  return chi2_value

x0 = 2
res = minimize(pdf_to_minimize, x0, method='Nelder-Mead', tol=1e-9)

# print(f'res.x = {res.x}')

sample_values = accepted_x_vals
expected_values = np.zeros(len(sample_values))
for i in range(len(sample_values)):
  expected_values[i] = pdf(sample_values[i], C, 2)

chi2_value, chi2_pval = chisquare(sample_values, expected_values)
# print(f'chi2_value = {chi2_value}')
# print(f'chi2_pval = {chi2_pval}')

all_accepted_x_vals = []

for i in range(1000):
  accepted_x_vals, _, _ = von_neumann(pdf, xmin, xmax, ymin, ymax, N_points=5, f_arg=C)
  all_accepted_x_vals.append(accepted_x_vals)


hell = len(accepted_x_vals)
print(f'hell = {hell}')


# def chi2(xvals, yvals, n_parameters):
#   expected_values = np.zeros
#   std = np.std(values)
#   ndof = len(values) - n_parameters
#   chi2_value = 0

#   if len(uncertainties.shape) == 0:
#     uncertainties = [uncertainties] * len(values)
  
#   for observed_value, uncertainty in zip(values, uncertainties):
#     chi2_value += (observed_value - mean)**2 / uncertainty**2

#   p_chi2 = stats.chi2.sf(chi2_value, ndof)

#   return chi2_value, p_chi2

# res = minimize(cumulative_days_func, x0, method='Nelder-Mead', tol=1e-9)

binwidth = (xmax - xmin) / n_bins
y, bin_edges = np.histogram(accepted_x_vals, bins=n_bins, range=(xmin, xmax))
x = 0.5*(bin_edges[1:] + bin_edges[:-1])
sy = np.sqrt(y)      # This is the standard for histograms - bin entries are Poisson distributed!

fig2, ax2 = plotify.get_figax()

hist_data = ax2.errorbar(x, y, sy, fmt='.', linewidth=2, label="Data")
ax2.set(xlabel="x values (generated)", ylabel = "Frequency / 0.01", title = "Distribution of x values")

# Plot fit result on top of histograms:
x_ulfit = np.linspace(xmin, xmax, 1000) # Create the x-axis for the plot of the fitted function
y_ulfit = np.zeros(len(x_ulfit)) 

for i in range(len(x_ulfit)):
  y_ulfit[i] = pdf_to_plot(x_ulfit[i], C, xmin, xmax, n_bins, n_points)



ax2.plot(x_ulfit, y_ulfit, '--', color='white', linewidth=2, label='Fit (unbinned LLH)')
# plt.show()

print(f'rms = {rms}')




# plotify.plot(xvals, yvals)
