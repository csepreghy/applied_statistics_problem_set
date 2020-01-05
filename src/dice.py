import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import math

from scipy.integrate import quad, simps
from scipy.optimize import minimize
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chisquare, norm
from minepy import MINE

from distributions_probability import func_gaussian_pdf, func_binomial_pmf, func_poisson_pmf
from read_ufo_sightings import read_ufo_sightings

from plotify import Plotify

plotify = Plotify()

xvals = np.linspace(0, 12, 13)
xmin = 0
xmax = 12

observed_vals = np.array([185, 1149, 3265, 5475, 6114, 5194, 3067, 1331, 403, 105, 14, 4, 0])
observed_vals_mean = np.mean(observed_vals)
observed_vals_std = np.std(observed_vals)
n_throws = np.sum(observed_vals)
expected_vals = np.zeros(len(observed_vals))

p = 1/3

for i in range(len(observed_vals)):
  expected_vals[i] = n_throws * func_binomial_pmf(i, 12, p)

print(f'observed_vals = {observed_vals}')
print(f'expected_vals = {expected_vals}')

chi2, chi2_pval = chisquare(observed_vals, expected_vals)
print(f'chi2 = {chi2}')
print(f'chi2_pval = {chi2_pval}')

fig, ax = plotify.get_figax()

x = np.linspace(xmin, xmax, 1000)
yvals_binomial = n_throws * func_binomial_pmf(x, 12, 1/3)
yvals_poisson = n_throws * func_poisson_pmf(x, 4)
yvals_gaussian = n_throws * func_gaussian_pdf(x, 4, 1.75)
ax.scatter(xvals, observed_vals)
ax.plot(x, yvals_binomial)
ax.plot(x, yvals_poisson)
ax.plot(x, yvals_gaussian)

plt.show()

def chi2_gauss(sigma):
  expected_gauss_vals = np.zeros(len(observed_vals))
  for i in range(len(observed_vals)):
    expected_gauss_vals[i] = n_throws * func_gaussian_pdf(i, 4, sigma)

  chi2, chi2_pval = chisquare(observed_vals, expected_gauss_vals)

  return chi2

def chi2_poisson(Lambda):
  expected_poisson_vals = np.zeros(len(observed_vals))
  for i in range(len(observed_vals)):
    expected_poisson_vals[i] = n_throws * func_poisson_pmf(i, Lambda)

  chi2, chi2_pval = chisquare(observed_vals, expected_poisson_vals)

  return chi2

def chi2_binomial(p):
  expected_binomial_vals = np.zeros(len(observed_vals))
  for i in range(len(observed_vals)):
    expected_binomial_vals[i] = n_throws * func_binomial_pmf(i, 12, p)

  chi2, chi2_pval = chisquare(observed_vals, expected_binomial_vals)

  return chi2

# ------------------------------------------ #
# ---------------- Gaussian ---------------- #
# ------------------------------------------ #

x0 = 1.6
res_gauss = minimize(chi2_gauss, x0, method='Nelder-Mead', tol=1e-9)
print(f'res_gauss gauss x = {res_gauss.x}')
sigma = res_gauss.x

expected_gaussian_vals = np.zeros(len(observed_vals))
for i in range(len(observed_vals)):
  expected_gaussian_vals[i] = n_throws * func_gaussian_pdf(i, 4, sigma)

chi2_value_gaussian, chi2_pval_gaussian = chisquare(observed_vals, expected_gaussian_vals)
print(f'chi2_value_gaussian = {chi2_value_gaussian}')
print(f'chi2_pval_gaussian = {chi2_pval_gaussian} \n\n')

# ------------------------------------------ #
# ---------------- Poisson ----------------- #
# ------------------------------------------ #

x0 = 4
res_poisson = minimize(chi2_poisson, x0, method='Nelder-Mead', tol=1e-9)
print(f'res_poisson poisson x = {res_poisson.x}')
Lambda = res_poisson.x

expected_poisson_vals = np.zeros(len(observed_vals))
for i in range(len(observed_vals)):
  expected_poisson_vals[i] = n_throws * func_poisson_pmf(i, Lambda)

chi2_value_poisson, chi2_pval_poisson = chisquare(observed_vals, expected_poisson_vals)
print(f'chi2_value_poisson = {chi2_value_poisson}')
print(f'chi2_pval_poisson = {chi2_pval_poisson} \n\n')

# ------------------------------------------ #
# ---------------- Binomial ---------------- #
# ------------------------------------------ #

x0 = 1/3
res_binomial = minimize(chi2_binomial, x0, method='Nelder-Mead', tol=1e-9)
p = res_binomial.x

expected_binomial_vals = np.zeros(len(observed_vals))
for i in range(len(observed_vals)):
  expected_binomial_vals[i] = n_throws * func_binomial_pmf(i, 12, p)

chi2_value_binomial, chi2_pval_binomial = chisquare(observed_vals, expected_binomial_vals)
print(f'chi2_value_binomial = {chi2_value_binomial}')
print(f'chi2_pval_binomial = {chi2_pval_binomial}')

fig2, ax2 = plotify.get_figax()
yvals_binomial = n_throws * func_binomial_pmf(x, 12, p)
yvals_poisson = n_throws * func_poisson_pmf(x, Lambda)
yvals_gaussian = n_throws * func_gaussian_pdf(x, 4, sigma)
ax2.scatter(xvals, observed_vals)
ax2.plot(x, yvals_binomial)
ax2.plot(x, yvals_poisson)
ax2.plot(x, yvals_gaussian)

plt.show()

# def compute_chi2(p):
#   ndof = 6
#   chi2_value = 0

#   for index, observed in enumerate(observed_vals):
#     expected = n_throws * func_binomial_pmf(index, 12, p)
#     print(f'expected = {expected}')
#     chi2_value += (observed - expected) ** 2 / observed_vals_std ** 2

#   p_chi2 = stats.chi2.sf(chi2_value, ndof)
#   print(f'p_chi2 = {p_chi2}')


