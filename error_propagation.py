import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from iminuit import Minuit
import sys
from scipy import stats
from scipy.stats import binom, poisson, norm
from scipy.integrate import quad, simps
from scipy.optimize import minimize
import math

from distributions_probability import func_gaussian_pdf
from plotify import Plotify
from ExternalFunctions import nice_string_output, add_text_to_ax, Chi2Regression, integrate1d

plotify = Plotify()

def chi2(values, uncertainties, n_parameters):
  mean = np.mean(values)
  std = np.std(values)
  ndof = len(values) - n_parameters
  chi2_value = 0

  if len(uncertainties.shape) == 0:
    uncertainties = [uncertainties] * len(values)
  
  for observed_value, uncertainty in zip(values, uncertainties):
    chi2_value += (observed_value - mean)**2 / uncertainty**2

  p_chi2 = stats.chi2.sf(chi2_value, ndof)

  return chi2_value, p_chi2

def exercise_2_1():    
  n_parameters_A = 1
  values_A = np.array([2.05, 2.61, 2.46, 2.48])
  uncertainties_A = np.array([0.11, 0.10, 0.13, 0.12])

  chi2_value_A, p_chi2_A = chi2(values_A, uncertainties_A, n_parameters_A) 

  print(f'chi2_value_A = {chi2_value_A}')
  print(f'p_chi2_A = {p_chi2_A}')

  x_A = np.array([1, 2, 3, 4])
  fig, ax = plotify.get_figax(figsize=(8,6), use_grid=False)

  ax.scatter(x_A, values_A, c=plotify.c_orange)
  ax.errorbar(x_A, values_A, yerr=uncertainties_A, fmt='o', c=plotify.c_orange)
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.set_title("Measurements With Uncertainties")
  ax.set_xlabel("Measurement Number")
  ax.set_ylabel("Value / Uncertainty")
  plt.show()


  values_B = np.array([2.69,2.71,2.56,2.48,2.34,2.79,2.54,2.68,2.69,2.58,2.66,2.70])
  values_mean_B = np.mean(values_B)
  uncertainty_B = np.std(values_B)
  n_parameters_B = 1

  print(f'uncertainty_B.shape = {uncertainty_B.shape}')

  chi2_value_B, p_chi2_B = chi2(values_B, uncertainty_B, n_parameters_B) 

  print(f'chi2_value_B = {chi2_value_B}')
  print(f'p_chi2_B = {p_chi2_B}')

  x_B = np.linspace(2.1, 3.1, 1000)
  y_B = func_gaussian_pdf(x_B, values_mean_B, uncertainty_B)

  fig, ax = plotify.get_figax(figsize=(8,6))

  ax.plot(x_B, y_B, c=plotify.c_orange)
  ax.scatter(x=values_B, 
            y=[0] * len(values_B),
            c=plotify.c_orange)
  ax.set_title("Measurements Without Uncertainties")
  ax.set_xlabel("Values")

  plt.show()

def compute_B(v, T):
  h = 6.626e-34
  c = 299.7e6
  kB = 1.381e-23

  first_term = (2 * h * v ** 3) / c ** 2
  second_term = 1 / (math.exp( (h * v) / (kB * T) ) - 1)

  return first_term * second_term

v = 0.566e15
T = 5.50e3
print(f'v = {v}')
print(f'T = {T}')

B = compute_B(v, T)
print(f'B = {B}')


def main():
  print('hello')
  # exercise_2_1()

if __name__ == '__main__':
  main()