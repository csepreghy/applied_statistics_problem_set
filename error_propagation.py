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

plotify = Plotify()
r = np.random

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

def compute_B(v, T):
  h = 6.626e-34
  c = 299.7e6
  kB = 1.381e-23

  first_term = (2 * h * v ** 3) / c ** 2
  second_term = 1 / (math.exp( (h * v) / (kB * T) ) - 1)

  return first_term * second_term

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

def exercise_2_2(correlation):
  v = 0.566e15
  v_error = 0.025e15

  T = 5.50e3
  T_error = 0.29e3

  n_experiments = 1000

  # Define parameters for two random numbers (Gaussianly distributed):
  mu_v = v
  sig_v =  v_error
  mu_T = T
  sig_T =  T_error
  rho = correlation    # Correlation parameter

  # Method from class
  theta = 0.5 * np.arctan(2.0 * rho * sig_v * sig_T / (np.square(sig_v) - np.square(sig_T)))
  sigu = np.sqrt(np.abs((((sig_v*np.cos(theta)) ** 2) - (sig_T * np.sin(theta))**2 ) / ((np.cos(theta)) ** 2 - np.sin(theta)**2)))
  sigv = np.sqrt(np.abs((((sig_T*np.cos(theta)) ** 2) - (sig_v * np.sin(theta))**2 ) / ((np.cos(theta)) ** 2 - np.sin(theta)**2)))

  v = r.normal(0.0, sigu, n_experiments)
  T = r.normal(0.0, sigv, n_experiments)
  v_vals = mu_v + np.cos(theta) * v - np.sin(theta) * T
  T_vals = mu_T + np.sin(theta) * v + np.cos(theta) * T
  B_vals = np.zeros(len(v_vals))
  # v_T_vals = np.array([v_vals, T_vals])

  v_vals_std = np.std(v_vals)
  T_vals_std = np.std(T_vals)

  v_vals_mean = np.mean(v_vals)
  T_vals_mean = np.mean(T_vals)


  print(f'v_vals_mean = {v_vals_mean}')
  print(f'v = {v}')

  print(f'T_vals_mean = {T_vals_mean}')
  print(f'T = {T}')


  print(f'v_vals_std = {v_vals_std}')
  print(f'v_error = {v_error}')


  print(f'T_vals_std = {T_vals_std}')
  print(f'T_error = {T_error}')


  for i in range(len(B_vals)):
    B_vals[i] = compute_B(v_vals[i], T_vals[i])

  B_mean = np.mean(B_vals)
  B_error = np.std(B_vals)

  print(f'B_mean = {B_mean}')
  print(f'B_error = {B_error}')


def main():
  print('hello')
  # exercise_2_1()
  # exercise_2_2(0)
  # exercise_2_2(0.87)

if __name__ == '__main__':
  main()