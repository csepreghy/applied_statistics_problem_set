import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
import sys
from scipy import stats
from scipy.stats import binom, poisson, norm
from scipy.integrate import quad, simps
from scipy.optimize import minimize
import math

from plotify import Plotify
from ExternalFunctions import nice_string_output, add_text_to_ax, Chi2Regression, integrate1d

# plotify is a skin over matplotlib I wrote to automate styling of plots
# https://github.com/csepreghy/plotify
plotify = Plotify()
r = np.random

def func_binomial_pmf(x, n, p):
    return binom.pmf(np.floor(x+0.5), n, p)


def binomial_integral(bounds, n, p):
    list_of_ps = []

    for i in range(bounds[0], bounds[1]):
        current_p = func_binomial_pmf(i, n, p)
        list_of_ps.append(current_p)

    sum_over_ps = sum(list_of_ps)
    return sum_over_ps


# ------------------------------------------ #
# -------------- Exercise 1.3 -------------- #
# ------------------------------------------ #

# What is the lambda that will give 19 "delay days" 
# out of 356 days with 7 or more delays each?

def func_poisson_pmf(x, lamb):
    # return poisson.cdf(np.floor(x+0.5), lamb)
    return poisson.pmf(np.floor(x+0.5), lamb)

def func_gaussian_pdf(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

def integrate_poisson(lamb):
    days_list = []

    for x in range(0, 8):
        n_day_with_x_delays = func_poisson_pmf(x, lamb) * 365
        days_list.append(n_day_with_x_delays)

    cumulative_days = sum(days_list)
    return 365 - cumulative_days

def cumulative_days_func(lamb):
    desired_value = 19
    cumulative_days = integrate_poisson(lamb)
    error = np.abs(desired_value - cumulative_days)

    return error

# ------------------------------------------ #
# ------------- Exercise 1.1.1 ------------- #
# ------------------------------------------ #
def exercise_1_1_1():
    xmin = 0
    xmax = 50

    n = 50
    p = 18/37

    x = np.linspace(xmin, xmax, 1000)
    y = func_binomial_pmf(x, n, p)

    plotify.plot(x=x,
                 y=y,
                 title="Binomial PDF of Little Pete's Winning Probabilities",
                 xlabel="Number of Successes",
                 ylabel="P",
                 show_plot=False,
                 save=False)

    x_from_26 = np.linspace(26, xmax, 1000)
    y_from_26 = func_binomial_pmf(x_from_26, n, p)
    plt.fill_between(x_from_26, y_from_26, alpha=0.3, color=plotify.c_orange)
    plt.legend({'P of winning at least 26 times', 'Binomial Distribution'}, facecolor="#282D33")
    plt.savefig(('plots/' + 'casino_binomial'), facecolor=plotify.background_color, dpi=180)
    plt.show()

    pete_wins_25 = func_binomial_pmf(25, n, p)
    print(f'pete_wins_25 = {pete_wins_25}')

    sum_over_ps = binomial_integral(bounds=(26, 50),
                                n=50,
                                p=18/37)

    print(f'sum_over_ps = {sum_over_ps}')

    I = quad(func_binomial_pmf, 26, 50, args=(n,p), epsabs=100)
    print(f'integral of binomial between 26 and 50 is: {I[0]}')

# ------------------------------------------ #
# ------------- Exercise 1.1.2 ------------- #
# ------------------------------------------ #

def exercise_1_1_2():
    x = 20

    # compute for which n (number of trials) will Pete have 
    # at least p = 0.95 to have 20 or more successes

    n = 20 # you need at least 20 trials to win 20 or more
    success_p = 0.95
    trial_p = 0

    while trial_p <= success_p:
        trial_p = binomial_integral(bounds=(20, n), n=n, p=18/37)
        if trial_p <= success_p: n += 1

    print(f'It requires {n} trials to reach 20 or more successes with p = {trial_p}')

# ------------------------------------------ #
# -------------- Exercise 1.2 -------------- #
# ------------------------------------------ #
def exercise_1_2():
    x = np.linspace(-5, 5, 1000)
    y = func_gaussian_pdf(x, 0, 1)

    I_gaussian = quad(func_gaussian_pdf, 1.25, 2.5, args=(0, 1), epsabs=200)
    gaussian_result = I_gaussian[0] * 2
    print(f'gaussian_result = {gaussian_result}')

    plotify.plot(x,
                y,
                show_plot=False,
                title='Gaussian with values between 1.25 and 2.5 σs aways from the mean',
                xlabel="",
                ylabel="",
                tickfrequencyone=True)

    x_fill_1 = np.linspace(-2.5, -1.25, 1000)
    y_fill_1 = func_gaussian_pdf(x_fill_1, 0, 1)
    x_fill_2 = np.linspace(1.25, 2.5, 1000)
    y_fill_2 = func_gaussian_pdf(x_fill_2, 0, 1)
    plt.fill_between(x_fill_1, y_fill_1, alpha=0.5, color=plotify.c_orange)
    plt.fill_between(x_fill_2, y_fill_2, alpha=0.5, color=plotify.c_orange)
    plt.show()

def exercise_1_3():
    x0 = 4
    # I know Iminuit is better but I'm used to scipy
    res = minimize(cumulative_days_func, x0, method='Nelder-Mead', tol=1e-9)
    Lambda = res.x
    print(f'res.x = {res.x}')

    result = integrate_poisson (Lambda)
    print(f'result = {result}') 

    xvals = np.linspace(0, 20, 1000)
    y = func_poisson_pmf(xvals, Lambda) * 365 # to have the integral equal 365 instead of 1, which will make


    plotify.plot(xvals, y, tickfrequencyone=True, show_plot=False)

    x_from_8 = np.linspace(8, 20, 1000)
    y_from_8 = func_poisson_pmf(x_from_8, Lambda) * 365

    plt.fill_between(x_from_8, y_from_8, alpha=0.5, color=plotify.c_orange)
    plt.show()


def main():
    exercise_1_1_1()
    exercise_1_1_2()
    exercise_1_2()
    exercise_1_3()

if __name__ == '__main__':
    main()
