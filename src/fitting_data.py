import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from iminuit import Minuit
from scipy import stats
from scipy.stats import binom, poisson, norm
from scipy.integrate import quad, simps
from scipy.optimize import minimize
import math

from plotify import Plotify

plotify = Plotify()

x = []

with open('data/data_GammaSpectrum.txt', 'r' ) as infile:
    for line in infile:
        line = line.strip().split()
        x.append(float(line[0]))


x = np.array(x)
x = np.multiply(np.subtract(x, subtracting_factor),  multiplying_factor)



print(f'x = {x}')


fig, ax = plotify.get_figax(figsize=(12, 7))

# peak1 = 242
# peak2 = 295
# peak3 = 352
# peak4 = 609
# peak5 = 1120

peak1 = 7.4
peak2 = 19.3

print(f'np.min(y) = {np.min(x)}')
print(f'np.max(y) = {np.max(x)}')

ax.hist(x, bins=1000, range=(np.min(x), np.max(x)), histtype='step', linewidth=1, color=plotify.c_orange)
plt.axvline(x=peak1, linestyle='--', alpha=0.5)
plt.axvline(x=peak2, linestyle='--', alpha=0.5)
plt.axvline(x=peak3, linestyle='--', alpha=0.5)
plt.axvline(x=peak4, linestyle='--', alpha=0.5)
plt.axvline(x=peak5, linestyle='--', alpha=0.5)

plt.show()


print("The number of entries in the file was: ", len(x))