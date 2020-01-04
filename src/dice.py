import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import math

from scipy.integrate import quad, simps
from scipy.optimize import minimize
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from minepy import MINE

from distributions_probability import func_gaussian_pdf, func_binomial_pmf
from read_ufo_sightings import read_ufo_sightings

from plotify import Plotify

plotify = Plotify()

xvals = np.linspace(0, 12, 13)
occurrences = np.array([185, 1149, 3265, 5475, 6114, 5194, 3067, 1331, 403, 105, 14, 4, 0])

fig, ax = plotify.get_figax

xvals = np.linspace(xmin, xmax, 1000)
yvals = func_binomial_pmf(x, n, p)
ax.scatter(xvals, occurrences)

