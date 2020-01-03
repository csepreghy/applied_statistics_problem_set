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

