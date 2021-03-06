import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

import math

from scipy.integrate import quad, simps
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chisquare

from minepy import MINE

from sklearn.feature_selection import chi2 as skchi2

from distributions_probability import func_gaussian_pdf
from read_ufo_sightings import read_ufo_sightings

from plotify import Plotify

plotify = Plotify()
r = np.random

date, HourInDay, DayInYear, DayOfWeek, USstate, UScoast, Shape, DurationInSec = read_ufo_sightings()

west_indeces = np.argwhere(UScoast == 1)
print(len(west_indeces))

east_indeces = np.argwhere(UScoast == 2)
print(len(east_indeces))


def exercise_4_1_1():
  print(f'DurationInSec = {DurationInSec}')

  xmin = np.min(DurationInSec)
  xmax = np.max(DurationInSec)
  n_bins = 100

  fig, ax1 = plotify.get_figax()
  ax1.hist(DurationInSec, bins=n_bins, range=(xmin, xmax), histtype='bar', linewidth=2, color=plotify.c_orange)
  ax1.set_title("UFO Sighting Durations (in seconds)")
  ax1.set_xlabel("Reported UFO Sightings in seconds")
  ax1.set_ylabel("Number of observations in a bin")

  plt.savefig(('plots/' + 'ufo'), facecolor=plotify.background_color, dpi=180)
  plt.show()

  mean_duration = np.mean(DurationInSec)
  print(f'mean_duration = {mean_duration}')

  median_duration = np.median(DurationInSec)
  print(f'median_duration = {median_duration}')

  west_duration = DurationInSec[west_indeces]
  east_duration = DurationInSec[east_indeces]

  # east_duration = (len(east_duration) / len(west_duration)) * east_duration

  n_bins = 50

  xmin_west = np.min(west_duration)
  xmax_west = np.max(west_duration)

  print(f'xmin_west = {xmax_west}')

  fig2, ax2 = plotify.get_figax()
  plt.hist(west_duration, bins=n_bins, range=(xmin_west, xmax_west), histtype='step', label='West Durations', color=plotify.c_blue)

  # Plot fit result on top of histograms:
  x_west = np.linspace(xmin_west, xmax_west, 1000) # Create the x-axis for the plot of the fitted function
  y_west = np.zeros(len(x_west))

  xmin_east = np.min(east_duration)
  xmax_east = np.max(east_duration)

  ax2.hist(east_duration, bins=n_bins, range=(xmin_east, xmax_east), histtype='step', label='East Durations', color=plotify.c_orange)
  ax2.legend(facecolor="#282D33")

  ax2.set_title("UFO Sighting Durations in  the East and West coast")
  ax2.set_xlabel("Reported UFO Sightings in seconds")
  ax2.set_ylabel("Number of observations in a bin")

  plt.savefig(('plots/' + 'ufo_east_west'), facecolor=plotify.background_color, dpi=180)
  plt.show()

  west_duration = west_duration.flatten()
  east_duration = east_duration.flatten()

  print(f'west_duration = {west_duration}')
  ks_statistic = stats.ks_2samp(west_duration, east_duration)
  print(f'ks_statistic = {ks_statistic}')

def exercise_4_1_2():
  print(f'DayInYear = {DayInYear}')
  print(f'HourInDay = {HourInDay}')

  fig, ax = plotify.get_figax()
  ax.scatter(DayInYear, HourInDay, s=0.25)
  plt.show()

  covariance = np.cov(DayInYear, HourInDay)

  pearson_correlation, _ = pearsonr(DayInYear, HourInDay)
  print('Pearsons correlation: %.3f' % pearson_correlation)

  spearman_correlation, _ = spearmanr(DayInYear, HourInDay)
  print('Spearman correlation: %.3f' % spearman_correlation)

west_day_of_week = DayOfWeek[west_indeces]

# plt.show()

mondays = np.count_nonzero(west_day_of_week==0)
tuesdays = np.count_nonzero(west_day_of_week==1)
wednesdays = np.count_nonzero(west_day_of_week==2)
thursdays = np.count_nonzero(west_day_of_week==3)
fridays = np.count_nonzero(west_day_of_week==4)
saturdays = np.count_nonzero(west_day_of_week==5)
sundays = np.count_nonzero(west_day_of_week==6)

alldays = [mondays, tuesdays, wednesdays, thursdays, fridays, saturdays, sundays]

alldays_mean = np.mean(alldays)
alldays_std = np.std(alldays)
ndof = 6

allday_means = np.array([alldays_mean] * len(alldays))

chi2_value, chi2_p = chisquare(alldays)
print(f'chi2_value all week = {chi2_value}')
print(f'chi2_p all week = {chi2_p}')


n_bins = 7

xmin_west = np.min(0)
xmax_west = np.max(7)
xvals = np.linspace(0.5, 6.5, 7)

fig, ax = plotify.get_figax()
ax.hist(west_day_of_week, bins=n_bins, range=(xmin_west, xmax_west), histtype='step', linewidth=2, color=plotify.c_orange)
ax.errorbar(xvals, alldays, yerr=alldays_std, fmt='o', c=plotify.c_blue)
offset = matplotlib.transforms.ScaledTranslation(0.4, 0, fig.dpi_scale_trans)
plt.xticks(range(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], size='small')
for label in ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

ax.set_xlabel("Day")
ax.set_ylabel("Number of observations")
ax.set_title("Number of observations throughout the week")

plt.savefig(('plots/' + 'ufo_week'), facecolor=plotify.background_color, dpi=180)
plt.show()


mon_thu = [mondays, tuesdays, wednesdays, thursdays]

mon_thu_mean = np.mean(mon_thu)
mon_thu_std = np.std(mon_thu)

chi2_value, chi2_p = chisquare(mon_thu)
print(f'chi2_value half week = {chi2_value}')
print(f'p_chi2 half week = {chi2_p}')

n_bins = 4

xmin_west = np.min(0)
xmax_west = np.max(4)
xvals = np.linspace(0.5, 3.5, 4)

print(f'mondays = {mondays}')

west_day_of_week = west_day_of_week[west_day_of_week != 4]
west_day_of_week = west_day_of_week[west_day_of_week != 5]
west_day_of_week = west_day_of_week[west_day_of_week != 6]

fig, ax2 = plotify.get_figax()
ax2.hist(west_day_of_week, bins=n_bins, range=(xmin_west, xmax_west), histtype='step', linewidth=2, color=plotify.c_orange)
ax2.errorbar(xvals, mon_thu, yerr=mon_thu_std, fmt='o', c=plotify.c_blue)
plt.xticks(range(4), ['Monday', 'Tuesday', 'Wednesday', 'Thursday'], size='small')

for label in ax2.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

ax2.set_xlabel("Day")
ax2.set_ylabel("Number of observations")
ax2.set_title("Number of observations from Monday to Thursday")

plt.savefig(('plots/' + 'ufo_half_week'), facecolor=plotify.background_color, dpi=180)

plt.show()


def main():
  # exercise_4_1_1()
  # exercise_4_1_2()
  print('hello')

if __name__ == '__main__':
  main()

