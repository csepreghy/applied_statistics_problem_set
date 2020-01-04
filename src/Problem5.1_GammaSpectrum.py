#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ----------------------------------------------------------------------------------- #
#
#  Script for reading data for the Applied Statistics problem on "Gamma Spectrum".
#
#  Author: Troels C. Petersen (NBI)
#  Email:  petersen@nbi.dk
#  Date:   1st of December 2019
# ----------------------------------------------------------------------------------- #


from array import array
import numpy as np

# ----------------------------------------------------------------------------------- #
# Read data (channel numbers):
# ----------------------------------------------------------------------------------- #

x = []

with open( 'data_GammaSpectrum.txt', 'r' ) as infile :
    for line in infile:
        line = line.strip().split()
        x.append(float(line[0]))

        # Print the first 10 numbers as a sanity check:
        if (len(x) < 10) :
            print(x[-1])

x = np.array(x)
print("The number of entries in the file was: ", len(x))

