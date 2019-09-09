'''
    @ Put all *npy files to a list
    @ Flatten the 3D numpy array to 1D array
    @ Plot the histogram and seaborn of the data per layer
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import scipy.stats as st
import matplotlib.ticker as mticker
import matplotlib.ticker as MultipleLocator
from collections import Counter


data= np.load('./conv5.npy')
# data= np.load('./conv5_fr_1.npy')
# data= np.load('./conv5_fr_2.npy').flatten()

#np.savetxt('../../../data.csv', data.flatten(), delimiter = ',')

# values, counts = np.unique(data, return_counts = True)
# # Counter(data.flatten())
# # counts = Counter(data.flatten()).values()
# # values = Counter(data.flatten()).keys()
#
#
# np.savetxt('./value_dat.csv', values, delimiter = ',')
# np.savetxt('./counts_dat.csv', counts, delimiter = ',')
# print(len(values))
# print(np.max(values))
#
# print(len(counts))
# print(np.max(counts))
# print (values)
max_ = np.amax(data)


#
#plt.figure(figsize = [100,100])

plt.figure(figsize=(20,20))
ax = plt.hist(data.flatten(),bins=10, log=True, density=False)
#plt.xlim(0, max_)

plt.xlabel("Activation magnitude", fontsize=70)
plt.ylabel("Frequency", fontsize=80)
# Displaying the x tick as 1x10^{36}, not 1E^{36} by next three command lines
#
# f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
# g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
# plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(g))

plt.xticks(fontsize=50, rotation=0)
plt.yticks(fontsize=60, rotation=0)
plt.locator_params(axis='x', nbins=5)
# plt.savefig('./test.eps', format='eps')

#plt.grid(b=True, which='major', axis = 'both', color='k', linestyle='--')
# plt.legend(fontsize=25)
plt.savefig('./test.png', format='png')
# plt.show()
