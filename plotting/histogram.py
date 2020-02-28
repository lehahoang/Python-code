from numpy import array, log, pi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.pyplot as plt

fc1_fault =np.load('./fc1_fault.npy')

plt.figure()
plt.figure(figsize=(20,20))
plt.hist(fc1_fault.flatten(), bins=20, label='Fancy labels', log=True, density=False)

ax=plt.gca()
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.xaxis.get_major_formatter().set_scientific(True)
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.1e' % x))
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(g))
plt.locator_params(axis='x', nbins=5)# For data point 1
plt.xlabel("Activation magnitude", fontsize=60)
plt.ylabel("Frequency", fontsize=60)
plt.xticks(fontsize=55, rotation=0)
plt.yticks(fontsize=55, rotation=0)
# plt.show()
plt.savefig('sx.png', format='png')
