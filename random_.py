import numpy as np
import random

# We use seed value to assure reproducibility of the experiments
random.seed(1)
np.random.seed(1)
####

data = np.array([2, 4, 5, 6], dtype='int8')
faulty_entry_list=[0, 1, 2, 3 ]
r = [1, 2, 4, 7]
control_m = []

# x = np.ones(1, dtype='int8') # Bit flip
x = np.ones(1, dtype='int8') # stuck-at-1
mask = np.array([-1], dtype='int8') # stuck-at-0

control_m = np.random.randint(-1, 1, size = len(faulty_entry_list)) # Generating a number of -1 and 0

# print 'data:', data
# print faulty_entry_list
# print 'pos:',r
# print control_m

for i in range(len(faulty_entry_list)):
    # print control_m[i]
    # if control_m[i] == -1: print 'stuck-at-1'
    print 'org:', data[faulty_entry_list[i]]
    data[faulty_entry_list[i]] = np.bitwise_or(np.bitwise_and(np.bitwise_or(data[faulty_entry_list[i]], np.left_shift(x, r[i])), control_m[i]),
                               np.bitwise_and(np.bitwise_and(data[faulty_entry_list[i]], np.bitwise_xor(mask, np.left_shift(x, r[i]))), control_m[i])) # stuck-at-0
    print 'mutated:', data[faulty_entry_list[i]]
    print '--------'
