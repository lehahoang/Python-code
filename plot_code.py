'''
    @ Put all *npy files to a list
    @ Flatten the 3D numpy array to 1D array
    @ Plot the histogram and seaborn of the data per layer
>>> data=torch.load('logs/2019.11.06-103441/quantized_checkpoint.pth.tar', map_location='cpu')
>>> for e in data['state_dict'].keys():
...     if e.find('wrapped')!=-1 and e.endswith('weight'):
...             temp = data['state_dict'][e].numpy().flatten()
...             np.save(e+'.npy', temp)
FOR AlexNet model
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import scipy.stats as st
import matplotlib.ticker as mticker
import matplotlib.ticker as MultipleLocator
from collections import Counter

modules=['features.module.0.wrapped_module.weight',
       'features.module.3.wrapped_module.weight',
       'features.module.6.wrapped_module.weight',
       'features.module.8.wrapped_module.weight',
       'features.module.10.wrapped_module.weight',
       'classifier.1.wrapped_module.weight',
       'classifier.4.wrapped_module.weight',
       'classifier.6.wrapped_module.weight'
]

layers=['CONV-1',
        'CONV-2',
        'CONV-3',
        'CONV-4',
        'CONV-5',
        'FC-1',
        'FC-2',
        'FC-3'
        ]

path1 = 'logs/2019.11.06-103936/quantized_checkpoint.pth.tar' # symetric int8 with no per-channel quantization
path2 = 'logs/2019.11.06-103441/quantized_checkpoint.pth.tar' # symetric int8 with  per-channel quantization
path3 = 'logs/2019.11.06-105551/quantized_checkpoint.pth.tar' # asymetric int8 with no per-channel quantization
path4 = 'logs/2019.11.06-105759/quantized_checkpoint.pth.tar' # asymetric int8 with  per-channel quantization
path5 = 'logs/2019.11.06-164627/quantized_checkpoint.pth.tar' # symetric int8 with no per-channel quantization (FR=5e-4 and seed =675)
path6 = 'logs/2019.11.06-164702/quantized_checkpoint.pth.tar' # symetric int8 with  per-channel quantization (FR=5e-4 and seed =675)
path7 = 'logs/2019.11.06-164317/quantized_checkpoint.pth.tar' # asymetric int8 with no per-channel quantization (FR=5e-4 and seed =675)
path8 = 'logs/2019.11.06-164231/quantized_checkpoint.pth.tar' # asymetric int8 with  per-channel quantization (FR=5e-4 and seed =675)

model_1 = torch.load(path1, map_location = 'cpu')
# model_2 = torch.load(path8, map_location = 'cpu')

for module,layer in zip(modules, layers):
    data_1 = model_1['state_dict'][module].numpy().flatten()
    # data_2 = model_2['state_dict'][module].numpy().flatten()
    plt.figure(figsize=(20,20))

    a1 = plt.hist(data_1.flatten(),bins=50, alpha=0.5, density=False, label='Fault-free execution')
    # a2 = plt.hist(data_2.flatten(),bins=50, alpha=0.5, density=False, label='Faulty execution', color='brown')

    # a1 = plt.hist(data_1.flatten(),bins=50, alpha=0.5, density=False, label='no_per_channel')
    # a2 = plt.hist(data_2.flatten(),bins=50, alpha=0.5, density=False, label='per_channel', color = 'red')

    #plt.xlabel("Values", fontsize=70)
    #plt.ylabel("Frequency", fontsize=80)
    plt.xticks(fontsize=30, rotation=0)
    plt.yticks(fontsize=30, rotation=0)
    plt.locator_params(axis='x', nbins=6)# For data point 1
    plt.title(layer, fontsize=30)
    plt.legend(loc='upper right',fontsize=30)
    plt.savefig('alexnet/'+layer+'.png', format='png')
