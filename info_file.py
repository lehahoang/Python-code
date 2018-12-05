'''
This info_file must be executed once in his life
'''
import csv

read_data_file =True
# The header will be written if the above condition is False. This should be set
# at the beginning of everything
# Otherwise, information of the Python file will be written
header =[
        'Python file name',
        'Neuron model',
        '# Layers'
        '# Total Neurons| num',
        'Membrane time constant| tau_mem(ms)',
        'Membrane Firing threshold voltage| Vth(mV)',
        'Membrane Reset potential| Vreset(mV)',
        'Membrane Reversal potential| El(mV)',
        '# Excitatory neurons| Ne',
        'Membrane time constant of excitatory neurons| tau_mem_e(ms)',
        'Refractory period of excitatory neurons| tau_refrac_e(ms)'
        'Excitatory synaptic time constant| tau_syn_e(ms)',
        'Leak Reversal potential of excitatory neurons| Vleak_e(mV)',
        'Reset potential of excitatory neurons| Vreset_e(mV)',
        'Excitatory synaptic reversal potential| Vrev_e(mV)',
        'Firing threshold potential of excitatory neurons| Vth_e(mV)'
        '# Inhibitory neurons| Ni',
        'Membrane time constant of inhibitory neurons| tau_mem_i(ms)',
        'Refractory period of inhibitory neurons| tau_refrac_i(ms)'
        'Inhibitory synaptic time constant| tau_syn_i(ms)',
        'Leak Reversal potential of inhibitory neurons| Vleak_i(mV)',
        'Reset potential of inhibitory neurons| Vreset_i(mV)',
        'Inhibitory synaptic reversal potential| Vrev_i(mV)',
        'Firing threshold potential of inhibitory neurons| Vth_i(mV)'
]

# info = ['AC_alon_ok.py',            # Python Filename
#         str(num),                   # Number of neurons
#         'CUBA LIF with Alternative current', # neuron model
#         str(tau),                   # Membrane time constant
#         str(Vth),                   # Threshold voltage
#         str(VRest),                 # Resting voltage
#         str(El),                    # Reversal potential
#         str(1),                     # Number of layers
# ]
# '''
# Writing data to a csv file as a dictionary
# Reading a (dict) csv file
# '''
# with open('./info1.csv','w') as info_file:
#     fieldnames = ['#neurons','Neuron model']
#     writer     = csv.DictWriter(info_file,fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerow({'#neurons':str(num),
#                     'Neuron model':'LIF'})
#
# with open('./info1.csv','r') as info_file:
#     reader=csv.DictReader(info_file,delimiter=',')
#     for row in reader:
#         print(row)
# #---------------------------------------------------
if not read_data_file:
    print('Wrtiting the infofile\n')
    with open('./info.csv','w') as info_file:
        writer = csv.writer(info_file,delimiter=',')
        writer.writerow(header)
else:
    print('Reading the infofile\n')
    with open('./info.csv','r') as info_file:
        reader=csv.reader(info_file,delimiter=',')
        for row in reader:
            print(row)
