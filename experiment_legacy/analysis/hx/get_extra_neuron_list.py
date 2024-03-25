import os
import json
import numpy as np
from os.path import exists
import pandas as pd

num_neurons = 100
old_num_neurons = 25
old_extra_num_neurons = []
old_extra_neurons_files = []
assert len(old_extra_num_neurons) == len(old_extra_neurons_files)
extra_neurons = num_neurons - old_num_neurons - sum(old_extra_num_neurons)

# get old neuron nums
old_neurons_files = [f'neuron_list_{old_num_neurons}.tsv'] + old_extra_neurons_files
old_neurons = {}

for old_neurons_file in old_neurons_files:
    old_neurons_df = pd.read_csv(old_neurons_file, sep='\t')
    _kins = np.unique(old_neurons_df['#kins'].values)
    _seeds = np.unique(old_neurons_df['seed'].values)
    for kin in _kins:
        for seed in _seeds:
            if not f'{kin},{seed}' in old_neurons:
                old_neurons[f'{kin},{seed}'] = np.array([])
            old_neurons[f'{kin},{seed}'] = np.append(old_neurons[f'{kin},{seed}'], 
                                                     np.unique(old_neurons_df[(old_neurons_df['#kins'] == kin) & 
                                                                              (old_neurons_df['seed'] == seed)]['neuronNum'].values))


# create new file

neuron_list_name = f'neuron_list_extra_{extra_neurons}.tsv'
file_i = 0
while(exists(neuron_list_name)):
    file_i +=1
    neuron_list_name = f'neuron_list_extra_{extra_neurons}_{file_i}.tsv'

neuron_list = open(neuron_list_name, 'w')


config_file = '/data.nst/bcramer/hx/static_input/static/stdp_config.json'
with open(config_file) as h:
    config = json.load(h)

prefix = "stdp_spikes"
kins = config["kins"]
seeds = config["seeds"]
samples = config["samples"]

assert num_neurons <= 512

header = "#kins\tfilename\tseed\tneuronNum"
neuron_list.write("{}\n".format(header))

for i, kin in enumerate(kins):
    for j, seed in enumerate(seeds):
        neuron_nums = np.random.permutation(512)
        num_added_neurons = 0
        
        for neuron_num in neuron_nums:
            if neuron_num in old_neurons[f'{kin},{seed}']:
                continue
            
            filename = f"{prefix}_{i:03d}_{j:03d}"
            out = "{}\t{}\t{}\t{}".format(kin,
                                          filename,
                                          seed,
                                          neuron_num)
            print(out)
            neuron_list.write("{}\n".format(out))
            num_added_neurons += 1
            if (num_added_neurons == extra_neurons):
                break
neuron_list.close()
