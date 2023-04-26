import os
import json
import numpy as np

num_neurons = 25


neuron_list_name = f'neuron_list_{num_neurons}.tsv'
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
        for k in range(num_neurons):
            neuronNum = np.random.randint(512)
            filename = f"{prefix}_{i:03d}_{j:03d}"
            out = "{}\t{}\t{}\t{}".format(kin,
                                          filename,
                                          seed,
                                          neuronNum)
            print(out)
            neuron_list.write("{}\n".format(out))
neuron_list.close()
