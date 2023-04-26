from os.path import join
from sys import stderr, argv, exit
import argparse
import numpy as np

data_dir = '/data.nst/bcramer/hx/static_input/static/'

def print_spike_times(filename,
                      neuron_num):

    min_diff = 1

    for sample in range(30):
        last_spt = 0
        if sample > 0:
            print('----------')

        spikes = np.load(join(data_dir, 
                              f"{filename}_{sample:03d}.npy"))

        # spikes is array of shape [N_spikes, 2]
        # index 0 of last dimension: time stemps
        # index 1 of last dimension: address

        # extract spike times for every neuron
        # times are still in hardware domain, i.e. need to be multiplied
        # by 1000 to obtain biological equivalent time
        neuron_mask = (spikes[:, 1] == neuron_num)
        times = spikes[neuron_mask, 0] * 1e3
        for spt in times:
            if spt > 31:
                continue

            print(spt)
            if spt - last_spt < min_diff:
                min_diff = spt - last_spt
            last_spt = spt

    # print('min diff: ', min_diff)

def print_usage_and_exit(script_name):
    print('usage is python3 {} filename neuronNum'.format(script_name))
    exit()

if __name__ == "__main__":
    # format of the neuron list is
    # kins	filename	seed	neuronNum
    
    if not len(argv) == 3 \
       or not argv[2].isdecimal():
        print_usage_and_exit(argv[0])

    assert 0 <= int(argv[2]) < 512

    exit(print_spike_times(filename=argv[1],
                           neuron_num=int(argv[2])))
