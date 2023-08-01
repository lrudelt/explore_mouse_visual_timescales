import logging
log = logging.getLogger("tests")
log.setLevel("DEBUG")

import numpy as np
import h5py as h5
import os.path
import sys
sys.path.insert(0, './experiment_analysis/ana')

import utility as utl

data_path = os.path.abspath(os.path.dirname(__file__) + '/data/data_ref.h5')
log.debug(f"using test data from {data_path}")

def test_binning_formats():

    # get spike times and binned spike counts from data file
    with h5.File(data_path, 'r') as f:
        s1 = f['spike_times_1'][:]
        s2 = f['spike_times_2'][:]
        s3 = f['spike_times_3'][:]
        br1 = f['binned_spikes_1_100ms'][:]
        br2 = f['binned_spikes_2_100ms'][:]
        br3 = f['binned_spikes_3_100ms'][:]
        ref = f['binned_spikes_stacked_100ms'][:]

    stack = np.vstack((s1, s2, s3))

    # (neurons, time)
    binned_stack = utl.binned_spike_count(stack, bin_size=0.1)
    assert np.allclose(binned_stack, ref)

    # single-unit binned series are not the same as pulling them out of the stack
    # because the onset times are shifted on the block-level.
    b1 = utl.binned_spike_count(s1, bin_size=0.1)
    b2 = utl.binned_spike_count(s2, bin_size=0.1)
    b3 = utl.binned_spike_count(s3, bin_size=0.1)
    assert np.allclose(b1, br1)
    assert np.allclose(b2, br2)
    assert np.allclose(b3, br3)
