from sys import stderr, argv, exit
from os.path import realpath, dirname
import os
import shutil

import numpy as np
import pandas as pd

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

import yaml
import print_spike_times_for_neuron as pspt

SCRIPT_DIR = dirname(realpath(__file__))
with open('{}/../../information-theory/dev/ma_plots/dirs.yaml'.format(SCRIPT_DIR), 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)

## settings
data_directory = dir_settings['allen_data_dir']
manifest_path = os.path.join(data_directory, "manifest.json")
test_session_ids = [755434585, 816200189] # 721123822, 774875821

areas = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']

# 'brain_observatory_1.1'
# Natural Movie One was a 30 second clips repeated 20 times (2 blocks of 10),
# while Natural Movie Three was a 120 second clip repeated 10 times (2 blocks of 5)

# 'functional_connectivity'
# The Natural Movie One stimulus was presented a total of 60 times, with
# an additional 20 repeats of a temporally shuffled version.

stimuli = {'brain_observatory_1.1' : ["natural_movie_one",
                                      "natural_movie_three",
                                      "drifting_gratings",
                                      "gabors"],
           'functional_connectivity' : ["natural_movie_one_more_repeats",
                                        "natural_movie_one_shuffled",
                                        "drifting_gratings_contrast",
                                        "drifting_gratings_75_repeats",
                                        "gabors",
                                        "spontaneous"]}

# np.unique(presentations['stimulus_name'])
# 'brain_observatory_1.1'
# array(['drifting_gratings', 'flashes', 'gabors', 'natural_movie_one',
#        'natural_movie_three', 'natural_scenes', 'spontaneous',
#        'static_gratings'], dtype=object)
# 'functional_connectivity'
# array(['dot_motion', 'drifting_gratings_75_repeats',
#        'drifting_gratings_contrast', 'flashes', 'gabors',
#        'natural_movie_one_more_repeats', 'natural_movie_one_shuffled',
#        'spontaneous'], dtype=object)

quality_metrics = {'unmerged': {'presence_ratio_minimum': 0.9, # was -np.inf,  # default: 0.9
                                'amplitude_cutoff_maximum': 0.01,   # default: 0.1,
                                'isi_violations_maximum': 0.5},     # default: 0.5
                   'merged': {'presence_ratio_minimum': 0.9,
                              'amplitude_cutoff_maximum': 0.01,
                              'isi_violations_maximum': 0.5}}

def get_neuron_list(list_version,
                    test):
    ## load data
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()

    ## write neuron lists
    isi_violations_maximum = quality_metrics[list_version]['isi_violations_maximum']
    amplitude_cutoff_maximum = quality_metrics[list_version]['amplitude_cutoff_maximum']
    presence_ratio_minimum = quality_metrics[list_version]['presence_ratio_minimum']

    if test:
        neuron_list_name = 'allen_neuron_list_{}_test.tsv'.format(list_version)
    else:
        neuron_list_name = 'allen_neuron_list_{}.tsv'.format(list_version)

    allen_neuron_list = open(neuron_list_name, 'w')
    header = "#{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format("session_id",
                                                              "session_type",
                                                              "unit",
                                                              "ecephys_structure_acronym",
                                                              "stimulus",
                                                              "stimulus_blocks",
                                                              "block_durations",
                                                              "rec_len",
                                                              "firing_rates",
                                                              "ret_msg")
    allen_neuron_list.write("{}\n".format(header))

    if test:
        session_ids = test_session_ids
    else:
        session_ids = sessions.index.values

    for session_id in session_ids:
        session = cache.get_session_data(session_id,
                                         isi_violations_maximum = isi_violations_maximum,
                                         amplitude_cutoff_maximum = amplitude_cutoff_maximum,
                                         presence_ratio_minimum = presence_ratio_minimum)

        session_type = session.session_type

        stimulus_epochs = session.get_stimulus_epochs()

        units = session.units[session.units["ecephys_structure_acronym"].isin(areas)]

        for unit in units.index.values:
            ecephys_structure_acronym = units.loc[unit]['ecephys_structure_acronym']

            for stimulus in stimuli[session_type]:
                presentations = session.get_stimulus_table(stimulus)

                stimulus_blocks = [str(s) for s in
                                   stimulus_epochs[stimulus_epochs['stimulus_name']
                                                   == stimulus]['stimulus_block'].unique()]

                if list_version == 'merged':
                    stimulus_blocks = [','.join(stimulus_blocks)]
                
                for _stimulus_blocks in stimulus_blocks:
                    stimulus_blocks_idxs \
                        = pspt.get_stimulus_blocks_indices(stimulus_epochs,
                                                           [pspt.sbfmt(stimulus_block)
                                                            for stimulus_block
                                                            in _stimulus_blocks.split(',')])
                    block_durations \
                        = [block_duration
                           for block_duration
                           in stimulus_epochs[stimulus_epochs.index.isin(
                            stimulus_blocks_idxs)]['duration'].values]
                    target_length = sum(block_durations)
                    block_durations = ",".join(["{:.2f}".format(s) for s in block_durations])
                    
                    stimulus_presentation_ids \
                        = pspt.get_stimulus_presentation_ids(presentations,
                                                             stimulus_epochs,
                                                             session_type,
                                                             stimulus,
                                                             _stimulus_blocks)

                    spike_times = pspt.get_spike_times(session,
                                                       unit,
                                                       stimulus_presentation_ids,
                                                       target_length,
                                                       transient=0)

                    ret_val = pspt.assert_appropriateness(spike_times,
                                                          session,
                                                          session_id,
                                                          unit,
                                                          session_type,
                                                          stimulus,
                                                          _stimulus_blocks,
                                                          target_length,
                                                          transient=0)

                    rec_lens = ",".join(["{:.2f}".format(spt[-1] - spt[0])
                                         if len(spt) > 1
                                         else "0.00"
                                         for spt in spike_times.values()])

                    firing_rates = ",".join(["{:.2f}".format(len(spt) / (spt[-1] - spt[0]))
                                             if len(spt) > 1
                                             else "0.00"
                                             for spt in spike_times.values()])
                    
                    ret_msg = list(pspt.return_values.keys())[list(pspt.return_values.values()).index(ret_val)]

                    out = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(session_id,
                                                                          session_type,
                                                                          unit,
                                                                          ecephys_structure_acronym,
                                                                          stimulus,
                                                                          _stimulus_blocks,
                                                                          block_durations,
                                                                          rec_lens,
                                                                          firing_rates,
                                                                          ret_msg)

                    print(out)
                    allen_neuron_list.write("{}\n".format(out))

    exit()


def print_usage_and_exit(script_name):
    print('usage is python3 {} list_version [test]'.format(script_name))
    print('with list_version one of [unmerged, merged]')
    print("if 'test' is passed, it is run for two sessions only")
    exit()
    
if __name__ == "__main__":    
    if len(argv) < 2 or len(argv) > 3 \
       or not argv[1] in ['unmerged', 'merged']:
        print_usage_and_exit(argv[0])

    test = False
    if len(argv) == 3:
        if not argv[2] == 'test':
            print_usage_and_exit(argv[0])
        else:
            test = True

    exit(get_neuron_list(list_version=argv[1],
                         test=test))
