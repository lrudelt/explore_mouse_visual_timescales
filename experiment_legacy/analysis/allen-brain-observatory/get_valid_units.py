#%%
from sys import stderr, argv, exit, path
from os.path import realpath, dirname, exists
import os
import shutil

import numpy as np
import pandas as pd
import h5py
import logging
import argparse

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)
log.setLevel("DEBUG")

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

import yaml
path.insert(2, "../../allen_src/")
import load_spikes

from importlib import reload
reload(load_spikes)

SCRIPT_DIR = dirname(realpath(__file__))
# SCRIPT_DIR = "./"
with open('{}/../../dirs.yaml'.format(SCRIPT_DIR), 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)

## settings
data_directory = dir_settings['allen_data_dir']
manifest_path = os.path.join(data_directory, "manifest.json")
session_type = 'brain_observatory_1.1'
areas = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
# 'brain_observatory_1.1'
# Natural Movie One was a 30 second clips repeated 20 times (2 blocks of 5),
# while Natural Movie Three was a 120 second clip repeated 10 times (2 blocks of 10)

# 'functional_connectivity'
# The Natural Movie One stimulus was presented a total of 60 times, with
# an additional 20 repeats of a temporally shuffled version.

stimuli = {'brain_observatory_1.1' : ["natural_movie_three"],
           'functional_connectivity' : ["natural_movie_one_more_repeats",
                                        "spontaneous"]}
defined_stimuli = ["movie, spontaneous"]

target_length = 1080
transient = 60

#%%

stimulus = 'natural_movie_three'
stimulus_blocks = ['3.0', '6.0']
stimulus_blocks_str = '3.0,6.0'
analysis = 'allen_bo'

#%%
if __name__ == "__main__":
    ## load data
    #%%
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()
    session_ids = sessions[sessions['session_type'] == session_type].index.values
    unit_list = [] 
    session_id_list = [] 
    session_type_list = []
    ecephys_structure_acronym_list = []
    stimulus_list = [] 
    stimulus_blocks_list = [] 
    rec_lengths_list = []  
    firing_rates_list = []
    ret_msg_list = []  
    for session_id in session_ids:
        filename = f"{data_directory}/session_{session_id}/session_{session_id}_spike_data.h5"
        metadata = pd.read_hdf(filename, f"session_{session_id}_stimulus_{stimulus}_stimulus_block_{stimulus_blocks[0]}_metadata")
        for i, unit in enumerate(metadata["unit_id"].values):
            spike_times_merged = load_spikes.get_spike_times(session_id,
                    unit,
                    stimulus, 
                    stimulus_blocks,
                    target_length,
                    transient,
                    data_directory)
            try: # if there is no spike in one of the blocks, skip the unit
                rec_lengths = [spt[-1]-spt[0] for spt in spike_times_merged]
                firing_rates = [len(spt)/(spt[-1]-spt[0]) for spt in spike_times_merged]
            except:
                continue
            unit_list += [unit]
            session_id_list += [session_id]
            session_type_list += [session_type]
            stimulus_list += [stimulus]
            stimulus_blocks_list += [stimulus_blocks_str] 
            ecephys_structure_acronym_list += [metadata["ecephys_structure_acronym"].values[i]]
            ret_msg_list += [load_spikes.assert_appropriateness(unit,
                           session_id,
                           stimulus,
                           stimulus_blocks,
                           target_length,
                           transient,
                           data_directory)]
            if len(spike_times_merged) > 1:
                rec_lengths_list += [f"{rec_lengths[0]},{rec_lengths[1]}"]
                firing_rates_list += [f"{firing_rates[0]},{firing_rates[1]}"]
            else:
                rec_lengths_list += [f"{rec_lengths[0]}"]
                firing_rates_list += [f"{firing_rates[0]}"]      
    d = {"session_id": session_id_list, "session_type": session_type_list, "unit": unit_list, "ecephys_structure_acronym": ecephys_structure_acronym_list, 
         "stimulus": stimulus_list, "stimulus_blocks": stimulus_blocks_list, "rec_lengths": rec_lengths_list, "firing_rates": firing_rates_list, "ret_msg": ret_msg_list}
    df = pd.DataFrame(data = d)     
    df = df[df.ret_msg != "ERR_REC_LEN"] # Exclude units with too small recording length 
    df = df[df.ret_msg != "ERR_EMPTY"] # Exclude units with no spikes
    df = df[df.ret_msg != "ERR_INVALID_TIMES"] # Exclude units with invalid spike times
    df.to_csv(f'valid_unit_list_{analysis}_movie.tsv', sep="\t", index = False)


            
# %%
