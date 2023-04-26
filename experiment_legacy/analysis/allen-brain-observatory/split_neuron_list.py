import argparse
from sys import stderr, argv, exit
from os.path import isfile,realpath, dirname
import os
import shutil

import numpy as np
import pandas as pd

list_versions = {'stimuli' : {'natural_movies' : ["natural_movie_one",
                                                  "natural_movie_three",
                                                  "natural_movie_one_more_repeats",
                                                  "natural_movie_one_shuffled"],
                              'drifting_gratings' : ["drifting_gratings",
                                                     "drifting_gratings_contrast",
                                                     "drifting_gratings_75_repeats"],
                              'gabors' : ["gabors"],
                              'spontaneous' : ["spontaneous"]},
                 'analysis_v1' : {'analysis_v1' : ["gabors",
                                                   "drifting_gratings_contrast",
                                                   "drifting_gratings_75_repeats",
                                                   "natural_movie_one_more_repeats",
                                                   "spontaneous"],
                                  'analysis_v1_nm_sp' : ["natural_movie_one_more_repeats",
                                                         "spontaneous"]},
                 'analysis_v2' : {'analysis_v2' : ["natural_movie_three"]},
                 }
                 

def split_list(list_name,
               list_version,
               min_length):
    neuron_list = pd.read_csv(list_name, sep='\t', keep_default_na=False)

    empty_filter = [True]*len(neuron_list)

    session_filter = empty_filter
    ret_msg_filter = empty_filter
    
    if list_version == 'analysis_v1':
        session_filter = [session_type == "functional_connectivity"
                          for session_type in neuron_list['session_type'].values]
        ret_msg_filter = [ret_msg == "SUCCESS"
                          for ret_msg in neuron_list['ret_msg'].values]
    elif list_version == 'analysis_v2':
        session_filter = [session_type == "brain_observatory_1.1"
                          for session_type in neuron_list['session_type'].values]
        ret_msg_filter = [ret_msg == "SUCCESS" or ret_msg == "ERR_NON_STATIONARY" 
                          for ret_msg in neuron_list['ret_msg'].values]

    
    rec_len_filter = [sum([float(rec_len) for rec_len in str(rec_lens).split(',')]) >= min_length
                      for rec_lens in neuron_list['rec_len'].values]

    for target_file_name in list_versions[list_version]:
        target_stimuli = list_versions[list_version][target_file_name]
        neuron_list[(neuron_list['stimulus'].isin(target_stimuli)) &
                    session_filter &
                    ret_msg_filter &
                    rec_len_filter].to_csv('{}_{}.tsv'.format(list_name[:-4],
                                                              target_file_name),
                                           index = False,
                                           sep='\t')
        

def parse_arguments(list_versions):
    # parse arguments
    parser = argparse.ArgumentParser()
    optional_arguments = parser._action_groups.pop()
    
    required_arguments = parser.add_argument_group("required arguments")
    required_arguments.add_argument('list_name', action="store")

    optional_arguments.add_argument("-v", "--list-version", metavar="NAME", action="store", help="Define lists to be created.  One of {}.".format(list_versions),
                                    default="stimuli")
    optional_arguments.add_argument("-l", "--min-length", metavar="LENGTH", action="store", default=0)

    parser._action_groups.append(optional_arguments)
    args = parser.parse_args()

    # check that parsed arguments are valid
    list_name = args.list_name
    list_version = args.list_version
    min_length = args.min_length

    if not isfile("{}".format(list_name)):
        print('{} not found.'.format(list_name))
        exit()

    if not list_version in list_versions:
        print('list version must be one of {}'.format(list_versions))

    try:
        min_length = float(min_length)
    except:
        print('min_length must be a number.')
        exit()           

    return list_name, list_version, min_length
    
if __name__ == "__main__":    
    list_name, list_version, min_length \
        = parse_arguments(list_versions.keys())
    
    exit(split_list(list_name=list_name,
                    list_version=list_version,
                    min_length=min_length))
