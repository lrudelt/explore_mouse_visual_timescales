#%%
import argparse

defined_stimuli = ["movie",
                   "spontaneous"]

parser = argparse.ArgumentParser()
parser.add_argument('stimulus', type=str, help=f'one of {defined_stimuli}, default: movie', nargs='?', default="movie")
parser.add_argument('--bo', dest='allen_bo', action='store_true',
                    help=f'use brain observatory data set for movie stimulus')
args = parser.parse_args()
if not args.stimulus in defined_stimuli:
    parser.print_help()
    exit()
if args.allen_bo:
    if not args.stimulus == "movie":
        parser.print_help()
        exit()

#%%
from sys import exit, path
from os.path import realpath, dirname, isfile, isdir
import os
import yaml

# SCRIPT_DIR = dirname(realpath(__file__))
SCRIPT_DIR = "./"
with open('{}/../../dirs.yaml'.format(SCRIPT_DIR), 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)

path.insert(1, dir_settings['hdestimator_src_dir'])
path.insert(2, "../../allen_src/")

import hde_utils as hde_utl

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import analysis_utils as utl
import load_spikes
import mre_estimation

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

import urllib.request

from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors

import mrestimator as mre

# supress matplotlib warnings
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings( "ignore", category=MatplotlibDeprecationWarning)
from importlib import reload
reload(utl)
reload(load_spikes)
reload(mre_estimation)

# import logging
# logging.getLogger("mrestimator.utility").setLevel(logging.WARNING)

#%%

### settings
if args.allen_bo:
    stimulus = 'natural_movie_three'
    stimulus_blocks = ['3.0', '6.0']
    analysis = 'allen_bo'
elif args.stimulus == "movie":
    stimulus = 'natural_movie_one_more_repeats'
    stimulus_blocks = ['3.0', '8.0']
    analysis = 'allen_fc'
elif args.stimulus == "spontaneous":
    stimulus = 'spontaneous'
    stimulus_blocks = ['null']
    analysis = 'allen_fc'

#%%
selected_structures = 'cortex' # ONLY UNITS FROM CORTEX!
N_units = 3

# setup analysis
plot_settings = utl.get_default_plot_settings()
plot_settings['panel_width'] = 0.5 * plot_settings['textwidth']
plot_settings['panel_height'] = 2.3
plot_settings['rcparams']['figure.figsize'] =  [plot_settings['panel_width'], plot_settings['panel_height']]
plt.rcParams.update(plot_settings['rcparams'])

stats_dir = dir_settings['stats_dir']

csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_10000_two_timescales.csv".format(stats_dir, analysis)

structures = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
structures_map = utl.get_structures_map()
structures_cortex = [structure for structure in structures
                     if structures_map[structure]['parent_structure'] == 'cortex']

if (selected_structures == 'cortex+thalamus') :
    _structures = structures
elif (selected_structures == 'cortex') :
    _structures = structures_cortex

data_directory = dir_settings['allen_data_dir']

### import data

data = utl.get_analysis_data(csv_file_name, analysis,
                             mre_stats_file_name=mre_stats_file_name)
data = data[utl.df_filter(data, structures=_structures, stimuli=[stimulus], T0=0.03, tmin=0.03)]
sample_units, session_ids = utl.get_sample_units(data = data, size = N_units)


#%%
# TODO: Iterate over conditions here
bin_size = 0.005
dtunit='s'
tmin=0.03
# tmax=2.5
tmax=10
target_length = 1080 # 18 minutes target 
transient = 360 # 6 minute transient per block

fig0, ax0 = plt.subplots()#figsize=plot_settings["panel_size"])
fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1)

for j, unit in enumerate(sample_units):
    session_id = session_ids[j]
    # Load spike data
    spike_times_merged = load_spikes.get_spike_times(session_id,
                    unit,
                    stimulus, 
                    stimulus_blocks,
                    target_length,
                    transient,
                    data_directory)
    color = structures_map[data[(data['unit'] == unit)]['ecephys_structure_acronym'].values[0]]['color']
    binned_spt = load_spikes.get_binned_spike_times(spike_times_merged, bin_size)

    rk = mre.coefficients(mre.input_handler(binned_spt),
                            method='ts',
                            steps=(int(tmin/bin_size),
                                    int(tmax/bin_size)),
                            dt=bin_size,
                            dtunit=dtunit)

    single_timescale_fit = mre_estimation.single_timescale_fit(rk)
    two_timescales_fit, tau_two_timescales, A_two_timescales, tau_rejected, A_rejected = mre_estimation.two_timescales_fit(rk)
    mre_out = mre.OutputHandler(rk, ax0)
    mre_out.add_coefficients(rk, color=color, lw=0.5, alpha=0.6)
    mre_out.add_fit(two_timescales_fit, color=color)#, lw=0.5)
    # TODO: (OPTIONAL) Add vertical lines and diamonds for intrinsic timescales for each neuron
    # mre_out.add_fit(m, color=color)#, lw=0.5)
    # ax0.axvline(x=tau_two_timescales, ls = "--", lw =2, color = color)
    # ax0.plot([tau_two_timescales], [0], "d", markersize = 3, color = color )
    ax0.legend().set_visible(False)

utl.make_plot_pretty(ax0)
ax0.set_ylim([-0.004, 0.108])
ax0.set_ylabel('C(T)')
ax0.set_xticks([0, 100, 200.0, 300, 400.0])
# ax0.set_xticks([0, 500, 1000, 1500, 2000])
ax0.set_xticklabels([f"{x*bin_size*1000:.0f}" for x in ax0.get_xticks()])
ax0.set_xlabel('time lag $T$ (ms)')
ax0.set_xlim([0, 400])
# ax0.set_xscale("log")

# TODO: Add condition as title
# ax0.set_title('intrinsic timescale')

utl.save_plot(plot_settings, f"{__file__[:-3]}")
