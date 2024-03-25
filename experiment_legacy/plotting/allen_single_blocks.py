import argparse

from sys import exit, path
import os
import yaml

with open('dirs.yaml', 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import patsy as pt
import scipy.stats as st
import arviz as az

path.insert(1, "../../allen_src/")
import analysis_utils as utl

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors
import seaborn as sns

import importlib
importlib.reload(utl);

def get_relative_within_unit_block_difference(data_block_1, data_block_2, measure):
    block_differences = (data_block_2[measure].values - data_block_1[measure].values) / ((data_block_1[measure].values+(data_block_2[measure].values))/2)
    return block_differences

def get_relative_between_unit_differences(data_block_1, data_block_2, measure):
    block_differences = []
    for i, measure_block_1 in enumerate(data_block_1[measure].values):
        session_id = data_block_1["session_id"].values[i]
        block_differences_unit = (data_block_2[measure].values[data_block_2["session_id"]==session_id] - measure_block_1)/ ((data_block_2[measure].values[data_block_2["session_id"]==session_id] + measure_block_1)/2)
        block_differences = np.append(block_differences, block_differences_unit)
    return block_differences

def get_absolute_within_unit_block_difference(data_block_1, data_block_2, measure):
    block_differences = (data_block_2[measure].values - data_block_1[measure].values)
    return block_differences

def get_absolute_between_unit_differences(data_block_1, data_block_2, measure):
    block_differences = []
    for i, measure_block_1 in enumerate(data_block_1[measure].values):
        session_id = data_block_1["session_id"].values[i]
        block_differences_unit = (data_block_2[measure].values[data_block_2["session_id"]==session_id] - measure_block_1)
        block_differences = np.append(block_differences, block_differences_unit)
    return block_differences

defined_measures = ["tau_C",
                    "tau_R",
                    "R_tot"]
defined_difference_types = ["absolute", "relative"]
parser = argparse.ArgumentParser()
parser.add_argument('measure', type=str, help=f'one of {defined_measures}')
parser.add_argument('difference_type', type=str, help=f'one of {defined_difference_types}')
args = parser.parse_args()
# args.measure = 'tau_C'
__file__ = 'allen_single_blocks.py'

seed = 12347
analysis = 'allen_fc_single_blocks'

center = 'median' # measure of central tendency
# T_measure = 'log_tau_R'
T_measure = 'tau_R'
R_measure = 'R_tot'
# C_measure = 'log_mre_tau'
C_measure = 'mre_tau'

stimuli = ['natural_movie_one_more_repeats']

T0 = 0.03 # 30 ms
#selected_structures = 'cortex+thalamus'
selected_structures = 'cortex'

plot_stars = False # plot stars or p-values for tests

# setup analysis
plot_settings = utl.get_default_plot_settings()
plt.rcParams.update(plot_settings['rcparams'])
plot_settings['imgdir'] = '../img/'

stats_dir = dir_settings['stats_dir']

csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_2500.csv".format(stats_dir, analysis)

structures = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
# structures = ['VISp','VISl','VISrl','VISal','VISpm','VISam']
structures_map = utl.get_structures_map()
structures_cortex = [structure for structure in structures
                     if structures_map[structure]['parent_structure'] == 'cortex']

data_directory = dir_settings['allen_data_dir']
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

### import data

analysis_metrics = utl.get_analysis_metrics(cache,
                                            analysis)
data = utl.get_analysis_data(csv_file_name, analysis,
                             analysis_metrics=analysis_metrics,
                             mre_stats_file_name=mre_stats_file_name)

# make sure data as expected
if (selected_structures == 'cortex+thalamus') :
    _structures = structures
elif (selected_structures == 'cortex') :
    _structures = structures_cortex
else:
    print('unknown structures selected')
    exit()

data = data[utl.df_filter(data, structures=_structures, stimuli=stimuli, T0=T0, tmin=T0)]
try:
    num_neurons = utl.get_expected_number_of_neurons(analysis,
                                                     selected_structures,
                                                     stimuli)
    assert np.isclose(len(data), num_neurons, atol=100)
except:
    print(f'number of neurons is {len(data)}, expected {num_neurons}')
    exit()

if args.measure == "tau_C":
    measure = C_measure
    measure_name = r'intrinsic timescale'# $τ_{\mathregular{C}}$ (ms)'
    measure_name_short = '$τ_{\mathregular{C}}$ (ms)'
    lims = [-2.6, 1.8]
    ticks = [-2., -1., 0., 1.]
elif args.measure == "tau_R":
    measure = T_measure
    measure_name = 'information timescale'# $τ_R$ (ms)'
    measure_name_short = '$τ_R$'
    lims = [-3.1, 0.1]
    ticks = [-3., -2., -1., 0.]
elif args.measure == "R_tot":
    measure = R_measure
    measure_name = r'predictable information'# $R_{\mathregular{tot}}$'
    measure_name_short = r'$R_{\mathregular{tot}}$'
    lims = [-0.03, 0.48]
    ticks = [0, 0.1, 0.2, 0.3, 0.4]


selection = utl.get_data_filter(data, measure)
data = data[selection]

# Selecting only units that appear in both blocks
_units = []
_stimulus_blocks = np.unique(data['stimulus_blocks'].values)
for unit in np.unique(data['unit'].values):
    if len(data[data['unit'] == unit]) == len(_stimulus_blocks):
        _units += [unit]
data = data[data['unit'].isin(_units)]

data_block_1 = data[data['stimulus_blocks'] == '3.0']
data_block_2 = data[data['stimulus_blocks'] == '8.0']

if args.difference_type == "relative":
    within_unit_differences = get_relative_within_unit_block_difference(data_block_1, data_block_2, measure)
    between_unit_differences = get_relative_between_unit_differences(data_block_1, data_block_2, measure)
elif args.difference_type == "absolute":
    within_unit_differences = get_absolute_within_unit_block_difference(data_block_1, data_block_2, measure)
    between_unit_differences = get_absolute_between_unit_differences(data_block_1, data_block_2, measure)

fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
selection_within_unit_differences = np.abs(within_unit_differences) <= 2.0
selection_between_unit_differences = np.abs(between_unit_differences) <= 2.0
within_unit_differences = within_unit_differences[selection_within_unit_differences]
between_unit_differences = between_unit_differences[selection_between_unit_differences]

if measure == 'mre_tau':
    selection_within_unit_differences = np.abs(within_unit_differences) <= 2.0
    selection_between_unit_differences = np.abs(between_unit_differences) <= 2.0
    within_unit_differences = within_unit_differences[selection_within_unit_differences]
    between_unit_differences = between_unit_differences[selection_between_unit_differences]
    if args.difference_type == "relative":
        ax0.set_xlabel(r'relative difference $\Delta \tau_C$')
    elif args.difference_type == "absolute":
        ax0.set_xlabel(r'absolute difference $\Delta \tau_C$ (s)')
        # within_unit_differences = within_unit_differences * 1000
        # between_unit_differences = between_unit_differences * 1000
        # selection_within_unit_differences = np.abs(within_unit_differences) <= 2000
        # selection_between_unit_differences = np.abs(between_unit_differences) <= 2000
    ax0.set_title(r'intrinsic timescale')
elif measure == 'tau_R':
    if args.difference_type == "relative":
        ax0.set_xlabel(r'relative difference $\Delta \tau_R$')
    elif args.difference_type == "absolute":
        selection_within_unit_differences = np.abs(within_unit_differences) <= 0.3
        selection_between_unit_differences = np.abs(between_unit_differences) <= 0.3
        within_unit_differences = within_unit_differences[selection_within_unit_differences]
        between_unit_differences = between_unit_differences[selection_between_unit_differences]
        ax0.set_xlabel(r'absolute difference $\Delta \tau_R$ (s)')
        ax0.set_xlim([-0.3,0.3])
        # within_unit_differences = within_unit_differences * 1000
        # between_unit_differences = between_unit_differences * 1000
    ax0.set_title(r'information timescale')
elif measure == 'R_tot':
    ax0.set_xlabel(r'%s difference $\Delta R_{tot}$'%args.difference_type)
    ax0.set_title(r'predictability')
sns.distplot(within_unit_differences, norm_hist=True, kde=False,
             bins=int(650/5), color = 'blue', ax = ax0, label = "same unit")
             # hist_kws={'edgecolor':'black'})
# sns.distplot(within_unit_differences, color = 'blue', ax = ax0, label = "same unit")
sns.distplot(between_unit_differences, norm_hist=True, kde=False,
             bins=int(650/5), color = 'red' , ax = ax0, label = "other units") #hist_kws={'edgecolor':'black'}
# sns.distplot(between_unit_differences, color = 'red', ax = ax0, label = "other units")
ax0.plot([np.median(within_unit_differences)], [0], marker = 'd', color = 'blue', markersize = 3)
ax0.plot([np.median(between_unit_differences)], [0], marker = 'd', color = 'red', markersize = 3)
ax0.set_ylabel('normalized frequency')
if args.difference_type == "relative":
    ax0.set_ylim([0, 1.6])
    ax0.set_yticks([0, 0.5, 1.0, 1.5])
# elif args.difference_type == "absolute":
#     ax0.set_ylim([0, 3.0])
#     ax0.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
ax0.legend(loc = 'upper right')
utl.make_plot_pretty(ax0)
utl.save_plot(plot_settings, f"{__file__[:-3]}_distribution_plot_{args.difference_type}", measure=args.measure)
plt.show()
plt.close()
#
#
# fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
#
# quantiles_block_1 = np.quantile(data_block_1[measure].values, np.arange(0.1, 1.0, 0.1))
# quantiles_block_2 = np.quantile(data_block_2[measure].values, np.arange(0.1, 1.0, 0.1))
#
# ax0.plot(data_block_1[measure].values, data_block_2[measure].values, 'k.')
#
# ax0.plot(quantiles_block_1, quantiles_block_2, '.', ms=10, color='0.6')
#
# utl.format_ax(ax0, measure, 'x')
# utl.format_ax(ax0, measure, 'y')
#
# ax0.set_xlabel('NM block 1')
# ax0.set_ylabel('NM block 2')
# ax0.set_title(measure_name)
# utl.make_plot_pretty(ax0)
# ax0.grid(axis = 'x', color='0.9', linestyle='-', linewidth=1)
#
# ax0.set_xlim(lims)
# ax0.set_ylim(lims)
#
# ax0.set_xticks(ticks)
# ax0.set_yticks(ticks)
#
# plt.show()
# plt.close()
#
# utl.save_plot(plot_settings, f"{__file__[:-3]}_qq_plot", measure=args.measure)
