
# %%
from pathlib import Path
from sys import exit
from os.path import realpath, dirname, isfile, isdir
import os
import yaml

SCRIPT_DIR = dirname(realpath(__file__))
with open('{}/dirs.yaml'.format(SCRIPT_DIR), 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from sys import path

path.insert(1, "../../allen_src/")
import analysis_utils as utl

from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors

from allensdk.core.reference_space_cache import ReferenceSpaceCache
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

import re

# %%
def get_layer_name(acronym):
    try:
        layer = int(re.findall(r'\d+', acronym)[0])
        if layer == 3:
            layer = 0
        return layer
    except IndexError:
        return 0

# Load analysis results 
center = 'median' # measure of central tendency
T_measure = 'tau_R'
R_measure = 'R_tot'
# C_measure = 'mre_tau'
C_measure = 'tau_two_timescales'
analysis = 'allen_fc'
unit_file = "../../cluster-jobs/allen-fc-merged/valid_unit_list_allen_fc_movie.tsv"
valid_units = pd.read_csv(unit_file, sep='\t', header=0).unit.values


# %%

stats_dir = dir_settings['stats_dir']
csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
R_stats = pd.read_csv(csv_file_name)
for label in R_stats.label:
    unit = label.split(";")[1]

# for R_stat in R_stats:
#     print(R_stat)
#     unit = R_stat.label.split(";")[1]


# %%
mre_stats_file_name = "{}/{}_mre_statistics_Tmin_30_Tmax_10000.csv".format(stats_dir, analysis)
mre_stats = pd.read_csv(mre_stats_file_name)
mre_stats = mre_stats.loc[mre_stats.stimulus == "natural_movie_one_more_repeats"].set_index("unit")

# %%

structures = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
structures_map = utl.get_structures_map()
structures_cortex = [structure for structure in structures
                     if structures_map[structure]['parent_structure'] == 'cortex']

### import data

data = utl.get_analysis_data(csv_file_name, analysis,
                             mre_stats_file_name=mre_stats_file_name)

# make sure data as expected
if (selected_structures == 'cortex+thalamus') :
    _structures = structures
elif (selected_structures == 'cortex') :
    _structures = structures_cortex
else:
    print('unknown structures selected')
    exit()

T0 = 0.03
data = data[utl.df_filter(data, structures=_structures, stimuli=stimuli, T0=T0, tmin=T0)]
try:
    num_neurons = utl.get_expected_number_of_neurons(analysis,
                                                     selected_structures,
                                                     stimuli)
    assert np.isclose(len(data), num_neurons, atol=100)
except:
    print(f'number of neurons is {len(data)}, expected {num_neurons}')
    exit()
# %%

# Load ccf coordinates

atlas_dir = '/data.nst/share/data/allen_brain_atlas'
data_directory = '/data.nst/share/data/allen_visual_coding_neuropixels'

manifest_path = os.path.join(data_directory, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

reference_space_key = os.path.join('annotation', 'ccf_2017')
resolution = 10

rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest=Path(atlas_dir) / 'manifest.json')
# ID 1 is the adult mouse structure graph
tree = rspc.get_structure_tree(structure_graph_id=1) 

# get unit information 
# TODO: adapt default filters to include all the units that we included in the analysis
units = cache.get_units()

# get annotation volume 
annotation, meta = rspc.get_annotation_volume()

# %%
layers = []
units_in_both = []
for unit in valid_units: 
    if unit in units.index:
        unit_data = units.loc[unit]
        # get structure id from ccf coordinates and annotation volume 
        x = np.amax((unit_data.anterior_posterior_ccf_coordinate / 10).astype('int'), 0)
        y = np.amax((unit_data.dorsal_ventral_ccf_coordinate / 10).astype('int'), 0)
        z = np.amax((unit_data.left_right_ccf_coordinate / 10).astype('int'), 0)
        try:
            structure_id = annotation[x, y, z] 
        except:
            print(x,y,z)
        structure = tree.get_structures_by_id([structure_id])
        structure_acronym = structure[0]["acronym"]
        layers += [get_layer_name(structure_acronym)]
        units_in_both += [unit] 
    else: 
        print(unit)

# %% 
layers = np.array(layers)
for layer in np.unique(layers):
    print(layer, len(layers[layers == layer]))


# %%

areas = ['VISp','VISl','VISrl','VISal','VISpm','VISam']
units_in_both = np.array(units_in_both).astype(int)

for area in areas:
    print(area)
    for layer in np.unique(layers):
        tau_C_per_layer = []
        for unit in units_in_both[layers == layer]:
            if unit in mre_stats.index:
                if units.ecephys_structure_acronym[unit] == area:
                    tau_C = mre_stats.loc[unit].tau_two_timescales
                    tau_C_per_layer += [tau_C]
        tau_C_per_layer_low,tau_C_per_layer_high = utl.get_CI_median(tau_C_per_layer)
        print(layer, np.median(tau_C_per_layer), tau_C_per_layer_low, tau_C_per_layer_high, len(tau_C_per_layer))


# %%
from scipy.stats import mannwhitneyu
from scipy.stats import scoreatpercentile
areas = areas[:4]
area_titles = ["primary visual cortex (V1)", "lateromedial area (LM)",  "rostrolateral area (RL)", "anterolateral area (AL)"]
fig, axs = plt.subplots(1, len(areas), figsize=(10, 2), sharey=True)

for i, area in enumerate(areas):
    ax = axs[i]
    ax.set_title(area_titles[i], fontsize = 11, loc='right')
    # ax.set_title("area "+ area)
    label = "intrinsic\n" + r"timescale $τ_{\mathregular{C}}$ (sec)"
    if i == 0:
        ax.set_ylabel(label, fontsize = 11)
    layer_data = []
    for j,layer in enumerate(np.unique(layers)[2:]):
        tau_C_per_layer = []
        for unit in units_in_both[layers == layer]:
            if unit in mre_stats.index and units.ecephys_structure_acronym[unit] == area:
                tau_C = mre_stats.loc[unit].tau_two_timescales
                tau_C_per_layer.append(tau_C)

        tau_C_per_layer_low, tau_C_per_layer_high = utl.get_CI_median(tau_C_per_layer)
        median = np.median(tau_C_per_layer)
        layer_data.append(tau_C_per_layer)
        boxprops = dict(linewidth=1.5, edgecolor='black')
        medianprops = dict(linestyle='-', linewidth=2.5, color='red')
        whiskerprops = dict(linewidth=1.5, linestyle='--', color='black')
        capprops = dict(linewidth=1.5, linestyle='-', color='black')

        boxplot = ax.boxplot([tau_C_per_layer], positions=[j], widths=0.5, showfliers=False,
                patch_artist=True, boxprops=boxprops, medianprops=medianprops,
                whiskerprops=whiskerprops, capprops=capprops)
        # print(median, tau_C_per_layer_low, tau_C_per_layer_high)
        # ax.errorbar(j, median, yerr=[[median - tau_C_per_layer_low], [tau_C_per_layer_high- median]],
                    # fmt='_', color='red', linewidth=2, capsize=5, zorder = 10)
        if j > 0:
            layer1_tau_C = layer_data[j-1]
            layer2_tau_C = layer_data[j]

            # Perform Mann-Whitney U test
            statistic, p_value = mannwhitneyu(layer1_tau_C, layer2_tau_C)

            # Check significance and add brackets on top of box plots
            if p_value < 0.001:
                x1, x2 = j-1 , j 
                y, h, col = boxplot['whiskers'][1].get_ydata()[1], 0.2, 'k'
                # y = 1.2 + j*0.5
                ax.plot([x1, x1, x2, x2], [y + h, y + h, y + h, y + h], linewidth=1.5, color=col)
                ax.text((x1 + x2) * 0.5, y + h, '***', ha='center', va='bottom', color=col)
            elif p_value < 0.01:
                x1, x2 = j-1 , j 
                y, h, col = boxplot['whiskers'][1].get_ydata()[1], 0.2, 'k'
                # y = 1.2 + j*0.5
                ax.plot([x1, x1, x2, x2], [y + h, y + h, y + h, y + h], linewidth=1.5, color=col)
                ax.text((x1 + x2) * 0.5, y + h, '**', ha='center', va='bottom', color=col)
            elif p_value < 0.05:
                x1, x2 = j-1 , j 
                y, h, col = boxplot['whiskers'][1].get_ydata()[1], 0.2, 'k'
                # y = 1.2 + j*0.5
                ax.plot([x1, x1, x2, x2], [y + h, y + h, y + h, y + h], linewidth=1.5, color=col)
                ax.text((x1 + x2) * 0.5, y + h, '*', ha='center', va='bottom', color=col)
            

    ax.set_xticks(np.arange(len(np.unique(layers)[2:])))
    ax.spines["top"].set_bounds(0,0)
    ax.spines["bottom"].set_bounds(0,3)
    ax.spines["left"].set_bounds(0,3)
    ax.set_ylim([-.2,3])
    ax.spines["right"].set_bounds(0,0)
    ax.set_xticklabels(["2/3", "4", "5", "6"])


# Set x-axis labels
# plt.xticks(np.arange(len(np.unique(layers)[2:])), ["2/3", "4", "5", "6"])

# Set common x-axis label
fig.text(0.5, -0.1, 'Cortical layer', ha='center', fontsize=11)



# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Show the plot
plt.show()
    # %%
