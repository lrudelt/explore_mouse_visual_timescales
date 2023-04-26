from sys import path, stderr, argv, exit
from os.path import realpath, dirname, isfile, isdir
import os
import yaml

SCRIPT_DIR = dirname(realpath(__file__))
with open('{}/dirs.yaml'.format(SCRIPT_DIR), 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import path

path.insert(1, "../../allen_src/")
import analysis_utils as utl

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

import urllib.request

from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

### settings

analysis = 'allen_fc'

stimuli = ['natural_movie_one_more_repeats']

T0 = 0.03 # 30 ms
selected_structures = 'cortex+thalamus'

# setup analysis
plot_settings = utl.get_default_plot_settings()
plot_settings['panel_width'] = 0.7 * plot_settings['textwidth']
plot_settings['panel_height'] = 3.2
plot_settings['rcparams']['figure.figsize'] =  [plot_settings['panel_width'], plot_settings['panel_height']]
plt.rcParams.update(plot_settings['rcparams'])

stats_dir = dir_settings['stats_dir']

csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_2500.csv".format(stats_dir, analysis)

structures = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
structures_map = utl.get_structures_map()

data_directory = dir_settings['allen_data_dir']
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# import data

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

data = data[utl.df_filter(data, structures=_structures, stimuli=stimuli, T0=T0, tmin=T0)]
try:
    num_neurons = utl.get_expected_number_of_neurons(analysis,
                                                     selected_structures,
                                                     stimuli)
    assert np.isclose(len(data), num_neurons, atol=100)
except:
    print(f'number of neurons is {len(data)}, expected {num_neurons}')
    exit()

_units_unfiltered = cache.get_units(amplitude_cutoff_maximum=np.inf,
                                    presence_ratio_minimum=-np.inf,
                                    isi_violations_maximum=np.inf)

presence_ratio_minimum, amplitude_cutoff_maximum, isi_violations_maximum = utl.get_quality_metrics(analysis)

_units = cache.get_units(amplitude_cutoff_maximum = amplitude_cutoff_maximum,
                         presence_ratio_minimum = presence_ratio_minimum,
                         isi_violations_maximum = isi_violations_maximum)

num_neurons = {
    'valid waveforms' : {},
    'quality metrics' : {},
    'functional connectivity set' : {},
    'stimuli' : {},
}
print('valid waveforms', len(_units_unfiltered))
for structure in structures:
    num_neurons['valid waveforms'][structure] \
    = len(_units_unfiltered[_units_unfiltered['ecephys_structure_acronym'] == structure])

print('quality metrics', len(_units))
for structure in structures:
    num_neurons['quality metrics'][structure] \
    = len(_units[_units['ecephys_structure_acronym'] == structure])

print('functional connectivity set',
      len(_units[_units['session_type'] == 'functional_connectivity']))
for structure in structures:
    num_neurons['functional connectivity set'][structure] \
    = len(_units[(_units['ecephys_structure_acronym'] == structure) &
                (_units['session_type'] == 'functional_connectivity')])

print('len(data): ', len(np.unique(data['unit'].values)))

for structure in structures:
    num_neurons['stimuli'][structure] \
    = len(np.unique(data[data['ecephys_structure_acronym'] == structure]['unit'].values))

num_neurons_evol = pd.DataFrame(num_neurons['valid waveforms'], index=[0])
num_neurons_evol = num_neurons_evol.append(num_neurons['quality metrics'], ignore_index=True)
num_neurons_evol = num_neurons_evol.append(num_neurons['functional connectivity set'],
                                           ignore_index=True)
num_neurons_evol = num_neurons_evol.append(num_neurons['stimuli'], ignore_index=True)

fig0, ax0 = plt.subplots(figsize=plot_settings['rcparams']['figure.figsize'])

utl.plot_stacked_area(num_neurons_evol, ax0, structures, structures_map)
ax0.set_ylabel('number of units')
ax0.set_xlim([0,3])
ax0.set_ylim([0,47000])
ax0.set_xticks([0,1,2,3])
ax0.set_xticklabels(['valid\nwaveforms', 'quality\nmetrics',
                     'functional\nconnectivity set', 'stimuli'], ha='center')

ax0.text(0.1, 44000, sum(num_neurons['valid waveforms'].values()))
ax0.text(1.1, 15000, sum(num_neurons['quality metrics'].values()))
ax0.text(2.0,  8000, sum(num_neurons['functional connectivity set'].values()))
ax0.text(2.8,  8000, sum(num_neurons['stimuli'].values()))

# legend = ax0.legend(fancybox=False)
# frame = legend.get_frame()
# frame.set_facecolor('0.9')
# frame.set_edgecolor('0.9')
utl.make_plot_pretty(ax0)

utl.save_plot(plot_settings, f"{__file__[:-3]}_per_filter") # name of the script except the .py extension


units = cache.get_units(amplitude_cutoff_maximum = amplitude_cutoff_maximum,
                        presence_ratio_minimum = presence_ratio_minimum,
                        isi_violations_maximum = isi_violations_maximum)


number_of_units_df \
    = pd.DataFrame(columns=['number_of_units',
                            'session_id',
                            'ecephys_structure_acronym'])
number_of_mice = {}

for structure in structures:
    _session_ids = data[utl.df_filter(data,
                                      structures=structure)]['session_id'].unique()
    number_of_mice[structure] \
        = len(units[units['ecephys_session_id'].isin(_session_ids)]['specimen_id'].unique())
    print(structure, number_of_mice[structure])
    for session_id in _session_ids:
        num_units = len(data[utl.df_filter(data,
                                           structures=structure,
                                           sessions=session_id)].unit.unique())
        number_of_units_df \
            = number_of_units_df.append({'number_of_units': num_units,
                                         'session_id': session_id,
                                         'ecephys_structure_acronym': structure},
                                        ignore_index=True)

fig0, ax0 = plt.subplots(figsize=plot_settings['rcparams']['figure.figsize'])

utl.plot_boxplots(ax0,
                  [number_of_units_df[utl.df_filter(number_of_units_df,
                                                    structures=structure)]['number_of_units'].values
                   for structure in structures],
                  [structures_map[structure]['name'] for structure in structures],
                  [structures_map[structure]['color'] for structure in structures])

# utl.make_boxplot_pretty(ax0)
ax0.set_ylabel('#units per session')

ax0.set_ylim([-5, 110])

y_loc1 = ax0.get_ylim()[0] - 0.27 * (ax0.get_ylim()[1] - ax0.get_ylim()[0])
y_loc2 = ax0.get_ylim()[0] - 0.37 * (ax0.get_ylim()[1] - ax0.get_ylim()[0])
ax0.text(-0.08 * len(ax0.get_xaxis().get_majorticklocs()), y_loc1, "Units: ",
         ha='right')
ax0.text(-0.08 * len(ax0.get_xaxis().get_majorticklocs()), y_loc2, "Mice: ",
         ha='right')
for structure, x_loc in zip(structures,
                            ax0.get_xaxis().get_majorticklocs()):
    ax0.text(x_loc,# - 0.02 * len(ax0.get_xaxis().get_majorticklocs()),
             y_loc1,
             number_of_units_df[utl.df_filter(number_of_units_df,
                                              structures=structure)]['number_of_units'].sum(),
             ha='center',
             fontsize=9)
    ax0.text(x_loc,# - 0.02 * len(ax0.get_xaxis().get_majorticklocs()),
             y_loc2,
             number_of_mice[structure],
             ha='center',
             fontsize=9)

utl.make_plot_pretty(ax0)

utl.save_plot(plot_settings, f"{__file__[:-3]}_per_session") # name of the script except the .py extension
