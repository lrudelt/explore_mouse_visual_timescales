from sys import exit, path
from os.path import realpath, dirname, isfile, isdir
import os
import yaml

SCRIPT_DIR = dirname(realpath(__file__))
with open('{}/dirs.yaml'.format(SCRIPT_DIR), 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as st
from scipy.stats import pearsonr

path.insert(1, "../../allen_src/")
import analysis_utils as utl

from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors

### settings

analysis = 'allen_fc'

center = 'median' # measure of central tendency
T_measure = 'log_tau_R'
R_measure = 'R_tot'
C_measure = 'log_mre_tau'

stimuli = ['natural_movie_one_more_repeats']
T0 = 0.03 # 30 ms
selected_structures = 'cortex'

# setup analysis
plot_settings = utl.get_default_plot_settings()
plt.rcParams.update(plot_settings['rcparams'])

stats_dir = dir_settings['stats_dir']

csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_10000_two_timescales.csv".format(stats_dir, analysis)
# mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_2500.csv".format(stats_dir, analysis)
pp_stats_file_name = "{}/{}_pp_statistics.csv".format(stats_dir, analysis)

structures = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
structures_map = utl.get_structures_map()
structures_cortex = [structure for structure in structures
                     if structures_map[structure]['parent_structure'] == 'cortex']

### import data

data = utl.get_analysis_data(csv_file_name, analysis,
                             mre_stats_file_name=mre_stats_file_name,
                             pp_stats_file_name=pp_stats_file_name)

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

# plot

for measure in [C_measure, T_measure, R_measure]:
    selection = utl.get_data_filter(data, measure)
    data = data[selection]

fig0, axes = plt.subplots(3,3, figsize=(2.0*plot_settings["panel_width"],3.75))
fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1, wspace=0.1, hspace=0.33)

pairplot_measures = [C_measure, T_measure, R_measure]

lims = {C_measure : (-2.3, 0.97),
        T_measure : (-3.1, -0.3),
        R_measure : (-0.01, 0.41)}
ticks = {C_measure : [-2.0, -1.0, 0.0],
         T_measure : [-3.0, -2.0, -1.0],
         R_measure : [0, 0.1, 0.2, 0.3, 0.4]}
labels = {C_measure : '$τ_{\mathregular{C}}$ (ms)',
          T_measure : '$τ_R$ (ms)',
          R_measure : r'$R_{\mathregular{tot}}$'}
fmts = {C_measure : utl.format_label_log_in_ms,
        T_measure : utl.format_label_log_in_ms,
        R_measure : None}

# m, b = np.polyfit(metric_selected_data, measure_selected_data, 1)
# x = np.linspace(x_min,x_max,10)
# ax.plot(x, m*x + b, color = ".2")


utl.plot_pairplot(data, pairplot_measures, axes, lims, ticks, labels, fmts)

# Plot p values in upper triangle of the matrix
for i, measure_pair in enumerate([[T_measure, C_measure], [R_measure, C_measure], [R_measure, T_measure]]):
    r , p_val = pearsonr(data[measure_pair[0]], data[measure_pair[1]])
    if i == 0:
        ax = axes[0][1]
        x_text = -2.9
        y_text = -2.2
    elif i == 1:
        ax == axes[0][2]
        x_text = -2.9
        y_text = -2.2
    elif i == 2:
        ax = axes[1][2]
        x_text = 0.01
        y_text = -3
    if i!=1:
        if p_val < 10**(-5):
            ax.text(x_text, y_text, r"$r_P \approx %s$; $P_P < 10^{-5}$"%(f'{r:.2f}'), fontsize = 7, zorder = np.inf)
        else:
            if (utl.myround(p_val) <0.001 and myround(p_val) > 0.0001):
                ax.text(x_text, y_text, r"$r_P \approx %s$; $P_P = %s$"%(f'{r:.2f}', f'{utl.myround(p_val):.4f}'), fontsize = 7, zorder = np.inf)
            else:
                ax.text(x_text, y_text, r"$r_P \approx %s$; $P_P = %s$"%(f'{r:.2f}', f'{utl.myround(p_val)}'), fontsize = 7, zorder = np.inf)
    else:
        x_text = 0.01
        y_text = -2.2
        axes[0,2].text(x_text, y_text, r"$r_P \approx %s$; $P_P < 10^{-5}$"%(f'{r:.2f}'), fontsize = 7, zorder = np.inf)


utl.save_plot(plot_settings, f"{__file__[:-3]}_measures", stimulus="movie")

# TODO: Get the p-values and plot them in the upper triangle of the matrics

fig0, axes = plt.subplots(3,3, figsize=(2*plot_settings["panel_width"],3.75))
fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1, wspace=0.1, hspace=0.33)

fr_measure = "log_fr"
isi_measure = "log_median_ISI"
cv_measure = "CV"

selection = data[cv_measure] < 5
data_len = len(selection)
filtered_data_len = sum(selection)
print('{} data points dropped by filter ({:.2f}% of data) for measure {}'.format(data_len - filtered_data_len,
                                                                             (1 - filtered_data_len / data_len) * 100,
                                                                             cv_measure))
data = data[selection]

# print(data[fr_measure].describe())
# print(data[isi_measure].describe())
# print(data[cv_measure].describe())

pairplot_measures_x = [fr_measure, isi_measure, cv_measure]
pairplot_measures_y = [C_measure, T_measure, R_measure]

lims = {fr_measure : (-1.01, 1.8),
        isi_measure : (-2.4, 0.7),
        cv_measure : (0.49, 4.9),
        C_measure : (-2.3, 0.97),
        T_measure : (-3.1, -0.3),
        R_measure : (-0.01, 0.41)}
ticks = {fr_measure : [-1., 0., 1.],
         isi_measure : [-2., -1., 0.],
         cv_measure : [1, 2, 3, 4],
         C_measure : [-2.0, -1.0, 0.0],
         T_measure : [-3.0, -2.0, -1.0],
         R_measure : [0, 0.1, 0.2, 0.3, 0.4]}
labels = {fr_measure : 'fir. rate (Hz)',
          isi_measure : 'median ISI',
          cv_measure : 'CV',
          C_measure : '$τ_{\mathregular{C}}$ (ms)',
          T_measure : '$τ_R$ (ms)',
          R_measure : r'$R_{\mathregular{tot}}$'}
fmts = {fr_measure : utl.format_label_log,
        isi_measure : utl.format_label_log,
        cv_measure : None,
        C_measure : utl.format_label_log_in_ms,
        T_measure : utl.format_label_log_in_ms,
        R_measure : None}

for i, x_measure in enumerate(pairplot_measures_x):
    for j, y_measure in enumerate(pairplot_measures_y):
            ax = axes[j][i]
            r , p_val = pearsonr(data[x_measure], data[y_measure])
            if y_measure == C_measure:
                y_text = -2.2
            elif y_measure == T_measure:
                y_text = -3.
            elif y_measure == R_measure:
                y_text = 0.3
            if x_measure == fr_measure:
                x_text = -0.9
            elif x_measure == isi_measure:
                x_text = - 2.2
            elif x_measure == cv_measure:
                x_text = 0.8

            if p_val < 10**(-5):
                ax.text(x_text, y_text, r"$r_P \approx %s$; $P_P < 10^{-5}$"%(f'{r:.2f}'), fontsize = 7, zorder = np.inf)
            else:
                if (utl.myround(p_val) <0.001 and utl.myround(p_val) > 0.0001):
                    ax.text(x_text, y_text, r"$r_P \approx %s$; $P_P = %s$"%(f'{r:.2f}', f'{utl.myround(p_val):.4f}'), fontsize = 7, zorder = np.inf)
                else:
                    ax.text(x_text, y_text, r"$r_P \approx %s$; $P_P = %s$"%(f'{r:.2f}', f'{utl.myround(p_val)}'), fontsize = 7, zorder = np.inf)
#session_id,unit,stimulus,stimulus_blocks,firing_rate_1,median_ISI_1,CV_1,firing_rate_2,median_ISI_2,CV_2,firing_rate,median_ISI,CV

utl.plot_pairplot(data, pairplot_measures_x, axes, lims, ticks, labels, fmts,
                  pairplot_measures_y=pairplot_measures_y, levels=4)

utl.save_plot(plot_settings, f"{__file__[:-3]}_stats", stimulus="movie")

# TODO: Get the p-values and plot them below the cloud for tau_R and tau_C, and above for Rtot.
