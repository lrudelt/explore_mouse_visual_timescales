import argparse
from bitsandbobs.plt import set_size, alpha_to_solid_on_bg


defined_measures = ["tau_C",
                    "tau_R",
                    "R_tot",
                    "firing_rate",
                    "median_ISI",
                    "CV"]
defined_stimuli = ["movie",
                   "spontaneous"]
parser = argparse.ArgumentParser()
parser.add_argument('measure', type=str, help=f'one of {defined_measures}')
parser.add_argument('stimulus', type=str, help=f'one of {defined_stimuli}, default: movie', nargs='?', default="movie")
parser.add_argument('--bo', dest='allen_bo', action='store_true',
                    help=f'use brain observatory data set for movie stimulus')
args = parser.parse_args()
if not args.measure in defined_measures:
    parser.print_help()
    exit()
if not args.stimulus in defined_stimuli:
    parser.print_help()
    exit()
if args.allen_bo:
    if not args.stimulus == "movie":
        parser.print_help()
        exit()

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

### settings

if args.allen_bo:
    analysis = 'allen_bo'
else:
    analysis = 'allen_fc'

center = 'median' # measure of central tendency
T_measure = 'tau_R'
R_measure = 'R_tot'
# C_measure = 'mre_tau'
C_measure = 'tau_two_timescales'

if args.allen_bo:
    stimuli = ['natural_movie_three']
elif args.stimulus == "movie":
    stimuli = ['natural_movie_one_more_repeats']
elif args.stimulus == "spontaneous":
    stimuli = ['spontaneous']
T0 = 0.03 # 30 ms
selected_structures = 'cortex'

# setup analysis
plot_settings = utl.get_default_plot_settings()
plot_settings['panel_width'] = 2.0
plot_settings['panel_height'] = 2.7
plot_settings['rcparams']['figure.figsize'] =  [plot_settings['panel_width'], plot_settings['panel_height']]
plt.rcParams.update(plot_settings['rcparams'])

stats_dir = dir_settings['stats_dir']

csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
if C_measure =='tau_two_timescales':
    mre_stats_file_name = "{}/{}_mre_statistics_Tmin_30_Tmax_10000.csv".format(stats_dir, analysis)
    # mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_10000_two_timescales.csv".format(stats_dir, analysis)
else:
    mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_2500.csv".format(stats_dir, analysis)

if args.allen_bo:
    pp_stats_file_name = None
else:
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

fig0, ax = plt.subplots()
fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1)

if args.allen_bo:
    if args.measure == "tau_C":
        lims = [0.23, 0.71]
        y_pos = 0.233
        measure = C_measure
    elif args.measure == "tau_R":
        lims = [0.029, 0.086]
        y_pos = 0.031
        measure = T_measure
    elif args.measure == "R_tot":
        lims = [0.059, 0.093]
        y_pos = 0.061
        measure = R_measure
elif args.stimulus == "movie":
    if args.measure == "tau_C":
        lims = [0.17, 0.52]
        y_pos = 0.18
        measure = C_measure
    elif args.measure == "tau_R":
        lims = [0.026, 0.076]
        y_pos = 0.031
        measure = T_measure
    elif args.measure == "R_tot":
        lims = [0.064, 0.094]
        y_pos = 0.066
        measure = R_measure
    elif args.measure == "firing_rate":
        lims = [2.45, 4.3]
        y_pos = 2.52
        measure = args.measure
    elif args.measure == "CV":
        lims = [1.83, 2.1]
        y_pos = 1.85
        measure = args.measure
    elif args.measure == "median_ISI":
        lims = [45, 100]
        y_pos = 52
        measure = args.measure
elif args.stimulus == "spontaneous":
    if args.measure == "tau_C":
        lims = [0.19, 0.53]
        y_pos = 0.205
        measure = C_measure
    elif args.measure == "tau_R":
        lims = [0.017, 0.076]
        y_pos = 0.021
        measure = T_measure
    elif args.measure == "R_tot":
        lims = [0.049, 0.074]
        y_pos = 0.051
        measure = R_measure

x_pos = -0.2

# elif stimulus == 'spontaneous':
#     lim_R = [0.049, 0.074]
#     lim_T = [0.017, 0.076]
#     lim_C = [0.17, 0.76]

#     x_pos_text = [-0.2, -0.2, -0.2]
#     y_pos_text = [0.21, 0.021, 0.051]

x = []
y = []
y_err = []
x_units = []
y_units = []
x_session = []
y_session = []

selection = utl.get_data_filter(data, measure)
data = data[selection]
sessions = np.unique(data['session_id'])

for structure in _structures:
    _x = structures_map[structure]['hierarchy_score']
    _y = utl.get_center(data[utl.df_filter(data,
                                           structures=structure)][measure].values,
                        center)
    _y_err = utl.get_sd(data[utl.df_filter(data,
    structures=structure)][measure].values,
    center)
    _y_CI_low, _y_CI_high = utl.get_CI_median(data[utl.df_filter(data, structures=structure)][measure].values)
    _y_units = data[utl.df_filter(data,
                                           structures=structure)][measure].values
    _x_units = np.zeros(len(_y_units)) + _x
    _y_session = []
    _x_session = []
    for session in sessions:
        session_median = utl.get_center(data[utl.df_filter(data,
                                               structures=structure, sessions = [session])][measure].values, center)
        if not np.isnan(session_median):
            _y_session += [session_median]
            _x_session += [_x]

    if measure == 'median_ISI':
        _y = _y * 1000 # from s to ms
        _y_err = _y_err * 1000
        _y_CI_low = _y_CI_low * 1000
        _y_CI_high = _y_CI_high * 1000

    ax.plot([_x, _x],
            # [_y - _y_err, _y + _y_err],
            [_y_CI_low, _y_CI_high],
            lw=2,
            color=structures_map[structure]['color'],
    )
    ax.plot(_x,
            _y,
            'o',
            color="white",
            mec=structures_map[structure]['color']
    )


    if structure in structures_cortex:
        x += [_x]
        y += [_y]
        y_err += [_y_err]
        x_units = np.append(x_units, _x_units)
        y_units = np.append(y_units, _y_units)
        x_session += _x_session
        y_session += _y_session

ax.set_xticks([-0.5, -0.25, 0, 0.25])
utl.make_plot_pretty(ax)
ax.set_ylim(lims)

slope,intercept,r,p,std = st.linregress(x, y)
x2 = np.linspace(min(x), max(x), 10)

ax.plot(x2, x2*slope+intercept, ls='--', zorder=1, color=alpha_to_solid_on_bg("black", alpha=0.5))

r_s, p_s = st.spearmanr(x, y)
r_p, p_p = st.pearsonr(x, y)

# Regression for data grouped by session
slope_session ,intercept_session, r , p ,std = st.linregress(x_session, y_session)
x2 = np.linspace(min(x), max(x), 10)

r_s_session, p_s_session = st.spearmanr(x_session, y_session)
r_p_session, p_p_session = st.pearsonr(x_session, y_session)

# Regression for data grouped by units
slope_units ,intercept_units, r , p ,std = st.linregress(x_units, y_units)
x2 = np.linspace(min(x), max(x), 10)

r_s_units, p_s_units = st.spearmanr(x_units, y_units)
r_p_units, p_p_units = st.pearsonr(x_units, y_units)

print("Structure medians: ", "r=", r_s," ", "p=", " ", p_s)
print("Session medians: ", "r=", r_s_session," ", "p=", " ", p_s_session)
print("Individual units: ", "r=", r_s_units," ", "p=", " ", p_s_units)

text = ""
if p_p < 0.0005:
    text += '$r_P$ = {:.2f}; '.format(r_p) + '$P_P < 10^{-3}$\n'
else:
    text += '$r_P$ = {:.2f}; $P_P$ = {:.3f}\n'.format(r_p, p_p)
if p_s < 0.0005:
    text += '$r_S$ = {:.2f}; '.format(r_s) + '$P_S$ < 10^{-3}'
else:
    text += '$r_S$ = {:.2f}; $P_S$ = {:.3f}'.format(r_s,
                                                    p_s)

# + str(np.around(pow(r_p,1),2)) + '' + str(np.around(p_p,6)) + '\n' + \
#         '$r_S$ = ' + str(np.around(pow(r_s,1),2)) + '; $P_S$ = ' + str(np.around(p_s,6))
ax.text(x_pos, y_pos,
        text, fontsize=8)

utl.format_ax(ax, measure, 'y', set_ticks=False, tight_layout=False)

ax.set_xlabel('hierarchy score')

# legend_elements = [Line2D([0], [0], marker='s', color=structures_map[structure]['color'],
#                           label=structures_map[structure]['name'],
#                           markerfacecolor=structures_map[structure]['color'],
#                           ms=10,
#                           linewidth=0)
#                    for structure in _structures]
# legend = axes[2].legend(handles=legend_elements,
#                         fancybox=False,
#                         loc="lower right",
#                         bbox_to_anchor=(1.55, 0.10))
# frame = legend.get_frame()
# frame.set_facecolor('0.9')
# frame.set_edgecolor('0.9')

utl.save_plot(plot_settings, f"allen_hierarchy", allen_bo=args.allen_bo, stimulus=args.stimulus, measure=args.measure)
