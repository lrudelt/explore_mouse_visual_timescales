# figure 8

import argparse

defined_measures = ["tau_C",
                    "tau_R"]
defined_stimuli = ["movie",
                   "spontaneous"]
defined_fit_functions = ["single_timescale", "two_timescales"]
parser = argparse.ArgumentParser()
parser.add_argument('measure', type=str, help=f'one of {defined_measures}')
parser.add_argument('stimulus', type=str, help=f'one of {defined_stimuli}, default: movie', nargs='?', default="movie")
parser.add_argument('fit_function', type=str, help=f'one of {defined_fit_functions}, default: single_timescale', nargs='?', default="single_timescale")
parser.add_argument('--Tmax500', dest='T_max500', action='store_true',
                    help=f'set T_max to 500ms for measure=tau_C')

args = parser.parse_args()
if not args.measure in defined_measures:
    parser.print_help()
    exit()
if not args.stimulus in defined_stimuli:
    parser.print_help()
    exit()
if not args.fit_function in defined_fit_functions:
    parser.print_help()
    exit()
if args.T_max500:
    if not args.measure == "tau_C":
        parser.print_help()
        exit()

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
import scipy.stats as st

path.insert(1, "../../allen_src/")
import analysis_utils as utl

from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors

### settings

analysis = 'allen_fc'

center = 'median' # measure of central tendency
T_measure = 'tau_R'
R_measure = 'R_tot'
if args.fit_function == "single_timescale":
    C_measure = 'mre_tau'
elif args.fit_function == "two_timescales":
    C_measure = "tau_two_timescales"

if args.stimulus == "movie":
    stimuli = ['natural_movie_one_more_repeats']
elif args.stimulus == "spontaneous":
    stimuli = ['spontaneous']
# T0 = 0.03 # 30 ms
if args.measure == "tau_C":
    measure = C_measure
    T0s = [.005, .01, .015, .03]
    if args.T_max500:
        T_max = 0.5
    else:
        # T_max = 2.5
        T_max = 10.
if args.measure == "tau_R":
    measure = T_measure
    T0s = [0.0, .001, .002, .005, .01, .03]
    T_max = 5
selected_structures = 'cortex'

# setup analysis
plot_settings = utl.get_default_plot_settings()
plt.rcParams.update(plot_settings['rcparams'])

stats_dir = dir_settings['stats_dir']

structures = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
structures_map = utl.get_structures_map()
structures_cortex = [structure for structure in structures
                     if structures_map[structure]['parent_structure'] == 'cortex']

### import data

csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)

if (selected_structures == 'cortex+thalamus') :
    _structures = structures
elif (selected_structures == 'cortex') :
    _structures = structures_cortex
else:
    print('unknown structures selected')
    exit()

T0_data = {}

for T0 in T0s:
    _csv_file_name = "{}/{}_statistics_T0_{:02d}.csv".format(stats_dir, analysis, int(T0 * 1000))
    if args.fit_function == "single_timescale":
        _mre_stats_file_name = "{}/{}_mre_statistics_T0_{:02d}_Tmax_{:02d}.csv".format(stats_dir, analysis, int(T0 * 1000), int(T_max * 1000))
    elif args.fit_function == "two_timescales":
        _mre_stats_file_name = "{}/{}_mre_statistics_T0_{:02d}_Tmax_{:02d}_two_timescales.csv".format(stats_dir, analysis, int(T0 * 1000), int(T_max * 1000))

    if not isfile(_csv_file_name):
        if not measure == T_measure:
            _csv_file_name = csv_file_name
        else:
            print('error, {} not found'.format(_csv_file_name))
            exit()

    if not isfile(_mre_stats_file_name):
        if not measure == C_measure:
            _mre_stats_file_name = None
        else:
            print('error, {} not found'.format(_mre_stats_file_name))
            exit()

    _data = utl.get_analysis_data(_csv_file_name, analysis,
                                  mre_stats_file_name=_mre_stats_file_name)

    # make sure data are as expected

    if measure == C_measure:
        _data = _data[utl.df_filter(_data, structures=_structures, stimuli=stimuli, tmin=T0)]
    else:
        _data = _data[utl.df_filter(_data, structures=_structures, stimuli=stimuli, T0=T0)]
    try:
        num_neurons = utl.get_expected_number_of_neurons(analysis,
                                                         selected_structures,
                                                         stimuli)
        assert np.isclose(len(_data), num_neurons, atol=100)
    except:
        print(f'number of neurons is {len(_data)}, expected {num_neurons}')
        exit()

    selection = utl.get_data_filter(_data, measure)
    T0_data[T0] = _data[selection]

# plot

if args.stimulus == "movie":
    if args.T_max500:
        lims = [0.07, 0.23]
        y_pos_text = 0.075
    elif args.measure == "tau_C":
        lims = [0.18, 0.63]
        y_pos_text = 0.21
    elif args.measure == "tau_R":
        lims = [0.027, 0.076]
        y_pos_text = 0.031
elif args.stimulus == "spontaneous":
    if args.T_max500:
        lims = [0.039, 0.163]
        y_pos_text = 0.045
    elif args.measure == "tau_C":
        lims = [0.18, 0.63]
        y_pos_text = 0.21
    elif args.measure == "tau_R":
        lims = [0.027, 0.076]
        y_pos_text = 0.031



fig2, axes = plt.subplots(1, len(T0s), figsize=(1.9*plot_settings['textwidth'],3), sharey=True)
fig2.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, wspace = 0.1)

for ax, T0, x_pos, y_pos in zip(axes,
                                T0s,
                                [-0.2]*len(T0s),
                                [y_pos_text]*len(T0s)):
    _data = T0_data[T0]

    ax.set_title(r'$T_{\mathrm{min}} = %d\,\mathrm{ms}$'%(int(T0 * 1000)))
    # ax.set_title(f'[{int(T0 * 1000)}ms, {int(T_max * 1000)}ms]')

    x = []
    y = []
    y_err = []

    for structure in _structures:
        _x = structures_map[structure]['hierarchy_score']
        _y = utl.get_center(_data[utl.df_filter(_data,
                                                structures=structure)][measure].values,
                            center)
        _y_err = utl.get_sd(_data[utl.df_filter(_data,
                                                structures=structure)][measure].values,
                            center),
        _y_CI_low, _y_CI_high = utl.get_CI_median(_data[utl.df_filter(_data, structures=structure)][measure].values)
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

    ax.set_xticks([-0.5, -0.25, 0, 0.25])
    utl.make_plot_pretty(ax)
    ax.set_ylim(lims)

    slope,intercept,r,p,std = st.linregress(x, y)
    x2 = np.linspace(min(x), max(x), 10)

    ax.plot(x2, x2*slope+intercept, '--k', alpha=0.5)

    r_s, p_s = st.spearmanr(x, y)
    r_p, p_p = st.pearsonr(x, y)

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

utl.format_ax(axes[0], measure, 'y', set_ticks=False, tight_layout=True)


for ax in axes:
    ax.set_xlabel('hierarchy score')

legend_elements = [Line2D([0], [0], marker='s', color=structures_map[structure]['color'],
                          label=structures_map[structure]['name'],
                          markerfacecolor=structures_map[structure]['color'],
                          ms=10,
                          linewidth=0)
                   for structure in _structures]
legend = axes[len(T0s) - 1].legend(handles=legend_elements,
                                   fancybox=False,
                                   loc="lower right",
                                   bbox_to_anchor=(1.55, 0.10))
frame = legend.get_frame()
frame.set_facecolor('0.9')
frame.set_edgecolor('0.9')

suffix = ""
if args.T_max500:
    suffix = "_Tmax_500"

suffix2 = ""
if args.fit_function == "two_timescales":
    suffix2 = "_two_timescales"

utl.save_plot(plot_settings, f"{__file__[:-3]}{suffix}{suffix2}", stimulus=args.stimulus, measure=args.measure)
