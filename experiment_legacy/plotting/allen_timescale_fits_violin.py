import argparse

from sys import exit, path
from os.path import realpath, dirname, isfile, isdir
import os
import yaml

SCRIPT_DIR = dirname(realpath(__file__))
SCRIPT_DIR = "."
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


def get_df_timescales_different_fits(data):
    # get timescales for single timescale fit with offset that where selected in the model comparison
    tau_offset_accepted = data["tau_offset"].values
    bic_passed_offset_accepted = data["bic_passed_offset"].values
    tau_offset_accepted_type = ["tau_offset_accepted" for tau in tau_offset_accepted]
    # tau_offset_accepted = data["tau_offset"][data["tau_offset"] == data["tau_two_timescales"]].values
    # bic_passed_offset_accepted = data["bic_passed_offset"][data["tau_offset"] == data["tau_two_timescales"]].values
    # tau_offset_accepted_type = ["tau_offset_accepted" for tau in tau_offset_accepted]

    # tau_offset_rejected = data["tau_offset"][data["tau_offset"] != data["tau_two_timescales"]].values
    # bic_passed_offset_rejected = data["bic_passed_offset"][data["tau_offset"] != data["tau_two_timescales"]].values
    # tau_offset_rejected_type = ["tau_offset_rejected" for tau in tau_offset_rejected]

    # get timescales (selected and rejected) for two timescale fit that where selected in the model comparison
    tau_two_timescales = data["tau_two_timescales"].values
    bic_passed_two_timescales = data["bic_passed_two_timescales"].values
    # tau_two_timescales = data["tau_two_timescales"][data["tau_offset"] != data["tau_two_timescales"]].values
    # bic_passed_two_timescales = data["bic_passed_two_timescales"][data["tau_offset"] != data["tau_two_timescales"]].values
    tau_two_timescales_type = ["tau_two_timescales" for tau in tau_two_timescales]

    tau_rejected = data["tau_rejected"].values
    tau_rejected_type = ["tau_rejected" for tau in tau_rejected]

    print("Proportion of higher timescale accepted: ", len(tau_rejected[tau_rejected<tau_two_timescales])/float(len(tau_rejected)))

    # tau = np.concatenate((tau_offset_accepted, tau_offset_rejected, tau_two_timescales, tau_rejected))
    tau = np.concatenate((tau_offset_accepted, tau_two_timescales, tau_rejected))
    # timescale_type = np.concatenate((tau_offset_accepted_type, tau_offset_rejected_type, tau_two_timescales_type, tau_rejected_type)).astype(str)
    timescale_type = np.concatenate((tau_offset_accepted_type, tau_two_timescales_type, tau_rejected_type)).astype(str)
    # bic_passed = np.concatenate((bic_passed_offset_accepted,bic_passed_offset_rejected,bic_passed_two_timescales,bic_passed_two_timescales))
    bic_passed = np.concatenate((bic_passed_offset_accepted, bic_passed_two_timescales, bic_passed_two_timescales))
    timescale_df = pd.DataFrame({"timescale_type":timescale_type, "log_mre_tau": np.log10(tau), "mre_bic_passed": bic_passed})
    return timescale_df

### settings

analysis = 'allen_fc'

center = 'median' # measure of central tendency
T_measure = 'log_tau_R'
R_measure = 'R_tot'
C_measure = 'log_mre_tau'

# stimuli = ['natural_movie_one_more_repeats', 'spontaneous']
stimuli = ['natural_movie_one_more_repeats']

T0 = 0.03 # 30 ms
selected_structures = 'cortex+thalamus'

plot_stars = False # plot stars or p-values for tests

# setup analysis
plot_settings = utl.get_default_plot_settings()
plt.rcParams.update(plot_settings['rcparams'])
plot_settings['panel_width'] = 0.52 * plot_settings['textwidth']
plot_settings['panel_height'] = 2.9
plot_settings['rcparams']['figure.figsize'] =  [plot_settings['panel_width'], plot_settings['panel_height']]
plot_settings['panel_size'] =  [plot_settings['panel_width'], plot_settings['panel_height']]

stats_dir = dir_settings['stats_dir']

csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_10000_two_timescales.csv".format(stats_dir, analysis)

structures = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
structures_map = utl.get_structures_map()
structures_cortex = [structure for structure in structures
                     if structures_map[structure]['parent_structure'] == 'cortex']
stimuli_map = utl.get_stimuli_map(bo_analysis=(analysis == 'allen_bo'))

### import data

data = utl.get_analysis_data(csv_file_name, analysis,
                             mre_stats_file_name=mre_stats_file_name)

# TODO: Create a new data frame with two dict entries, 'timescale_type' (one of tau_offset, tau_two_timescales, tau_rejected) and the value 'tau'. Make sure to only add the values which have been selected during model selection

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


timescale_data = get_df_timescales_different_fits(data)
# plot

fig0, ax = plt.subplots(figsize=plot_settings["panel_size"])
fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1)

lims = [-2.4, 1.9]
ax.set_title('intrinsic timescale distributions')
measure = C_measure

selection = timescale_data['mre_bic_passed'].values.astype(bool)
timescale_data = timescale_data[selection]
#
# bp_data = [data[utl.df_filter(data, stimuli=s)][measure].values
#            for s in stimuli]

# if 'log' in measure:
#     print(args.measure, "natural movie median: ", 10**utl.get_center(bp_data[0], center), "spontaneous activity median: ", 10**utl.get_center(bp_data[1], center))
# else:
#     print(args.measure, "natural movie median: ", utl.get_center(bp_data[0], center), "spontaneous activity median: ", utl.get_center(bp_data[1], center))

utl.fancy_violins(
    timescale_data,
    'timescale_type',
    measure,
    ax=ax,
    num_swarm_points=400,
    same_points_per_swarm=True,
    replace=False,
    palette=None)
#
# def y_pos(k):
#     return lims[0] + k * (lims[1] - lims[0])
#
# if 'log' in measure:
#     d = 10**utl.get_center(bp_data[1], center) \
#         - 10**utl.get_center(bp_data[0], center)
#     med_diff = d
#     med_diff_rel = d / 10**utl.get_center(bp_data[0], center) * 100
# else:
#     d = utl.get_center(bp_data[1], center) \
#         - utl.get_center(bp_data[0], center)
#     med_diff = d
#     med_diff_rel = d / utl.get_center(bp_data[0], center) * 100
# #print(sign_rf, structure, measure, med_diff_rel)
#
# if 'T' in measure or 'tau' in measure:
#     d *= 1000
#     ax.text(1.1, #0.9*lims[1],
#             y_pos(0.95), "d = {:.0f} ms".format(d),
#             color='k')
# else:
#     ax.text(1.1, #0.9*lims[1],
#             y_pos(0.95), "d = {:.2f}".format(d), color='k')
#
# if measure == T_measure:
#     y = y_pos(0.08)
# elif measure == C_measure:
#     # y = y_pos(0.1)
#     y = y_pos(0.08)
# elif measure == R_measure:
#     y = y_pos(0.08)
# else:
#     y = max([max(d) for d in bp_data]) + -0.05 * (lims[1] - lims[0])
#
# for x, d in zip([0, 1], bp_data):
#     ax.text(x + 0.43, y, "N={}".format(len(d)), ha='center', va='top', color='k')

#
# MW_u, MW_p = st.mannwhitneyu(bp_data[0], bp_data[1], alternative='two-sided')
#
# if MW_p < 0.05:
#     x1, x2 = 0, 1
#     w = 0.05 * (lims[1] - lims[0])
#
#     if measure == 'log_tau_R':
#         y = min(y_pos(0.8), max([max(d) for d in bp_data]) + 1.5 * w)
#     elif measure == 'log_mre_tau':
#         y = min(y_pos(0.8), max([max(d) for d in bp_data]) + 1.5 * w)
#     else:
#         # y = max([max(d) for d in bp_data]) + 1.5 * w
#         y = min(y_pos(0.8), max([max(d) for d in bp_data]) + 1.5 * w)
#
#     if plot_stars:
#         if MW_p < 0.001:
#             num_stars = 3
#         elif MW_p < 0.01:
#             num_stars = 2
#         else:
#             num_stars = 1
#         text = "*"*num_stars
#         text_y_pos = y+2*w
#     else:
#         if MW_p < 0.0005:
#             # text = '$p < 10^{-3}$'
#             text = '$p < 0.001$'
#         else:
#             text = '$p$ = {:.3f}\n'.format(MW_p)
#         text_y_pos = y+2.5*w

    # # ax.plot([x1, x1, x2, x2], [y, y+w, y+w, y], lw=1.5, c='k')
    # ax.plot([x1, x2], [y+w, y+w], lw=1.5, c='k')
    # ax.text((x1+x2)/2, text_y_pos, text, ha='center', va='top', color='k')


if measure == T_measure:
    ax.set_yticks([-3.0, -2.0, -1.0, 0.0])
if measure == C_measure:
    # ax.set_yticks([-3.0, -2.0, -1.0, 0.0, 1.0])
    ax.set_yticks([-2.0, -1.0, 0.0, 1.0])
if measure == R_measure:
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])

utl.format_ax(ax, measure, 'y', set_ticks=False, tight_layout=True)
ax.set_ylim(lims)
utl.make_plot_pretty(ax)
ax.set_xlabel("")
ax.set_xticklabels(["single \n timescale \n + offset","two \n timescales \n (selected)", "two \n timescales \n (rejected)"])
# ax.set_xticklabels([stimuli_map[stimulus]['name'] for stimulus in stimuli])

utl.save_plot(plot_settings, f"{__file__[:-3]}", measure=measure)
