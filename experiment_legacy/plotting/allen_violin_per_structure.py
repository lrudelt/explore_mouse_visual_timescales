from sys import exit, path
from os.path import realpath, dirname, isfile, isdir
import os
import yaml

SCRIPT_DIR = dirname(realpath(__file__))
with open('{}/dirs.yaml'.format(SCRIPT_DIR), 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as st

path.insert(1, "../../allen_src/")
import analysis_utils as utl

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

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

stimuli = ['natural_movie_one_more_repeats', 'spontaneous']
T0 = 0.03 # 30 ms
selected_structures = 'cortex+thalamus'

plot_stars = False # plot stars or p-values for tests

# setup analysis
plot_settings = utl.get_default_plot_settings()
plt.rcParams.update(plot_settings['rcparams'])

stats_dir = dir_settings['stats_dir']

csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
# mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_2500.csv".format(stats_dir, analysis)
mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_10000_two_timescales.csv".format(stats_dir, analysis)

structures = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
structures_map = utl.get_structures_map()
structures_cortex = [structure for structure in structures
                     if structures_map[structure]['parent_structure'] == 'cortex']
stimuli_map = utl.get_stimuli_map(bo_analysis=(analysis == 'allen_bo'))

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

# plot


## stimulus

fig0, axes = plt.subplots(len(structures), 3, figsize=(1.4 * plot_settings['textwidth'], 14))
fig0.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, wspace = 0.5, hspace = 0.33)

measures = [C_measure, T_measure, R_measure]

lim_C = [-2.01, 2.6]
lim_T = [-3.01, 1.6]
lim_R = [0, 0.65]

axes[0][0].set_title('intrinsic timescale')
axes[0][1].set_title('information timescale')
axes[0][2].set_title('predictability')

color_palette = sns.color_palette().as_hex()
hex_structures_colors = [color_palette[4],
                         color_palette[0],
                         color_palette[9],
                         color_palette[8],
                         color_palette[1],
                         color_palette[3],
                         color_palette[6],
                         color_palette[2]]


for axes_row, structure, hex_color in zip(axes,
                                          structures,
                                          hex_structures_colors):

    for ax, measure, lims in zip(axes_row,
                                 measures,
                                 [lim_C, lim_T, lim_R]):
        
        selection = utl.get_data_filter(data, measure)
        selection &= utl.df_filter(data,
                                   structures=structure)
        _data = data[selection]
        movie_data, sp_data, diff_data, valid_units = utl.get_data_for_stimulus_conditions(_data, stimuli, measure)

        palette = {'natural_movie_one_more_repeats': hex_color,
                   'spontaneous': hex_color}

        utl.fancy_violins(
            _data.loc[_data["unit"].isin(valid_units),:],
            'stimulus',
            measure,
            ax=ax,
            num_swarm_points=200,
            same_points_per_swarm=True,
            replace=False,
            palette=palette)
        
        ax.set_xticklabels([stimuli_map[stimulus]['name'] for stimulus in stimuli])

        def y_pos(k):
            return lims[0] + k * (lims[1] - lims[0])

        if 'log' in measure:
            d = 10**utl.get_center(sp_data, center) \
                - 10**utl.get_center(movie_data, center)
            med_diff = d
            med_diff_rel = d / 10**utl.get_center(movie_data, center) * 100
        else:
            d = utl.get_center(sp_data, center) \
                - utl.get_center(movie_data, center)
            med_diff = d
            med_diff_rel = d / utl.get_center(movie_data, center) * 100

        if 'T' in measure or 'tau' in measure:
            d *= 1000
            ax.text(1.1, #0.9*lims[1], 
                    y_pos(0.95), "d = {:.0f} ms".format(d),
                    color='k')
        else:
            ax.text(1.1, #0.9*lims[1], 
                    y_pos(0.95), "d = {:.2f}".format(d), color='k')

        if measure == T_measure:
            # y = y_pos(0.75)
            y = y_pos(0.1)
        elif measure == C_measure:
            y = y_pos(0.1)
            # y = y_pos(0.83)
        elif measure == R_measure:
            y = y_pos(0.54)
        else:
            y = max([max(d) for d in (movie_data,sp_data)]) + -0.05 * (lims[1] - lims[0])

        ax.text(0.3, y, "N={}".format(len(movie_data)), ha='center', va='top', color='k')


        # MW_u, MW_p = st.mannwhitneyu(movie_data, sp_data, alternative='two-sided')
        p_wilcoxon = st.wilcoxon(diff_data).pvalue
        bonferroni_correction = 1/(len(axes)*3) # we check len(axes) * 3 relations in total 
        if p_wilcoxon < 0.05*bonferroni_correction:
            x1, x2 = 0, 1
            w = 0.03 * (lims[1] - lims[0])

            if measure == 'log_tau_R':
                y = min(y_pos(0.9), max([max(d) for d in (movie_data, sp_data)]) + 1.5 * w)
            elif measure == 'log_mre_tau':
                y = min(y_pos(0.8), max([max(d) for d in (movie_data, sp_data)]) + 1.5 * w)
            else:
                y = max([max(d) for d in (movie_data, sp_data)]) + 1.5 * w

            if plot_stars:
                if p_wilcoxon < 0.001:
                    num_stars = 3
                elif p_wilcoxon < 0.01:
                    num_stars = 2
                else:
                    num_stars = 1
                text = "*"*num_stars
                text_y_pos = y+2*w
            else:
                if p_wilcoxon < 0.0005:
                    # text = '$p < 10^{-3}$'
                    text = '$p < 0.001$'
                else:
                    text = '$p$ = {:.3f}\n'.format(p_wilcoxon)
                text_y_pos = y+7*w

            # ax.plot([x1, x1, x2, x2], [y, y+w, y+w, y], lw=1.5, c='k')
            ax.plot([x1, x2], [y+w, y+w], lw=1.5, c='k')
            ax.text((x1+x2)/2, text_y_pos, text, ha='center', va='top', color='k')

        if measure == T_measure:
            ax.set_yticks([-3.0, -2.0, -1.0, 0.0])
        if measure == C_measure:
            ax.set_yticks([-2.0, -1.0, 0.0, 1.0])
            
        utl.format_ax(ax, measure, 'y', set_ticks=False, tight_layout=True)
        ax.set_ylim(lims)
        utl.make_plot_pretty(ax)

    axes_row[0].text(axes_row[0].get_xlim()[0] - 0.65 * (axes_row[0].get_xlim()[1]
                                                         - axes_row[0].get_xlim()[0]),
                     lim_C[0] + 0.45 * (lim_C[1] - lim_C[0]),
                     structures_map[structure]['name'], weight='bold')
    

utl.save_plot(plot_settings, f"{__file__[:-3]}_stimulus")



## rf v no rf

# fig0, axes = plt.subplots(len(structures), 3, figsize=(1.4 * plot_settings['textwidth'], 14))
# fig0.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, wspace = 0.5, hspace = 0.33)

# measures = [C_measure, T_measure, R_measure]

# lim_C = [-2.01, 2.6]
# lim_T = [-3.01, 1.6]
# lim_R = [0, 0.65]

# axes[0][0].set_title('intrinsic timescale')
# axes[0][1].set_title('information timescale')
# axes[0][2].set_title('predictability')

# color_palette = sns.color_palette().as_hex()
# hex_structures_colors = [color_palette[4],
#                          color_palette[0],
#                          color_palette[9],
#                          color_palette[8],
#                          color_palette[1],
#                          color_palette[3],
#                          color_palette[6],
#                          color_palette[2]]



# for axes_row, structure, hex_color in zip(axes,
#                                           structures,
#                                           hex_structures_colors):

#     for ax, measure, lims in zip(axes_row,
#                                  measures,
#                                  [lim_C, lim_T, lim_R]):
        
#         selection = utl.get_data_filter(data, measure)
#         selection &= utl.df_filter(data,
#                                    structures=structure,
#                                    stimuli='natural_movie_one_more_repeats',
#                                    sign_rf=[True, False])
#         _data = data[selection]
#         movie_data, sp_data, diff_data, valid_units = utl.get_data_for_stimulus_conditions(_data, stimuli, measure)

#         palette = {True: hex_color,
#                    False: hex_color}

#         utl.fancy_violins(
#             _data.loc[_data["unit"].isin(valid_units),:],
#             'sign_rf',
#             measure,
#             ax=ax,
#             num_swarm_points=200,
#             same_points_per_swarm=True,
#             replace=False,
#             palette=palette)
        
#         ax.set_xticklabels(['no rf', 'rf'])

#         def y_pos(k):
#             return lims[0] + k * (lims[1] - lims[0])

#         if 'log' in measure:
#             d = 10**utl.get_center(sp_data, center) \
#                 - 10**utl.get_center(movie_data, center)
#             med_diff = d
#             med_diff_rel = d / 10**utl.get_center(movie_data, center) * 100
#         else:
#             d = utl.get_center(sp_data, center) \
#                 - utl.get_center(movie_data, center)
#             med_diff = d
#             med_diff_rel = d / utl.get_center(movie_data, center) * 100

#         if 'T' in measure or 'tau' in measure:
#             d *= 1000
#             ax.text(1.1, #0.9*lims[1], 
#                     y_pos(0.95), "d = {:.0f} ms".format(d),
#                     color='k')
#         else:
#             ax.text(1.1, #0.9*lims[1], 
#                     y_pos(0.95), "d = {:.2f}".format(d), color='k')

#         if measure == T_measure:
#             # y = y_pos(0.75)
#             y = y_pos(0.1)
#         elif measure == C_measure:
#             y = y_pos(0.1)
#             # y = y_pos(0.83)
#         elif measure == R_measure:
#             y = y_pos(0.54)
#         else:
#             y = max([max(d) for d in (movie_data, sp_data)]) + -0.05 * (lims[1] - lims[0])

#         ax.text(0.3, y, "N={}".format(len(movie_data)), ha='center', va='top', color='k')


#         # MW_u, MW_p = st.mannwhitneyu(movie_data, sp_data, alternative='two-sided')
#         p_wilcoxon = st.wilcoxon(diff_data).pvalue
#         if p_wilcoxon < 0.05:
#             x1, x2 = 0, 1
#             w = 0.03 * (lims[1] - lims[0])

#             if measure == 'log_tau_R':
#                 y = min(y_pos(0.9), max([max(d) for d in (movie_data, sp_data)]) + 1.5 * w)
#             elif measure == 'log_mre_tau':
#                 y = min(y_pos(0.8), max([max(d) for d in (movie_data, sp_data)]) + 1.5 * w)
#             else:
#                 y = max([max(d) for d in (movie_data, sp_data)]) + 1.5 * w

#             if plot_stars:
#                 if p_wilcoxon < 0.001:
#                     num_stars = 3
#                 elif p_wilcoxon < 0.01:
#                     num_stars = 2
#                 else:
#                     num_stars = 1
#                 text = "*"*num_stars
#                 text_y_pos = y+2*w
#             else:
#                 if p_wilcoxon < 0.0005:
#                     # text = '$p < 10^{-3}$'
#                     text = '$p < 0.001$'
#                 else:
#                     text = '$p$ = {:.3f}\n'.format(p_wilcoxon)
#                 text_y_pos = y+7*w

#             # ax.plot([x1, x1, x2, x2], [y, y+w, y+w, y], lw=1.5, c='k')
#             ax.plot([x1, x2], [y+w, y+w], lw=1.5, c='k')
#             ax.text((x1+x2)/2, text_y_pos, text, ha='center', va='top', color='k')

#         if measure == T_measure:
#             ax.set_yticks([-3.0, -2.0, -1.0, 0.0])
#         if measure == C_measure:
#             ax.set_yticks([-2.0, -1.0, 0.0, 1.0])
            
#         utl.format_ax(ax, measure, 'y', set_ticks=False, tight_layout=True)
#         ax.set_ylim(lims)
#         utl.make_plot_pretty(ax)

#     axes_row[0].text(axes_row[0].get_xlim()[0] - 0.65 * (axes_row[0].get_xlim()[1]
#                                                          - axes_row[0].get_xlim()[0]),
#                      lim_C[0] + 0.45 * (lim_C[1] - lim_C[0]),
#                      structures_map[structure]['name'], weight='bold')
    

# utl.save_plot(plot_settings, f"{__file__[:-3]}_rf")
