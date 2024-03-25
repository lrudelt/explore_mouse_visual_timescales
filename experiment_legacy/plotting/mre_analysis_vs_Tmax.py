
from sys import exit, argv, path
from os.path import realpath, dirname, isfile, isdir, exists
import os
import yaml

SCRIPT_DIR = dirname(realpath(__file__))
with open('{}/dirs.yaml'.format(SCRIPT_DIR), 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)
# with open('./dirs.yaml', 'r') as dir_settings_file:
#     dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)

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
import seaborn as sns

plot_settings = utl.get_default_plot_settings()
plt.rcParams.update(plot_settings['rcparams'])
color = sns.color_palette().as_hex()[0] # same blue as for the violin plots

analysis = 'allen_fc'
selected_structures = 'cortex'
stimuli = ['natural_movie_one_more_repeats']

fit_selection_criterion = 'bic'
stats_dir = dir_settings['stats_dir']
stats_file_name = '{}/tau_C_hierarchy_vs_Tmax_{}_stats.csv'.format(stats_dir, fit_selection_criterion)
stats_file_name_two_timescales = '{}/tau_C_hierarchy_vs_Tmax_{}_stats_two_timescales.csv'.format(stats_dir, fit_selection_criterion)

structures = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
structures_map = utl.get_structures_map()
structures_cortex = [structure for structure in structures
                     if structures_map[structure]['parent_structure'] == 'cortex']

# make sure data as expected
if (selected_structures == 'cortex+thalamus') :
    _structures = structures
elif (selected_structures == 'cortex') :
    _structures = structures_cortex
else:
    print('unknown structures selected')
    exit()

Tmax_list = [500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 7500, 10000, 15000, 20000]
lims = [0.17, 0.62]
y_pos = 0.21
measure = 'mre_tau'
center = 'median'
x_pos = -0.2
csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
y_measure = argv[1]
if y_measure == 'tau_C':
    y_measure = 'tau_C_median'
# y_measure = 'r_p' # r_p, p_p, tau_C

# Load data
if exists(stats_file_name):
    stats = pd.read_csv(stats_file_name)
else:
    Tmax_arr = []
    T0_arr = []
    r_p_arr = []
    p_p_arr = []
    tau_C_median_arr = []
    tau_C_median_err_arr = []
    for Tmax in Tmax_list:
        for T0 in ['05', '30']:
            # fig0, ax = plt.subplots()
            # fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1)
            # ax.set_title('intrinsic timescale')
            mre_stats_file_name = "{}/{}_mre_statistics_T0_{}_Tmax_{}.csv".format(stats_dir, analysis, T0, Tmax)
            data = utl.get_analysis_data(csv_file_name, analysis,
                                 mre_stats_file_name=mre_stats_file_name, fit_selection_criterion = fit_selection_criterion)
            data = data[utl.df_filter(data, structures=_structures, stimuli=stimuli)]
            try:
                num_neurons = utl.get_expected_number_of_neurons(analysis,
                                                                 selected_structures,
                                                                 stimuli)
                assert np.isclose(len(data), num_neurons, atol=100)
            except:
                print(f'number of neurons is {len(data)}, expected {num_neurons}')
                exit()
            # Once fpr tje
            x = []
            y = []
            y_err = []
            x_units = []
            y_units = []
            x_session = []
            y_session = []
            all = []

            selection = utl.get_data_filter(data, measure)
            data = data[selection]
            sessions = np.unique(data['session_id'])
            for structure in _structures:
                _x = structures_map[structure]['hierarchy_score']
                _y = utl.get_center(data[utl.df_filter(data,
                                                       structures=structure)][measure].values, center)
                _y_err = utl.get_sd(data[utl.df_filter(data,
                structures=structure)][measure].values,
                center)
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

                # ax.plot([_x, _x],
                #         [_y - _y_err, _y + _y_err],
                #         lw=2,
                #         color=structures_map[structure]['color'],
                # )
                # ax.plot(_x,
                #         _y,
                #         'o',
                #         color="white",
                #         mec=structures_map[structure]['color']
                # )

                if structure in structures_cortex:
                    x += [_x]
                    y += [_y]
                    y_err += [_y_err]
                    x_units = np.append(x_units, _x_units)
                    y_units = np.append(y_units, _y_units)
                    all = np.append(all, y_units)
                    x_session += _x_session
                    y_session += _y_session

            # ax.set_xticks([-0.5, -0.25, 0, 0.25])
            # utl.make_plot_pretty(ax)
            # ax.set_ylim(lims)
            tau_C_median = utl.get_center(all, center)
            tau_C_median_err = utl.get_sd(all, center)
            slope,intercept,r,p,sd = st.linregress(x, y)
            x2 = np.linspace(min(x), max(x), 10)

            # ax.plot(x2, x2*slope+intercept, '--k', alpha=0.5)

            r_s, p_s = st.spearmanr(x, y)
            r_p, p_p = st.pearsonr(x, y)

            # Regression for data grouped by session
            slope_session ,intercept_session, r , p ,std = st.linregress(x_session, y_session)
            x2 = np.linspace(min(x), max(x), 10)

            r_s_session, p_s_session = st.spearmanr(x_session, y_session)
            r_p_session, p_p_session = st.pearsonr(x_session, y_session)

            Tmax_arr += [Tmax]
            T0_arr += [T0]
            r_p_arr += [r_p]
            p_p_arr += [p_p]
            tau_C_median_arr += [tau_C_median]
            tau_C_median_err_arr += [tau_C_median_err]
            # print('T0: ', T0, "Tmax: ", Tmax)
            # print("Structure medians: ", "r=", r_p," ", "p=", " ", p_p)
            # print("Session medians: ", "r=", r_p_session," ", "p=", " ", p_p_session)
    d = {"Tmax": Tmax_arr, "T0": T0_arr, "r_p": r_p_arr, "p_p": p_p_arr, "tau_C_median": tau_C_median_arr, "tau_C_median_err": tau_C_median_err_arr}
    stats = pd.DataFrame(data = d)
    stats.to_csv(stats_file_name)

measure_two_timescales = 'tau_two_timescales'
if exists(stats_file_name_two_timescales):
    stats_two_timescales = pd.read_csv(stats_file_name_two_timescales)
else:
    Tmax_arr = []
    T0_arr = []
    r_p_arr = []
    p_p_arr = []
    tau_C_median_arr = []
    tau_C_median_err_arr = []
    for Tmax in Tmax_list:
        for T0 in ['05','30']:
            # fig0, ax = plt.subplots()
            # fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1)
            # ax.set_title('intrinsic timescale')
            mre_stats_file_name = "{}/{}_mre_statistics_T0_{}_Tmax_{}_two_timescales.csv".format(stats_dir, analysis, T0, Tmax)
            data = utl.get_analysis_data(csv_file_name, analysis,
                                 mre_stats_file_name=mre_stats_file_name, fit_selection_criterion = fit_selection_criterion)
            print(len(data))
            data = data[utl.df_filter(data, structures=_structures, stimuli=stimuli)]
            try:
                num_neurons = utl.get_expected_number_of_neurons(analysis,
                                                                 selected_structures,
                                                                 stimuli)
                assert np.isclose(len(data), num_neurons, atol=100)
            except:
                print(f'number of neurons is {len(data)}, expected {num_neurons}')
                exit()

            # Once fpr tje
            x = []
            y = []
            y_err = []
            x_units = []
            y_units = []
            x_session = []
            y_session = []
            all = []

            selection = utl.get_data_filter(data, measure_two_timescales)
            data = data[selection]
            # offset_selection = data['bic_offset'] < data['bic_two_timescales']
            # data[offset_selection]['tau_two_timescales'] = data[offset_selection]['tau_offset']
            # print("Number of single timescales selected ", len(data[offset_selection]), " out of ", len(data))
            sessions = np.unique(data['session_id'])
            for structure in _structures:
                _x = structures_map[structure]['hierarchy_score']
                _y = utl.get_center(data[utl.df_filter(data,
                                                       structures=structure)][measure_two_timescales].values, center)
                _y_err = utl.get_sd(data[utl.df_filter(data,
                structures=structure)][measure_two_timescales].values,
                center)
                _y_units = data[utl.df_filter(data,
                                                       structures=structure)][measure_two_timescales].values
                _x_units = np.zeros(len(_y_units)) + _x
                _y_session = []
                _x_session = []
                for session in sessions:
                    session_median = utl.get_center(data[utl.df_filter(data,
                                                           structures=structure, sessions = [session])][measure_two_timescales].values, center)
                    if not np.isnan(session_median):
                        _y_session += [session_median]
                        _x_session += [_x]

                # ax.plot([_x, _x],
                #         [_y - _y_err, _y + _y_err],
                #         lw=2,
                #         color=structures_map[structure]['color'],
                # )
                # ax.plot(_x,
                #         _y,
                #         'o',
                #         color="white",
                #         mec=structures_map[structure]['color']
                # )

                if structure in structures_cortex:
                    x += [_x]
                    y += [_y]
                    y_err += [_y_err]
                    x_units = np.append(x_units, _x_units)
                    y_units = np.append(y_units, _y_units)
                    all = np.append(all, y_units)
                    x_session += _x_session
                    y_session += _y_session

            # ax.set_xticks([-0.5, -0.25, 0, 0.25])
            # utl.make_plot_pretty(ax)
            # ax.set_ylim(lims)
            tau_C_median = utl.get_center(all, center)
            tau_C_median_err = utl.get_sd(all, center)
            slope,intercept,r,p,sd = st.linregress(x, y)
            x2 = np.linspace(min(x), max(x), 10)

            # ax.plot(x2, x2*slope+intercept, '--k', alpha=0.5)

            r_s, p_s = st.spearmanr(x, y)
            r_p, p_p = st.pearsonr(x, y)

            # Regression for data grouped by session
            slope_session ,intercept_session, r , p ,std = st.linregress(x_session, y_session)
            x2 = np.linspace(min(x), max(x), 10)

            r_s_session, p_s_session = st.spearmanr(x_session, y_session)
            r_p_session, p_p_session = st.pearsonr(x_session, y_session)

            Tmax_arr += [Tmax]
            T0_arr += [T0]
            r_p_arr += [r_p]
            p_p_arr += [p_p]
            tau_C_median_arr += [tau_C_median]
            tau_C_median_err_arr += [tau_C_median_err]
            # print('T0: ', T0, "Tmax: ", Tmax)
            # print("Structure medians: ", "r=", r_p," ", "p=", " ", p_p)
            # print("Session medians: ", "r=", r_p_session," ", "p=", " ", p_p_session)
    d = {"Tmax": Tmax_arr, "T0": T0_arr, "r_p": r_p_arr, "p_p": p_p_arr, "tau_C_median": tau_C_median_arr, "tau_C_median_err": tau_C_median_err_arr}
    stats_two_timescales = pd.DataFrame(data = d)
    stats_two_timescales.to_csv(stats_file_name_two_timescales)


# Plotting
fig0, ax = plt.subplots(figsize = [1.1 * plot_settings['panel_width'], .8* plot_settings['panel_height']])
fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1)
if y_measure == 'tau_C_median':
    # single timescale analysis
    T0 = 5
    ax.errorbar(stats['Tmax'][stats['T0'] == T0].values, stats['tau_C_median'][stats['T0'] == T0].values, yerr =  stats['tau_C_median_err'][stats['T0'] == T0].values, lw = 2,
    marker = 'o', color=color, alpha = 0.5, label = r'$T_{\mathrm{min}} = %d$ ms'%T0)
    T0 = 30
    ax.errorbar(stats['Tmax'][stats['T0'] == T0].values, stats['tau_C_median'][stats['T0'] == T0].values, yerr =  stats['tau_C_median_err'][stats['T0'] == T0].values, lw = 2,
    marker = 'o', color=color, alpha = 1.0, label = r'$T_{\mathrm{min}} = %d$ ms'%T0)
    # two timescale analysis
    T0 = 5
    ax.errorbar(stats_two_timescales['Tmax'][stats_two_timescales['T0'] == T0].values, stats_two_timescales['tau_C_median'][stats_two_timescales['T0'] == T0].values, yerr =  stats_two_timescales['tau_C_median_err'][stats_two_timescales['T0'] == T0].values, lw = 2,
    marker = 'o', color='g', alpha = 0.5, label = r'$T_{\mathrm{min}} = %d$ ms'%T0)
    T0 = 30
    ax.errorbar(stats_two_timescales['Tmax'][stats_two_timescales['T0'] == T0].values, stats_two_timescales['tau_C_median'][stats_two_timescales['T0'] == T0].values, yerr =  stats_two_timescales['tau_C_median_err'][stats_two_timescales['T0'] == T0].values, lw = 2,
    marker = 'o', color='g', alpha = 1.0, label = r'$T_{\mathrm{min}} = %d$ ms'%T0)
else:
    # single timescale analysis
    T0 = 5
    ax.plot(stats['Tmax'][stats['T0'] == T0].values, stats[y_measure][stats['T0'] == T0].values, lw = 2,
    marker = 'o', color=color, alpha = 0.5, label = r'$T_{\mathrm{min}} = %d$ ms'%T0)
    T0 = 30
    ax.plot(stats['Tmax'][stats['T0'] == T0].values, stats[y_measure][stats['T0'] == T0].values, lw = 2,
    marker = 'o', color=color, alpha = 1.0, label = r'$T_{\mathrm{min}} = %d$ ms'%T0)
    # two timescale analysis
    T0 = 5
    ax.plot(stats_two_timescales['Tmax'][stats_two_timescales['T0'] == T0].values, stats_two_timescales[y_measure][stats_two_timescales['T0'] == T0].values, lw = 2,
    marker = 'o', color='g', alpha = 0.5, label = r'$T_{\mathrm{min}} = %d$ ms'%T0)
    T0 = 30
    # print(stats[y_measure][stats['T0'] == T0].values)
    ax.plot(stats_two_timescales['Tmax'][stats_two_timescales['T0'] == T0].values, stats_two_timescales[y_measure][stats_two_timescales['T0'] == T0].values, lw = 2,
    marker = 'o', color='g', alpha = 1.0, label = r'$T_{\mathrm{min}} = %d$ ms'%T0)
    # print(stats_two_timescales[y_measure][stats_two_timescales['T0'] == T0].values)


ax.plot([10000], stats_two_timescales[y_measure][(stats_two_timescales['T0'] == 30) & (stats_two_timescales['Tmax'] == 10000)].values, 'rd', lw = 2, zorder = 10)

utl.make_plot_pretty(ax)
ax.set_xlabel('maximum time lag $T_{max}$ (ms)')
ax.set_xscale('log')

if y_measure == 'r_p':
    ax.set_title('Pearson correlation coefficient')
    ax.set_ylabel(r'$r_P$')
elif y_measure == 'p_p':
    ax.set_title('significance of correlation')
    ax.set_ylabel(r'$P_P$')
    ax.set_yscale('log')
    # ax.minorticks_off()
    ax.tick_params(axis='y', which='minor', left=False)
    ax.plot([500, 20000], [0.05, 0.05], '--', color = '0.5')
    # ax.set_xlim([0,10500])
elif y_measure == 'tau_C_median':
    ax.set_title('intrinsic timescale')
    utl.format_ax(ax, measure, 'y', set_ticks=False, tight_layout=True)
    ax.set_ylim([0.04, 0.64])

# ax.set_xticks([0, 2000, 5000, 10000])
if y_measure == 'tau_C_median':
    ax.legend(loc=(0.35,0.0))
elif y_measure == "p_p":
    ax.legend(loc=(0.35,0.57))
else:
    ax.legend()
utl.save_plot(plot_settings, f"{__file__[:-3]}", allen_bo=False, stimulus='movie', measure=y_measure)
