import argparse

defined_measures = ["tau_C",
                    "R_tot"]
parser = argparse.ArgumentParser()
parser.add_argument('measure', type=str, help=f'one of {defined_measures}')
args = parser.parse_args()
if not args.measure in defined_measures:
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

path.insert(1, "../../allen_src/")
import analysis_utils as utl

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

# import logging
# logging.getLogger("mrestimator.utility").setLevel(logging.WARNING)
def f_powerlaw(k, tau, A, gamma, B):#, O):
    return np.abs(A)*np.exp(-k/tau)+ np.abs(B)*np.power(k, - gamma) #+ O*np.ones_like(k)

def f_two_timescales(k, tau1, A1, tau2, A2):
    return np.abs(A1)*np.exp(-k/tau1) + np.abs(A2)*np.exp(-k/tau2)

def f_powerlaw_only(k, gamma, B):#, O):
    return np.abs(B)*np.power(k, - gamma) #+ O*np.ones_like(k)

### settings

analysis = 'allen_fc'

center = 'median' # measure of central tendency
T_measure = 'log_tau_R'
R_measure = 'R_tot'
C_measure = 'log_mre_tau'

# setup analysis
plot_settings = utl.get_default_plot_settings()
plot_settings['panel_width'] = 0.5 * plot_settings['textwidth']
plot_settings['panel_height'] = 2.3
plot_settings['rcparams']['figure.figsize'] =  [plot_settings['panel_width'], plot_settings['panel_height']]
plt.rcParams.update(plot_settings['rcparams'])

stats_dir = dir_settings['stats_dir']

csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_2500.csv".format(stats_dir, analysis)
histdep_file_name = "{}/{}_histdep_data.csv".format(stats_dir, analysis)

structures = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
structures_map = utl.get_structures_map()

data_directory = dir_settings['allen_data_dir']
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

### import data

analysis_metrics = utl.get_analysis_metrics(cache,
                                            analysis)

data = utl.get_analysis_data(csv_file_name, analysis,
                             analysis_metrics=analysis_metrics,
                             mre_stats_file_name=mre_stats_file_name)

histdep_d = utl.get_histdep_data(histdep_file_name,
                                 analysis_data=data)


sample_units = utl.get_sample_units(fc_analysis=(analysis == 'allen_fc'))[:100]
session_id = 816200189
stimulus = 'natural_movie_one_more_repeats'
stimulus_blocks = '3.0,8.0'
neuron_data_dir = dir_settings["neuron_data_dir"]

if not isfile(f"{neuron_data_dir}/spike_data_{session_id}_{sample_units[0]}_{stimulus}_{stimulus_blocks}.npy"):
    session_file = os.path.join(data_directory, f'session_{session_id}', f'session_{session_id}.nwb')
    session_download_link = "http://api.brain-map.org//api/v2/well_known_file_download/1026124840"

    # download allen data
    if not isfile(session_file):
        print('downloading more than 2GB of data.. this may take a long time.')

        if not isdir(os.path.join(data_directory, f'session_{session_id}')):
            os.mkdir(os.path.join(data_directory, f'session_{session_id}'))

        urllib.request.urlretrieve(session_download_link,
                                   session_file)

    # extract data for sample neurons to .npy files
    if not isdir(neuron_data_dir):
        os.mkdir(neuron_data_dir)

    import export_spike_times as exp_st

    target_length = 1080
    transient = 360

    exp_st.save_spike_times_to_file(session_id,
                                    sample_units,
                                    stimulus,
                                    stimulus_blocks,
                                    target_length,
                                    transient,
                                    neuron_data_dir)

sample_unit_1 = 2
sample_unit_2 = 32
sample_unit_3 = 46

# colors = {sample_unit_2: '#2C53AC',}
colors = {sample_unit_1: '#FF912B',
          sample_unit_2: '#2C53AC',
          sample_unit_3: '#7CE426'}


bin_size = 0.005
dtunit='s'
tmin=0.03
# tmax=2.5
tmax=10


# tau, A, gamma, B, O
# fitpars_powerlaw = np.array([(0.1, 0.01, 2, 0.1, 0),
#                     (0.1, 0.1, 2, 0.1, 0),
#                     (1, 0.01, 2, 0.1, 0),
#                     (1, 0.1, 2, 0.1, 0),
#                     (0.1, 0.01, 3, 0, 0),
#                     (0.1, 0.1, 3, 0, 0),
#                     (1, 0.01, 3, 0, 0),
#                     (1, 0.1, 3, 0, 0),
#                     (0.1, 0.01, 1.5, 0, 0),
#                     (0.1, 0.1, 1.5, 0, 0),
#                     (1, 0.01, 1.5, 0, 0),
#                     (1, 0.1, 1.5, 0, 0),
#                     (0.1, 0.01, 1, 0, 0),
#                         (0.1, 0.1, 1, 0, 0),
#                         (1, 0.01, 1, 0, 0),
#                         (1, 0.1, 1, 0, 0),
#                             (0.1, 0.01, 0.5, 0, 0),
#                                 (0.1, 0.1, 0.5, 0, 0),
#                                 (1, 0.01, .5, 0, 0),
#                                 (1, 0.1, .5, 0, 0)])
fitpars_powerlaw = np.array([(0.1, 0.01, 2, 0.1),
                    (0.1, 0.1, 2, 0.1),
                    (1, 0.01, 2, 0.1),
                    (1, 0.1, 2, 0.1),
                    (0.1, 0.01, 3, 0),
                    (0.1, 0.1, 3, 0),
                    (1, 0.01, 3, 0),
                    (1, 0.1, 3, 0),
                    (0.1, 0.01, 1.5, 0),
                    (0.1, 0.1, 1.5, 0),
                    (1, 0.01, 1.5, 0),
                    (1, 0.1, 1.5, 0),
                    (0.1, 0.01, 1, 0),
                        (0.1, 0.1, 1, 0),
                        (1, 0.01, 1, 0),
                        (1, 0.1, 1, 0),
                            (0.1, 0.01, 0.5, 0),
                                (0.1, 0.1, 0.5, 0),
                                (1, 0.01, .5, 0),
                                (1, 0.1, .5, 0)])
fitpars_two_timescales = np.array([
                    (0.1, 0.01, 10, 0.01),
                    (0.1, 0.1, 10, 0.01),
                    (0.5, 0.01, 10, 0.001),
                    (0.5, 0.1, 10, 0.01),
                    (0.1, 0.01, 10, 0),
                    (0.1, 0.1, 10, 0),
                    (0.5, 0.01, 10, 0),
                    (0.5, 0.1, 10, 0)])

fitpars_powerlaw_only = np.array([(2, 0.1),
                        (3, 0.01),
                        (0.5, 0.01),
                        (1.5, 0.01)])

# tau, A, O
fitpars = np.array([(0.1, 0.01, 0),
                    (0.1, 0.1, 0),
                    (1, 0.01, 0),
                    (1, 0.1, 0)])

fig0, ax0 = plt.subplots()#figsize=plot_settings["panel_size"])
fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1)

if args.measure == "tau_C":
    for j, unit in enumerate(sample_units):
        analysis_num = data[(data['unit'] == unit) &
                            (data['stimulus_blocks'] == stimulus_blocks)]['analysis_num'].values[0]

        if j in colors:
            color = structures_map[data[(data['unit'] == unit)]['ecephys_structure_acronym'].values[0]]['color']

            binned_spt = mre.input_handler(f"{neuron_data_dir}/spike_data_{session_id}_{unit}_{stimulus}_{stimulus_blocks}.npy")

            rk = mre.coefficients(binned_spt,
                                  method='ts',
                                  steps=(int(tmin/bin_size),
                                         int(tmax/bin_size)),
                                  dt=bin_size,
                                  dtunit=dtunit)

            # rk_powerlaw = mre.coefficients(binned_spt,
            #                       method='ts',
            #                       steps=(int(tmin_powerlaw/bin_size),
            #                              int(tmax/bin_size)),
            #                       dt=bin_size,
            #                       dtunit=dtunit)

            m = mre.fit(rk,
                        fitpars=fitpars)
            m_two_timescales = mre.fit(rk,
                        fitpars=fitpars_two_timescales, fitfunc = f_two_timescales)
            tau_1_two_timescales = m_two_timescales.popt[0]
            A_1_two_timescales = np.abs(m_two_timescales.popt[1])
            tau_2_two_timescales = m_two_timescales.popt[2]
            A_2_two_timescales = np.abs(m_two_timescales.popt[3])
            # Choose the timescale with higher coefficient A
            tau_two_timescales = (tau_1_two_timescales ,tau_2_two_timescales)[np.argmax((A_1_two_timescales,A_2_two_timescales))]*200
            mre_out = mre.OutputHandler(rk, ax0)
            mre_out.add_coefficients(rk, color=color, lw=0.5, alpha=0.6)
            mre_out.add_fit(m_two_timescales, color=color)#, lw=0.5)
            # TODO: Add vertical lines and diamonds for intrinsic timescales for each neuron
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

    ax0.set_title('intrinsic timescale')


elif args.measure == "R_tot":
    for j, unit in enumerate(sample_units):
        analysis_num = data[(data['unit'] == unit) &
                            (data['stimulus_blocks'] == stimulus_blocks)]['analysis_num'].values[0]


        ax0.plot(histdep_d[histdep_d['analysis_num'] == analysis_num]['T'],
                histdep_d[histdep_d['analysis_num'] == analysis_num][f'max_R_shuffling'],
                alpha = 0.05, color='k')

        if j in colors:
            color = structures_map[data[(data['unit'] == unit)]['ecephys_structure_acronym'].values[0]]['color']

            ax0.plot(histdep_d[histdep_d['analysis_num'] == analysis_num]['T'],
                    histdep_d[histdep_d['analysis_num'] == analysis_num][f'max_R_shuffling'],
                    color=color, lw=2, zorder=3)

    ax0.set_xscale('log')
    ax0.set_xlabel('past range $T$ (ms)')
    ax0.set_ylabel('R(T)')
    ax0.set_xticks([0.01, 0.1, 1])
    ax0.xaxis.set_major_formatter(ticker.FuncFormatter(utl.format_label_in_ms))
    utl.make_plot_pretty(ax0)
    ax0.set_title('predictability')

utl.save_plot(plot_settings, f"{__file__[:-3]}", measure=args.measure)
