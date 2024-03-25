#%%
from sys import path, stderr, argv, exit
from os.path import realpath, dirname, join, exists
import os
# import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('seaborn-darkgrid')
import pandas as pd
import seaborn as sns
#sns.set_style("white")
import scipy.stats as st
# from scipy.stats import linregress, pearsonr, spearmanr
import analysis_utils as utl
import argparse
import json
import mrestimator as mre
from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import Normalize
import importlib
importlib.reload(utl);
# %% codecell
# supress matplotlib warnings
from matplotlib import MatplotlibDeprecationWarning

def apply_filter(seed_index, data, measure):
    data_seed = data[data['seed_index']==seed_index]
    # if measure == 'mre_tau':
    #     return data_seed[measure]
    if measure == 'mre_tau':
        return data_seed[data_seed['BIC test'] == 'passed']['tau']
        # return data_seed[(data_seed['mre_bic_passed']) & (data_seed['mre_tau'] < 2500)][measure]
# (_data[measure] > np.nanpercentile(_data[measure].values, 2.5)) & (_data[measure] < np.nanpercentile(_data[measure].values, 97.5))
    if measure == 'tau_R':
        return data_seed[data_seed['tau_R'] >= 0.0][measure]
    if measure == 'R_tot':
        return data_seed['R_tot']

def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('data', df)
    store.get_storer('data').attrs.metadata = kwargs
    store.close()

def h5load(store):
    data = store['data']
    metadata = store.get_storer('data').attrs.metadata
    return data, metadata

def merge_and_load(data_dir, tmax,  tmin, N_seeds, target_rate, T_recording, network):
    m_ar_file_name ="m_ar_target_rate_%d_Trec_%d_%s.csv"%(target_rate, T_recording, network)
    m_ar = pd.read_csv(m_ar_file_name)
    filename = 'intrinsic_timescale_tmin_%d_tmax_%d_target_rate_%d_Trec_%d_%s.h5'%(tmin, tmax, target_rate, T_recording, network)
    if exists(filename):
        with pd.HDFStore(filename) as store:
            data_merged, metadata = h5load(store)
    else:
        data_merged = pd.DataFrame()
        m_index_list = np.unique(m_ar['m_index'].values)
        for m_index in range(15):
            for seed_index in range(N_seeds):
                if network == 'ER':
                    filename_seed = '%s/intrinsic_timescale_simulation_tmin_%d_tmax_%d_m_index_%d_seed_index_%d_target_rate_%d_Trec_%d_ER.h5'%(data_dir, tmin, tmax, m_index, seed_index, target_rate, T_recording)
                else:
                    filename_seed = '%s/intrinsic_timescale_simulation_tmin_%d_tmax_%d_m_index_%d_seed_index_%d_target_rate_%d_Trec_%d_%s.h5'%(data_dir, tmin, tmax, m_index, seed_index, target_rate, T_recording, network)
                print(filename_seed)
                with pd.HDFStore(filename_seed) as store:
                    data, metadata = h5load(store)
                data_merged = data_merged.append(data, ignore_index = True)
        h5store(filename, data_merged, **metadata)
    return m_ar, data_merged

# def merge_and_load(m_ar, data_dir, tmax,  tmin, target_rate, N_seeds):
#     df = pd.DataFrame()
#     m_index_list = np.unique(m_ar['m_index'].values)
#     filename = 'intrinsic_timescale_tmin_%d_tmax_%d_target_rate_%d'%(tmin, tmax, target_rate)
#     for m_index in range(15):
#         for seed_index in range(N_seeds):
#             filename_seed = '%s/intrinsic_timescale_simulation_tmin_%d_tmax_%d_m_index_%d_seed_index_%d_target_rate_%d.h5'%(data_dir, tmin, tmax, m_index, seed_index, target_rate)
#             print(filename_seed)
#             with pd.HDFStore(filename_seed) as store:
#                 data, metadata = h5load(store)
#             df = df.append(data, ignore_index = True)
#     h5store(filename + '.h5', df, **metadata)
#     return df, metadata

#%% Load the parameters 
import xarray as xr
import zarr
sigma = 0.01
correlated_input = True
input_type = "OU"
tau_OU = 30./1000
adaptation = False
heterogeneous_weights = True
if correlated_input:
    if input_type == "OU":
        input_fname = f"_correlated_input_OU_tau{int(tau_OU*1000)}ms_sigma{sigma}"
    else: 
        input_fname = f"_correlated_input"
else:
    input_fname = "_no_correlated_input"
if adaptation:
    adaptation_fname = f"_adaptation_sigma{sigma}"
else:
    adaptation_fname = "_no_adaptation"
if heterogeneous_weights:
    heterogeneous_fname = "_hetero_weights"
filename=f"/data.nst/share/projects/information_timescales/branching_network/res_dset{adaptation_fname}{input_fname}{heterogeneous_fname}_sparse.zarr"
print(filename)
res_dset = xr.load_dataset(filename, engine = "zarr")


#%%
"""ANALYSIS PARAMETERS"""
N_seeds = int(argv[1])
# N_seeds = 25
target_rate = float(argv[2]) # to be provided in Hz target_rate = 1
T_recording = int(argv[3]) # 1200 or 5400 sec T_recording = 1200
network = argv[4] #all-to-all/ER (Erdoes-Renyi) network = 'all-to-all'

data_dir = '../data/stats'
tmax = 750
tmin = 0 # first ms excluded for autocorrelation fit
if network == 'all-to-all_wadaptation':
    tmax = 750
    tmin = 20
bin_size = 5
N = 512
center = 'median'
plotting_measures = ['mre_tau', 'tau_R','R_tot']
predictability_filename = '%s/statistics_branching.csv'%data_dir    
if network == 'all-to-all_wadaptation':
    predictability_filename = 'statistics_branching_w_adaptation_05.csv'
data_filter_name = 'no_filter'
analysis = 'analysis_v0'
estimation_method = 'shuffling'
predictability_data = utl.get_analysis_data(predictability_filename, analysis,
                                    estimation_method,
                                    data_filter_name)
predictability_data = predictability_data.rename(columns = {'kin': 'm_index', 'seed':'seed_index'})
m_ar, mre_data = merge_and_load(data_dir, tmax,  tmin, N_seeds, target_rate, T_recording, network)
# m_ar_file_name ="m_ar_target_rate%d.csv"%target_rate
# m_ar = pd.read_csv(m_ar_file_name)
#
# """LOAD DATA"""
# target_rate = 1
# intrinsic_timescale_filename = 'intrinsic_timescale_tmin_%d_tmax_%d_target_rate_%d.h5'%(tmin, tmax, target_rate)
# if not exists(intrinsic_timescale_filename):
#     data_low_rate = merge_and_load(m_ar, data_dir, tmax,  tmin, target_rate, N_seeds)
# else:
#     with pd.HDFStore(intrinsic_timescale_filename) as store:
#         data_low_rate, metadata = h5load(store)
#
# target_rate = 5
# intrinsic_timescale_filename = 'intrinsic_timescale_tmin_%d_tmax_%d_target_rate_%d.h5'%(tmin, tmax, target_rate)
# if not exists(intrinsic_timescale_filename):
#     data_high_rate = merge_and_load(m_ar, data_dir, tmax,  tmin, target_rate, N_seeds)
# else:
#     with pd.HDFStore(intrinsic_timescale_filename) as store:
#         data_high_rate, metadata = h5load(store)

# %% codecell
save_fig = True
imgdir = '../img'
img_format = 'pdf'
textwidth = 5.787402103

rcparams = {
    'axes.labelsize': 12,
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': 'Computer Modern Roman',
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    # 'text.usetex': True,
    'figure.figsize': [4.6299216824, 3]  # 0.8 * width
}
plt.rcParams.update(rcparams)

# Get Kin values and set color map
m_list = np.linspace(0.6,0.9,15)
h_list = h = (1-m_list) * target_rate  # external input rate (per neuron)
cmap = cm.get_cmap('viridis', 50)
cmapn = Normalize(vmin=min(h_list), vmax=max(h_list))

# Apply filters:
# For tau_C: Throw away BIC failed.
# data_filtered = data[(data['mre_A'] > 0.1) & (data['mre_O'] < 0.2)
#             & (data['mre_tau'] < 1500)]
fig0, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(1.4*textwidth,2.1))
fig0.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, wspace=0.5)

"""Plotting results for individual seeds"""
for ax, data, measure in zip([ax0, ax1, ax2], [mre_data, predictability_data, predictability_data], plotting_measures):
    for m_index in range(15):
        h = h_list[m_index]
        data_m_index = data[data['m_index'].values.astype(int)==m_index]
        measure_seeds = []
        m_seeds = []
        for seed_index in range(N_seeds):
            # get mean over samples, and median over neurons for individual seeds of network initialization
            data_seed = apply_filter(seed_index, data_m_index, measure)
            measure_seed = utl.get_center(data_seed, center) # median over neurons within the seed
            measure_seeds += [measure_seed]
            m_seed = m_ar['m_ar'].values[(m_ar['m_index']==m_index) & (m_ar['seed_index']==seed_index)] # mean m over samples for that seed
            # m_seed = [np.mean(m_arr[0, 0])]
            m_seeds += [m_seed]
            ax.plot([m_seed], [measure_seed], marker = 'o', markersize = .5, color  = cmap(cmapn(h)))
        measure_kin = utl.get_center(measure_seeds , center)
        measure_kin_err = utl.get_sd(measure_seeds , center)
        m_kin = utl.get_center(m_seeds, center)
        m_kin_err = utl.get_sd(m_seeds, center)
        ax.errorbar(m_kin, measure_kin, xerr = m_kin_err, yerr = measure_kin_err, fmt = 'o', color = cmap(cmapn(h)))
# utl.format_ax(ax, measure, 'y', set_ticks=False, tight_layout=True)
    # if measure == "mre_tau":
    #     ax.set_ylabel(r'$\tau_C$ (ms)')
    #     ax.yaxis.set_major_formatter(ticker.FuncFormatter(utl.format_label_in_ms))
    #     ax.set_title('intrinsic timescale', loc='center', fontsize = 12.)
    # if measure == "tau_R":
    #     ax.set_ylabel(r"$\tau_R$ (ms)")
    #     ax.yaxis.set_major_formatter(ticker.FuncFormatter(utl.format_label_in_ms))
    #     ax.set_title('information timescale', loc='center', fontsize = 12.)
    # if measure == "R_tot":
    #     ax.set_ylabel(r"$R_{tot}$")
    #     ax.set_title('predictable information', loc='center', fontsize = 12.)

    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(utl.format_label_in_ms))
    #ax.set_yticks([0.04, 0.06, 0.08, 0.1])
    #ax.set_xticks([-0.5, -0.25, 0, 0.25])
    utl.make_plot_pretty(ax)
    ax.set_xlabel(r'neural efficacy $m$')
    m_range = np.arange(np.min(m_ar['m_ar'].values),np.max(m_ar['m_ar'].values), 0.005)
    if measure == "mre_tau":
        ax.set_ylabel(r'$\tau_C$ (ms)')
        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(utl.format_label_in_ms))
        ax.set_title('intrinsic timescale', loc='center', fontsize = 12.)
        ax.set_ylim([0,50])
    if measure == "tau_R":
        ax.set_ylabel(r"$\tau_R$ (ms)")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(utl.format_label_in_ms))
        ax.set_title('information timescale', loc='center', fontsize = 12.)
    if measure == "R_tot":
        ax.set_ylabel(r"$R_{tot}$")
        ax.set_title('predictable information', loc='center', fontsize = 12.)

# if network == 'all-to-all_wadaptation':
#     plt.title('branching network (all-to-all + adaptation)', loc='center', fontsize = 12.)
# else:
#     plt.title('branching network (%s)'%network, loc='center', fontsize = 12.)
# ax1.set_title('5 Hz', loc='center', fontsize = 12.)
fig0.colorbar(cm.ScalarMappable(norm=cmapn, cmap=cmap), ax=ax2,
              label="$h$ (Hz)")

if save_fig:
    plt.savefig('{}/measures_by_m_tmax_C_{}_tmin_C_{}_Trec_{}_{}.{}'.format(imgdir,
                                              tmin, tmax, T_recording, network,
                                              img_format),
                bbox_inches='tight')

else:
    plt.show()
    plt.close()
