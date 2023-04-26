import numpy as np
import matplotlib.pyplot as plt
import mrestimator as mre
from os.path import realpath, dirname, exists, join
import os

import yaml
from sys import stderr, argv, exit, path
import pandas as pd
import h5py
import logging

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

SCRIPT_DIR = dirname(realpath(__file__))
SCRIPT_DIR = "./"
with open('{}/../../dirs.yaml'.format(SCRIPT_DIR), 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)

path.insert(1, dir_settings['hdestimator_src_dir'])

import hde_utils as utl

stimuli = {'brain_observatory_1.1' : ["natural_movie_one",
                                      "natural_movie_three",
                                      "drifting_gratings",
                                      "gabors"],
           'functional_connectivity' : ["natural_movie_one_more_repeats",
                                        "spontaneous"]}

def load_spike_data_hdf5(filepath, session_id, stimulus, stimulus_block, unit_index):
    filename = f"{filepath}/session_{session_id}/session_{session_id}_spike_data.h5"
    f = h5py.File(os.path.expanduser(filename), "r", libver="latest")
    spike_data = f[f"session_{session_id}_stimulus_{stimulus}_stimulus_block_{stimulus_block}_spiketimes"]
    spike_times_unit = spike_data[unit_index]
    spike_times_unit = spike_times_unit[np.isfinite(spike_times_unit)].astype(float)
    f.close()
    metadata = pd.read_hdf(filename, f"session_{session_id}_stimulus_{stimulus}_stimulus_block_{stimulus_block}_metadata")
    metadata_unit = metadata.loc[unit_index]
    return spike_times_unit, metadata_unit

"""
Computes power spectral density for a point process.

- events: list of event times of point process -> spike_times
- min_omega: minimum measured frequency of spectrum -> 0.001 Hz
- max_omega: maximum measured frequency of spectrum -> 10 Hz
- n_bins: number of bins for spectrum between min_omega and max_omega -> 50 to 100
"""
def power_spectrum(events, min_omega, max_omega, n_bins):
    omegas = np.exp(np.linspace(np.log(min_omega),np.log(max_omega),num=n_bins))
    spectrum = np.sum(np.exp(-1j*2*np.pi * np.outer(omegas, events)), axis=1)
    T = events[-1]
    return np.abs(spectrum)**2 / T, omegas

def plot_spectrum(ax, power, omegas):
    norm = np.sum(power)
    ax.plot(omegas, power/norm)
    # ax.plot(omegas, 1/np.power(omegas,2), 'g', label = r"1/f^2")
    # plt.close()

def get_power_spectrum(spike_times,
        session_id,
        unit,
        min_omega = 0.003,
        max_omega = 10,
        n_bins = 50,
        n_snippets = 3,
):
    power_averaged = np.zeros(n_bins)
    n_blocks = len(spike_times)
    n_snippets = int(n_snippets / n_blocks)
    for spike_times_block in spike_times:
        T_recording = spike_times_block[-1] - spike_times_block[0]
        T_snippet = T_recording/n_snippets
        T_snippet_list = np.random.uniform(0.7, 1.3, size = n_snippets) * T_snippet
        T_snippet_list = T_snippet_list * T_recording / np.sum(T_snippet_list)
        spike_times_block = np.array(spike_times_block)
        for i in range(n_snippets):
            if i > 0:
                T_low = np.sum([T_snippet_list[j] for j in range(i)])
            else:
                T_low = 0.
            T_up = np.sum([T_snippet_list[j] for j in range(i+1)])
            selection = spike_times_block >= spike_times_block[0] + T_low
            selection &= spike_times_block < spike_times_block[0] + T_up
            # print(spike_times_block[(spike_times_block >= i*T_snippet)])
            spikes = spike_times_block[selection]
            n_snippets_eff = 0
            if len(spikes) > 0:
                power, omegas = power_spectrum(spikes, min_omega, max_omega, n_bins)
                power_averaged += power/n_blocks
                n_snippets_eff += 1
            power_averaged = power_averaged / n_snippets_eff
    # plot_spectrum(power, omegas)
    return power_averaged, omegas


def get_binned_spike_times(spike_times,
                           bin_size):
    # prepare data
    num_blocks = len(spike_times)
    max_spt = np.max(np.hstack(spike_times))

    binned_spike_times = np.zeros([num_blocks, int(max_spt / bin_size) + 1])

    for block in range(num_blocks):
        # binned_spt = np.zeros(int(spt[-1] / bin_size) + 1, dtype=int)
        for spike_time in spike_times[block]:
            binned_spike_times[block, int(spike_time / bin_size)] += 1
            # binned_spt = np.array([binned_spt])

    return mre.input_handler(binned_spike_times)

#  TODO: Can this go maybe?
# def get_ac_fit_residuals(m, rk):
#     N_next_steps = 10 #5 # number of past bins after first lag that are checked for consistency
#     N_steps = len(rk.steps)
#     var_fit = m.ssres / N_steps
#     std_fit = np.sqrt(var_fit)
#
#     #print(f"N_steps={N_steps}, ssres={m.ssres}, std_fit={std_fit}")
#
#     next_residuals = rk.coefficients[:N_next_steps] - m.fitfunc(rk.steps[:N_next_steps]*m.dt,*m.popt)
#     var_next_residuals = np.sum(next_residuals**2) / N_next_steps
#     std_next_residuals = np.sqrt(var_next_residuals)
#
#     return var_fit, std_fit, std_next_residuals

def f_two_timescales(k, tau, A, tau_long, B):
    return np.abs(A)*np.exp(-k/tau) + np.abs(B)*np.exp(-k/tau_long)

def get_BIC(fit):
    K = len(fit.popt)
    N = len(fit.steps)
    var = np.sum(fit.ssres) / N
    BIC = np.log(N) * K + N * np.log(var)
    return BIC

def get_autocorrelation_fit(spike_times_merged, bin_size = 0.005, tmin = 0.03, tmax = 10.0, dtunit = 's', fit_method = "two_timescales"):
    binned_spt = get_binned_spike_times(spike_times_merged, bin_size)
    mre_out =  mre.OutputHandler()
    mre_out.add_ts(binned_spt)

    rk = mre.coefficients(binned_spt,
                          method='ts',
                          steps=(int(tmin/bin_size),
                                 int(tmax/bin_size)),
                          dt=bin_size,
                          dtunit=dtunit)
    fitpars_offset = np.array([(0.1, 0.01, 0),
                        (0.1, 0.1, 0),
                        (1, 0.01, 0),
                        (1, 0.1, 0)])
    fitpars_two_timescales = np.array([
                        (0.1, 0.01, 10, 0.01),
                        (0.1, 0.1, 10, 0.01),
                        (0.5, 0.01, 10, 0.001),
                        (0.5, 0.1, 10, 0.01),
                        (0.1, 0.01, 10, 0),
                        (0.1, 0.1, 10, 0),
                        (0.5, 0.01, 10, 0),
                        (0.5, 0.1, 10, 0)])
    fit_offset = mre.fit(rk, fitpars=fitpars_offset)
    fit_two_timescales = mre.fit(rk, fitpars=fitpars_two_timescales, fitfunc = f_two_timescales)
    if get_BIC(fit_offset) < get_BIC(fit_two_timescales):
        fit = fit_offset
    else:
        fit = fit_two_timescales
    plt.plot(rk.steps*bin_size, fit.fitfunc(rk.steps*bin_size, *fit.popt))
    plt.plot(rk.steps*bin_size, rk.coefficients,lw = 0.1)
    return fit, rk


def get_power_spectrum_autocorrelation(fit):
    bin_size = fit.dt
    N = len(fit.steps)
    N = 2000
    spectral_density = np.fft.fft(fit.fitfunc(np.arange(2000)*bin_size, *fit.popt))
    len(spectral_density)
    power_spectrum = np.abs(spectral_density)
    omegas = np.arange(2000) / bin_size / N
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(omegas, power_spectrum)
    return power_spectrum, omegas
    #
    # save_to_csv(csv_output_dir,
    #             session_id,
    #             unit,
    #             stimulus,
    #             stimulus_blocks,
    #             rk,
    #             bin_size,
    #             dtunit,
    #             tmin,
    #             tmax,
    #             fit_offset,
    #             fit_two_timescales)

    # ores = mre.OutputHandler([rk, m])
    # ores.save(mre_output_dir)

    # plt.show()

# eg run:
# python3 print_spike_times_for_neuron.py 816200189 951141184 spontaneous null 900 60 | python3 mre_analysis.py /dev/stdin 816200189 951141184 spontaneous null settings.yaml

if __name__ == "__main__":
    data_directory = dir_settings['allen_data_dir']
    manifest_path = join(data_directory, "manifest.json")
    session_type = 'functional_connectivity'
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()
    session_ids = sessions[sessions['session_type'] == session_type].index.values
    bin_size = 0.005
    dtunit = 's'
    tmin  = 0.03
    tmax  = 10.0
    if argv[1] == "movie":
        stimulus = "natural_movie_one_more_repeats"
        stimulus_blocks = ['3.0', '8.0']
    elif argv[1] == "spontaneous":
        stimulus = argv[1]
        # stimulus = "spontaneous"
        stimulus_blocks = ['null']
    session_id = session_ids[int(argv[2])]
    # session_id = 816200189
    unit_index = int(argv[3])
    # unit_index = 50
    fig, ax = plt.subplots(figsize = [3.1,2.1])#figsize=plot_settings["panel_size"]) # FIG SIZE: 3.1, 2.1
    fig.subplots_adjust(right=0.98, left=0.1, bottom=0.1)
    # ax.set_title(f"{session_id}, {unit}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('power')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    filename = f"{data_directory}/session_{session_id}/session_{session_id}_spike_data.h5"
    metadata = pd.read_hdf(filename, f"session_{session_id}_stimulus_{stimulus}_stimulus_block_{stimulus_blocks[0]}_metadata")
    n_units = len(metadata.unit_id.values)
    n_bins = 50
    log_mean_power = np.zeros(n_bins)
    for unit_index in range(n_units):
        spike_times_merged =[]
        for stimulus_block in stimulus_blocks:
            spike_times, metadata = load_spike_data_hdf5(data_directory, session_id, stimulus, stimulus_block, unit_index)
            spike_times_merged += [spike_times]
        power, omegas = get_power_spectrum(spike_times_merged, session_id, metadata["unit_id"], n_bins = n_bins)
        log_mean_power += np.log(power)/n_units
        norm = np.sum(power)
        ax.plot(omegas, power/norm, color = 'b', alpha = 0.1, lw = 0.5)
    norm = np.sum(1/omegas)
    ax.plot(omegas, 1/omegas/norm, color = "orange", zorder = 300, lw = 2, label = "1/f")
    norm = np.sum(np.exp(log_mean_power))
    ax.plot(omegas, np.exp(log_mean_power)/ norm, color = 'b', lw = 2)
    ax.legend()
    plt.show()
            # print(metadata)
    # load settings
    # bin_size = float(analysis_settings["bin_size"])
    # dtunit = analysis_settings["dtunit"]
    # tmin  = float(analysis_settings["tmin"])
    # tmax  = float(analysis_settings["tmax"])
    # mre_output_dir = analysis_settings["output_dir"]
    # csv_output_dir = analysis_settings["csv_output_dir"]

    exit()
