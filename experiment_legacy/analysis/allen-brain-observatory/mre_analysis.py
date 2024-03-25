import numpy as np
import matplotlib.pyplot as plt
import mrestimator as mre
from os.path import realpath, dirname, exists

import yaml
from sys import stderr, argv, exit, path
import pandas as pd


path.insert(1, "../../allen_src/")

import load_spikes
import mre_estimation

stimuli = {'brain_observatory_1.1' : ["natural_movie_one",
                                      "natural_movie_three",
                                      "drifting_gratings",
                                      "gabors"],
           'functional_connectivity' : ["natural_movie_one_more_repeats",
                                        "natural_movie_one_shuffled",
                                        "drifting_gratings_contrast",
                                        "drifting_gratings_75_repeats",
                                        "gabors",
                                        "spontaneous"]}

def main(spike_times,
         bin_size,
         dtunit,
         tmin,
         tmax,
         csv_output_dir,
         session_id,
         unit,
         stimulus,
         stimulus_blocks):
    
    title = f'mre_{session_id}_{unit}_{stimulus}_{stimulus_blocks}'

    binned_spt = load_spikes.get_binned_spike_times(spike_times, bin_size)

    mre_out =  mre.OutputHandler()
    mre_out.add_ts(binned_spt)

    rk = mre.coefficients(binned_spt,
                          method='ts',
                          steps=(int(tmin/bin_size),
                                 int(tmax/bin_size)),
                          dt=bin_size,
                          dtunit=dtunit,
                          desc=title)

    fit_offset = mre_estimation.single_timescale_fit(rk)
    fit_two_timescales, tau_two_timescales, A_two_timescales, tau_rejected, A_rejected = mre_estimation.two_timescales_fit(rk)

    bic_offset, bic_passed_offset = mre_estimation.test_BIC(fit_offset, rk)
    aic_offset, aic_passed_offset = mre_estimation.test_AIC(fit_offset, rk)
    bic_two_timescales, bic_passed_two_timescales = mre_estimation.test_BIC(fit_two_timescales, rk)
    aic_two_timescales, aic_passed_two_timescales = mre_estimation.test_AIC(fit_two_timescales, rk)

    # stats = {
    #     "session_id" : str(session_id),
    #     "unit" : str(unit),
    #     "stimulus" : str(stimulus),
    #     "stimulus_blocks" : stimulus_blocks,
    #     "tau_offset" : str(fit_offset.tau),
    #     "A_offset" : str(np.abs(fit_offset.popt[1])),
    #     "O_offset" : str(fit_offset.popt[2]),
    #     "ssres_offset" : str(fit_offset.ssres),
    #     "bic_offset" : str(bic_offset),
    #     "bic_passed_offset" : str(bic_passed_offset),
    #     "aic_offset" : str(aic_offset),
    #     "aic_passed_offset" : str(aic_passed_offset),
    #     "tau_two_timescales" : str(tau_two_timescales),
    #     "A_two_timescales" : str(A_two_timescales),
    #     "tau_rejected" : str(tau_rejected),
    #     "A_rejected" : str(A_rejected),
    #     "ssres_two_timescales" : str(fit_two_timescales.ssres),
    #     "bic_two_timescales" : str(bic_two_timescales),
    #     "bic_passed_two_timescales" : str(bic_passed_two_timescales),
    #     "aic_two_timescales" : str(aic_two_timescales),
    #     "aic_passed_two_timescales" : str(aic_passed_two_timescales),
    #     "bin_size" : str(bin_size),
    #     "dtunit" : str(dtunit),
    #     "tmin" : str(tmin),
    #     "tmax" : str(tmax)
    # }
    stats = {
        "session_id" : [session_id],
        "unit" : [unit],
        "stimulus" : [stimulus],
        "stimulus_blocks" : [str(stimulus_blocks)],
        "tau_offset" : [fit_offset.tau],
        "A_offset" : [np.abs(fit_offset.popt[1])],
        "O_offset" : [fit_offset.popt[2]],
        "ssres_offset" : [fit_offset.ssres],
        "bic_offset" : [bic_offset],
        "bic_passed_offset" : [bic_passed_offset],
        "aic_offset" : [aic_offset],
        "aic_passed_offset" : [aic_passed_offset],
        "tau_two_timescales" : [tau_two_timescales],
        "A_two_timescales" : [A_two_timescales],
        "tau_rejected" : [tau_rejected],
        "A_rejected" : [A_rejected],
        "ssres_two_timescales" : [fit_two_timescales.ssres],
        "bic_two_timescales" : [bic_two_timescales],
        "bic_passed_two_timescales" : [bic_passed_two_timescales],
        "aic_two_timescales" : [aic_two_timescales],
        "aic_passed_two_timescales" : [aic_passed_two_timescales],
        "bin_size" : [bin_size],
        "dtunit" : [dtunit],
        "tmin" : [tmin],
        "tmax" : [tmax]
    }

    # Store as csv
    df = pd.DataFrame(data = stats)
    df.to_csv(f"{csv_output_dir}/mre_out_{session_id}_{unit}_{stimulus}_{stimulus_blocks}.csv", index = False)
    
    # csv_file_name = f"mre_out_two_timescales_{session_id}_{unit}_{stimulus}_{stimulus_blocks}.csv"
    # with open("{}/{}".format(csv_output_dir,
    #                          csv_file_name], 'w') as csv_file:

    #     print(fit_offset.tau, fit_offset.popt[0])

    #     csv_file.write("{}\n".format(",".join(stats.keys())))
    #     csv_file.write("{}\n".format(",".join(stats.values())))

    # csv_file.close()

def print_usage_and_exit(script_name):
    print('usage is python3 {} spike_times_file_name session_id unit stimulus stimulus_blocks settings_file_name'.format(script_name))
    print('with stimulus one of {} for the brain_observatory_1.1 stimulus set,'.format(
        stimuli['brain_observatory_1.1']))
    print('or one of {} for the functional_connectivity stimulus set.'.format(
        stimuli['functional_connectivity']))
    exit()

if __name__ == "__main__":
    # if not len(argv) == 8 \
    #    or not argv[2].isdecimal() \
    #    or not argv[3].isdecimal() \
    #    or ((not argv[4] in stimuli['brain_observatory_1.1']) &
    #        (not argv[4] in stimuli['functional_connectivity'])):
    #     print_usage_and_exit(argv[0])
 
 
    # if not exists(spike_times_file_name):
    #     print("Spike times file {} not found.  Aborting.".format(spike_times_file_name),
    #           file=stderr, flush=True)
    #     exit(1)
    # spike_times = utl.get_spike_times_from_file(spike_times_file_name)

    session_id = int(argv[1]) 
    unit = int(argv[2])
    stimulus = argv[3]
    stimulus_blocks = argv[4]
    target_length = int(argv[5]) 
    transient = int(argv[6])
    tmin  = argv[7]
    tmax  = argv[8]
    data_directory = argv[9]
    settings_file_name = argv[10]

    spike_times_merged = load_spikes.get_spike_times(session_id,
                                                    unit,
                                                    stimulus, 
                                                    stimulus_blocks.split(","),
                                                    target_length,
                                                    transient,
                                                    data_directory)


    # if not isinstance(spike_times, np.ndarray):
    #     print("Error loading spike times. Aborting.",
    #           file=stderr, flush=True)
    #     exit(1)
    for spike_times in spike_times_merged:
        if not len(spike_times) > 0:
            print("Spike times are empty. Aborting.",
              file=stderr, flush=True)
            exit(1)

    if not exists(settings_file_name):
        print("Settings file {} not found.  Aborting.".format(spike_times_file_name),
              file=stderr, flush=True)
        exit(1)

    with open(settings_file_name, 'r') as analysis_settings_file:
        analysis_settings = yaml.load(analysis_settings_file, Loader=yaml.BaseLoader)

    # load settings
    bin_size = float(analysis_settings["bin_size"])
    dtunit = analysis_settings["dtunit"]
    csv_output_path = analysis_settings["csv_output_path"]
    csv_output_dir = f"{csv_output_path}/csv_output_{tmin}_{tmax}_new"

    exit(main(spike_times_merged,
              bin_size,
              dtunit,
              int(tmin)/1000., # transform to ms 
              int(tmax)/1000., # transform to ms
              csv_output_dir,
              session_id=session_id,
              unit=unit,
              stimulus=stimulus,
              stimulus_blocks=stimulus_blocks
              ))
