import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import mrestimator as mre
from os.path import realpath, dirname, exists

import yaml
from sys import stderr, argv, exit, path

SCRIPT_DIR = dirname(realpath(__file__))
mre_analysis_dir = f'{SCRIPT_DIR}/mre_analysis_30'
data_dir = '/data.nst/bcramer/hx/static_input/static/'

with open('{}/settings.yaml'.format(mre_analysis_dir), 'r') as analysis_settings_file:
    analysis_settings = yaml.load(analysis_settings_file, Loader=yaml.BaseLoader)
    
# load settings
bin_size = float(analysis_settings["bin_size"])
dtunit = analysis_settings["dtunit"]
tmin  = float(analysis_settings["tmin"])
tmax  = float(analysis_settings["tmax"])
mre_output_dir = analysis_settings["output_dir"]
csv_output_dir = analysis_settings["csv_output_dir"]

def save_to_csv(csv_output_dir,
                filename,
                neuronNum,
                mre_results):
    csv_file_name \
        = f"mre_out_{filename}_{neuronNum}.csv"
    
    with open("{}/{}".format(csv_output_dir,
                             csv_file_name), 'w') as csv_file:

        print(mre_results.tau, mre_results.popt[0])
        
        stats = {
            "filename" : str(filename),
            "neuronNum" : str(neuronNum),
            "mre_tau" : str(mre_results.tau),
            "mre_A" : str(mre_results.popt[1]),
            "mre_O" : str(mre_results.popt[2]),
            "mre_m" : str(mre_results.mre),
            "mre_ssres" : str(mre_results.ssres),
        }

        csv_file.write("#{}\n".format(",".join(stats.keys())))
        csv_file.write("{}\n".format(",".join(stats.values())))

    csv_file.close()


def get_spike_times(filename,
                    neuron_num):

    spike_times_list = []
    num_trials = 30

    for trial in range(num_trials):
        spikes = np.load(join(data_dir, 
                              f"{filename}_{trial:03d}.npy"))

        # spikes is array of shape [N_spikes, 2]
        # index 0 of last dimension: time stemps
        # index 1 of last dimension: address

        # extract spike times for every neuron
        # times are still in hardware domain, i.e. need to be multiplied
        # by 1000 to obtain biological equivalent time
        neuron_mask = (spikes[:, 1] == neuron_num) \
                      & (spikes[:,0] <= 0.31)
        spike_times_list += [spikes[neuron_mask, 0] * 1e3]


    max_spt = np.max(np.hstack(spike_times_list))
    spike_times = np.zeros([num_trials, int(max_spt / bin_size) + 1])

    for trial in range(num_trials):
        for spike_time in spike_times_list[trial]:
            spike_times[trial, int(spike_time / bin_size)] += 1

    #print(spike_times.shape)
    #exit()
    return mre.input_handler(spike_times)


def main(spt,
         bin_size,
         dtunit,
         tmin,
         tmax,
         mre_output_dir,
         csv_output_dir,
         filename,
         neuronNum):
    title = f'mre_{filename}_{neuronNum}'
    
    # prepare data
    # binned_spt = np.zeros(int(spt[-1] / bin_size) + 1, dtype=int)
    # for spike_time in spt:
    #     binned_spt[int(spike_time / bin_size)] += 1
    # binned_spt = np.array([binned_spt])

    # prepare fit
    fitpars = np.array([(0.1, 0.01, 0),
                        (0.1, 0.1, 0),
                        (1, 0.01, 0),
                        (1, 0.1, 0)])    

    # _ = mre.full_analysis(
    #     data=binned_spt,
    #     coefficientmethod='ts',
    #     targetdir=mre_output_dir,
    #     title=title,
    #     dt=bin_size, dtunit=dtunit,
    #     tmin=tmin, tmax=tmax,
    #     fitfuncs=['exp_offs'],
    # )

    mre_out =  mre.OutputHandler()
    mre_out.add_ts(spt)

    rk = mre.coefficients(spt,
                          method='ts',
                          steps=(int(tmin/bin_size),
                                 int(tmax/bin_size)),
                          dt=bin_size,
                          dtunit=dtunit,
                          desc=title)

    m = mre.fit(rk,
                fitpars=fitpars)
    
    save_to_csv(csv_output_dir,
                filename,
                neuronNum,
                m)

    # ores = mre.OutputHandler([rk, m])
    # ores.save(mre_output_dir)

    # plt.show()


# eg run:
# python3 mre_analysis.py hom_spikes_000_000 0
    
def print_usage_and_exit(script_name):
    print('usage is python3 {} filename neuronNum'.format(script_name))
    exit()
    
if __name__ == "__main__":
    if not len(argv) == 3 \
       or not argv[2].isdecimal():
        print_usage_and_exit(argv[0])

    filename = argv[1]
    neuronNum = int(argv[2])

    spike_times = get_spike_times(filename,
                                  neuronNum)

    if not isinstance(spike_times, np.ndarray):
        print("Error loading spike times. Aborting.",
              file=stderr, flush=True)
        exit(1)
    elif not len(spike_times) > 0:
        print("Spike times are empty. Aborting.",
              file=stderr, flush=True)
        exit(1)

    exit(main(spike_times,
              bin_size,
              dtunit,
              tmin,
              tmax,
              mre_output_dir,
              csv_output_dir,
              filename=filename,
              neuronNum=neuronNum))
