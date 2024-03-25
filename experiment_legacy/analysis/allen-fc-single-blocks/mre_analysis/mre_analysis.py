import numpy as np
import matplotlib.pyplot as plt
import mrestimator as mre
from os.path import realpath, dirname, exists

import yaml
from sys import stderr, argv, exit, path

SCRIPT_DIR = dirname(realpath(__file__))
with open('{}/../../../information-theory/dev/ma_plots/dirs.yaml'.format(SCRIPT_DIR), 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)
    
path.insert(1, dir_settings['hdestimator_src_dir'])

import hde_utils as utl

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

def save_to_csv(csv_output_dir,
                session_id,
                unit,
                stimulus,
                stimulus_blocks,
                mre_results,
                bic,
                bin_size,
                dtunit,
                tmin,
                tmax):
    csv_file_name \
        = f"mre_out_{session_id}_{unit}_{stimulus}_{stimulus_blocks}.csv"
    
    with open("{}/{}".format(csv_output_dir,
                             csv_file_name), 'w') as csv_file:

        print(mre_results.tau, mre_results.popt[0])
        
        stats = {
            "session_id" : str(session_id),
            "unit" : str(unit),
            "stimulus" : str(stimulus),
            "stimulus_blocks" : str(";".join(stimulus_blocks.split(','))),
            "mre_tau" : str(mre_results.tau),
            "mre_A" : str(mre_results.popt[1]),
            "mre_O" : str(mre_results.popt[2]),
            "mre_m" : str(mre_results.mre),
            "mre_ssres" : str(mre_results.ssres),
            "mre_bic_passed" : str(bic),
            "bin_size" : str(bin_size),
            "dtunit" : str(dtunit),
            "tmin" : str(tmin),
            "tmax" : str(tmax)
        }

        csv_file.write("#{}\n".format(",".join(stats.keys())))
        csv_file.write("{}\n".format(",".join(stats.values())))

    csv_file.close()


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

def get_ac_fit_residuals(m, rk):
    N_next_steps = 10 #5 # number of past bins after first lag that are checked for consistency
    N_steps = len(rk.steps)
    var_fit = m.ssres / N_steps
    std_fit = np.sqrt(var_fit)
    
    #print(f"N_steps={N_steps}, ssres={m.ssres}, std_fit={std_fit}")
    
    next_residuals = rk.coefficients[:N_next_steps] - m.fitfunc(rk.steps[:N_next_steps]*m.dt,*m.popt)
    var_next_residuals = np.sum(next_residuals**2) / N_next_steps
    std_next_residuals = np.sqrt(var_next_residuals)
    
    return var_fit, std_fit, std_next_residuals

def test_BIC(m, rk): #Sort out units that have very shallow autocorrelation
    var_fit, std_fit, std_next_residuals = get_ac_fit_residuals(m, rk)
    N_steps = len(rk.steps)
    var_constant = np.var(rk.coefficients) # variance around a constant mean/offset of the AC
    BIC_exponential = N_steps * np.log(var_fit) + 3 * np.log(N_steps)
    BIC_constant = N_steps * np.log(var_constant) + 1 * np.log(N_steps)
    if BIC_exponential< BIC_constant:
        return True
    else:
        return False

def main(spike_times,
         bin_size,
         dtunit,
         tmin,
         tmax,
         mre_output_dir,
         csv_output_dir,
         session_id,
         unit,
         stimulus,
         stimulus_blocks):
    title = f'mre_{session_id}_{unit}_{stimulus}_{stimulus_blocks}'

    binned_spt = get_binned_spike_times(spike_times, bin_size)

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
    mre_out.add_ts(binned_spt)

    rk = mre.coefficients(binned_spt,
                          method='ts',
                          steps=(int(tmin/bin_size),
                                 int(tmax/bin_size)),
                          dt=bin_size,
                          dtunit=dtunit,
                          desc=title)

    m = mre.fit(rk,
                fitpars=fitpars)

    bic = test_BIC(m, rk)
    
    save_to_csv(csv_output_dir,
                session_id,
                unit,
                stimulus,
                stimulus_blocks,
                m,
                bic,
                bin_size,
                dtunit,
                tmin,
                tmax)

    # ores = mre.OutputHandler([rk, m])
    # ores.save(mre_output_dir)

    # plt.show()


# eg run:
# python3 print_spike_times_for_neuron.py 816200189 951141184 spontaneous null 900 60 | python3 mre_analysis.py /dev/stdin 816200189 951141184 spontaneous null settings.yaml
    
def print_usage_and_exit(script_name):
    print('usage is python3 {} spike_times_file_name session_id unit stimulus stimulus_blocks settings_file_name'.format(script_name))
    print('with stimulus one of {} for the brain_observatory_1.1 stimulus set,'.format(
        stimuli['brain_observatory_1.1']))
    print('or one of {} for the functional_connectivity stimulus set.'.format(
        stimuli['functional_connectivity']))
    exit()
    
if __name__ == "__main__":
    if not len(argv) == 7 \
       or not argv[2].isdecimal() \
       or not argv[3].isdecimal() \
       or ((not argv[4] in stimuli['brain_observatory_1.1']) &
           (not argv[4] in stimuli['functional_connectivity'])):
        print_usage_and_exit(argv[0])
    spike_times_file_name = argv[1]
    settings_file_name = argv[6]

    for stimulus_block in argv[5].split(','):
        try:
            float(stimulus_block)
        except:
            if not stimulus_block == 'null':
                print('stimulus_blocks must be a number or "null" or several instanced thereof, '
                      'separated by commas')
                exit()

    if not exists(spike_times_file_name):
        print("Spike times file {} not found.  Aborting.".format(spike_times_file_name),
              file=stderr, flush=True)
        exit(1)

    spike_times = utl.get_spike_times_from_file(spike_times_file_name)

    if not isinstance(spike_times, np.ndarray):
        print("Error loading spike times. Aborting.",
              file=stderr, flush=True)
        exit(1)
    elif not len(spike_times) > 0:
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
    tmin  = float(analysis_settings["tmin"])
    tmax  = float(analysis_settings["tmax"])
    mre_output_dir = analysis_settings["output_dir"]
    csv_output_dir = analysis_settings["csv_output_dir"]

    exit(main(spike_times,
              bin_size,
              dtunit,
              tmin,
              tmax,
              mre_output_dir,
              csv_output_dir,
              session_id=argv[2],
              unit=argv[3],
              stimulus=argv[4],
              stimulus_blocks=argv[5]))
