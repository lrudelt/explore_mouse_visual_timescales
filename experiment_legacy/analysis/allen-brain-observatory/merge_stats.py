from sys import exit, argv, stderr, path
from os import listdir, replace
from os.path import isfile, isdir, realpath, dirname
import yaml
from subprocess import call
import pandas as pd

mre_analysis_dir = dirname(realpath(__file__))
stats_dir = f"{mre_analysis_dir}/../../paper_plots/data/stats"
two_timescales = True

def clean_up(csv_output_dir):
    call(f"rm {csv_output_dir}/*", shell = True)

def merge_csv_files(csv_output_dir, stats_dir, merged_file_name="mre_statistics_merged.csv"):
    single_file_prefix = 'mre_out'
    analysis_files = [f for f in sorted(listdir(csv_output_dir))
                    if f.startswith(single_file_prefix)]

    if len(analysis_files) == 0:
        print(f"No analysis files found in {csv_output_dir}. Aborting.")
        return

    if isfile("{}/{}".format(stats_dir,
                             merged_file_name)):
        replace("{}/{}".format(stats_dir, merged_file_name),
                "{}/{}.old".format(stats_dir, merged_file_name))

  
    df_merged = pd.read_csv(f"{csv_output_dir}/{analysis_files[0]}")
    print(df_merged["stimulus_blocks"])
    # with open("{}/{}".format(stats_dir,
    #                          merged_file_name), 'w') as merged_csv_file:
    #     stats_file = open("{}/{}".format(csv_output_dir,
    #                                      analysis_files[0]), 'r')
    #     header, content = stats_file.readlines()
    #     merged_csv_file.write(header)
    #     merged_csv_file.write(content)

        # stats_file.close()

    for i in range(1, len(analysis_files)):
        try:
            df_file = pd.read_csv(f"{csv_output_dir}/{analysis_files[i]}")
            # stats_file = open("{}/{}".format(csv_output_dir,
            #                                  analysis_files[i]), 'r')
            # this_header, content = stats_file.readlines()
            # if not this_header == header:
            #     print("The headers of the csv files in {} and {} do not match.  Please re-create the csv files.".format(analysis_files[0], analysis_files[i]))
            #     return
            df_merged = df_merged.append(df_file) 
            # merged_csv_file.write(content)
        except:
            print(analysis_files[i])

        # stats_file.close()

    # merged_csv_file.close()
    # Write to csv
    df_merged.to_csv(f"{stats_dir}/{merged_file_name}", index = False)

def print_usage_and_exit(script_name):
    print('usage is python3 {} Tmin Tmax'.format(script_name))
    exit()

if __name__ == "__main__":
    if not len(argv) == 3:
        print_usage_and_exit(argv[0])

    with open("mre_settings.yaml", 'r') as analysis_settings_file:
        analysis_settings = yaml.load(analysis_settings_file, Loader=yaml.BaseLoader)

    tmin = argv[1]
    tmax = argv[2]
    analysis = analysis_settings["analysis"]
    csv_output_path = analysis_settings["csv_output_path"]
    csv_output_dir = f"{csv_output_path}/csv_output_{tmin}_{tmax}_new"

    if not isdir(csv_output_dir):
        print(f'{csv_output_dir} not found. aborting.')
        exit()

    stats_file_name = f"{analysis}_mre_statistics_Tmin_{tmin}_Tmax_{tmax}.csv"
    print(csv_output_dir, stats_file_name)
    merge_csv_files(csv_output_dir, stats_dir, stats_file_name)
    exit(clean_up(csv_output_dir))

