from sys import exit, argv, stderr, path
from os import listdir, replace
from os.path import isfile, isdir, realpath, dirname
import yaml

mre_analysis_dir = dirname(realpath(__file__))

def merge_csv_files(target_dir, merged_file_name="mre_statistics_merged.csv"):
    single_file_prefix = 'mre_out'

    analysis_files = [f for f in sorted(listdir(target_dir))
                      if f.startswith(single_file_prefix)]

    if len(analysis_files) == 0:
        print(f"No analysis files found in {target_dir}. Aborting.")
        return

    if isfile("{}/{}".format(mre_analysis_dir,
                             merged_file_name)):
        replace("{}/{}".format(mre_analysis_dir, merged_file_name),
                "{}/{}.old".format(mre_analysis_dir, merged_file_name))

    with open("{}/{}".format(mre_analysis_dir,
                             merged_file_name), 'w') as merged_csv_file:
        stats_file = open("{}/{}".format(target_dir,
                                         analysis_files[0]), 'r')
        header, content = stats_file.readlines()
        merged_csv_file.write(header)
        merged_csv_file.write(content)

        stats_file.close()
        
        for i in range(1, len(analysis_files)):
            try:
                stats_file = open("{}/{}".format(target_dir,
                                                 analysis_files[i]), 'r')
                this_header, content = stats_file.readlines()
                if not this_header == header:
                    print("The headers of the csv files in {} and {} do not match.  Please re-create the csv files.".format(analysis_files[0], analysis_files[i]))
                    return
                merged_csv_file.write(content)
            except:
                print(analysis_dirs[i])

        stats_file.close()

    merged_csv_file.close()

def print_usage_and_exit(script_name):
    print('usage is python3 {} settings_file_name'.format(script_name))
    exit()
    
if __name__ == "__main__":
    if not len(argv) == 2:
        print_usage_and_exit(argv[0])

    settings_file_name = argv[1]

    if not isfile(settings_file_name):
        print("Settings file {} not found.  Aborting.".format(spike_times_file_name),
              file=stderr, flush=True)
        exit(1)

    with open(settings_file_name, 'r') as analysis_settings_file:
        analysis_settings = yaml.load(analysis_settings_file, Loader=yaml.BaseLoader)

    csv_output_dir = analysis_settings["csv_output_dir"]

    if not isdir(csv_output_dir):
        print(f'{csv_output_dir} not found. aborting.')
        exit()

    stats_file_name = settings_file_name
    if stats_file_name.endswith(".yaml"):
        stats_file_name = stats_file_name[:-len(".yaml")]
    if stats_file_name.startswith("settings_"):
        stats_file_name = stats_file_name[len("settings_"):]
    if len(stats_file_name) > 0:
        stats_file_name = f"mre_statistics_merged_{stats_file_name}.csv"
    else:
        stats_file_name = f"mre_statistics_merged.csv"

    exit(merge_csv_files(csv_output_dir, stats_file_name))
