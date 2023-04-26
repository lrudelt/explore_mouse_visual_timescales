#!/bin/bash
# below are no comments but flags!
#$ -S /bin/bash
#$ -N allenMREp
#$ -q rostam.q
#$ -l h_vmem=6G                                         # job is killed if exceeding this
#$ -wd /scratch03.local/lucas/analysis/allen_fc_merged              # run the job from this directory
#$ -o /scratch03.local/lucas/cluster_output/allen-fc-merged/logs-mre/       # write log file here
#$ -e /scratch03.local/lucas/cluster_output/allen-fc-merged/errors-mre/     # write log file here (errors)
# #$ -t 1-5971 # for spontaneous activity 
#$ -t 1-5677 # for natural movie


# avoid multithreading in numpy
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python_dir=/home/lucas/anaconda2/envs/hierarchy_signatures/bin
analysis_dir=/scratch03.local/lucas/analysis/allen_fc_merged
code_dir=/data.nst/lucas/history_dependence/signatures_of_temporal_processing_paper_code/cluster-jobs/allen-fc-merged
spike_data_dir=/scratch03.local/lucas/analysis/allen_fc_merged/spike_data
neuron_list_name=allen_neuron_list_merged_analysis_v3.tsv # for natural movie
# neuron_list_name=allen_neuron_list_merged_analysis_v3_sp.tsv # for spontaneous
settings_file_name=settings_15_10000.yaml

session_id=$(awk "NR==$(($SGE_TASK_ID + 2))" $code_dir/$neuron_list_name | awk '{ print $1 }')
# session_type=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $2 }')
unit=$(awk "NR==$(($SGE_TASK_ID + 2))" $code_dir/$neuron_list_name | awk '{ print $3 }')
# structure_acronym=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $4 }')
stimulus=$(awk "NR==$(($SGE_TASK_ID + 2))" $code_dir/$neuron_list_name | awk '{ print $5 }')
stimulus_blocks=$(awk "NR==$(($SGE_TASK_ID + 2))" $code_dir/$neuron_list_name | awk '{ print $6 }')

target_length=1080
transient=360
two_timescales=True
merged_blocks=False

# print the start date to log
hostname
date

echo $python_dir/python3 $code_dir/print_spike_times_for_neuron.py $session_id $unit $stimulus $stimulus_blocks $target_length $transient $spike_data_dir \| $python_dir/python3 $code_dir/mre_analysis.py /dev/stdin $session_id $unit $stimulus $stimulus_blocks $code_dir/$settings_file_name $two_timescales

$python_dir/python3 $code_dir/print_spike_times_for_neuron.py $session_id $unit $stimulus $stimulus_blocks $target_length $transient $spike_data_dir | $python_dir/python3 $code_dir/mre_analysis.py /dev/stdin $session_id $unit $stimulus $stimulus_blocks $code_dir/$settings_file_name $two_timescales

date
