#!/bin/bash
# below are no comments but flags!
#$ -S /bin/bash
#$ -N allenMREp
#$ -q rostam.q
#$ -l h_vmem=6G                                         # job is killed if exceeding this
#$ -wd /scratch03.local/lucas/analysis/allen_brain_observatory              # run the job from this directory
#$ -o /scratch03.local/lucas/cluster_output/allen-brain-observatory/logs-mre/       # write log file here
#$ -e /scratch03.local/lucas/cluster_output/allen-brain-observatory/errors-mre/     # write log file here (errors)
#$ -t 1-6270 #for natural movie

# avoid multithreading in numpy
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python_dir=/home/lucas/anaconda2/envs/hierarchy_signatures/bin
analysis_dir=/scratch03.local/lucas/analysis/allen_brain_observatory
code_dir=/data.nst/lucas/history_dependence/signatures_of_temporal_processing_paper_code/cluster-jobs/allen-brain-observatory
spike_data_dir=/scratch03.local/lucas/analysis/allen_brain_observatory/spike_data
neuron_list_name=allen_neuron_list_merged_analysis_v2.tsv # for natural movie
settings_file_name=settings_30_10000.yaml

session_id=$(awk "NR==$(($SGE_TASK_ID + 2))" $code_dir/$neuron_list_name | awk '{ print $1 }')
# session_type=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $2 }')
unit=$(awk "NR==$(($SGE_TASK_ID + 2))" $code_dir/$neuron_list_name | awk '{ print $3 }')
# structure_acronym=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $4 }')
stimulus=$(awk "NR==$(($SGE_TASK_ID + 2))" $code_dir/$neuron_list_name | awk '{ print $5 }')
stimulus_blocks=$(awk "NR==$(($SGE_TASK_ID + 2))" $code_dir/$neuron_list_name | awk '{ print $6 }')

target_length=1080
transient=60
two_timescales=True

# print the start date to log
hostname
date


echo $python_dir/python3 $code_dir/print_spike_times_for_neuron.py $session_id $unit $stimulus $stimulus_blocks $target_length $transient $spike_data_dir \| $python_dir/python3 $code_dir/mre_analysis/mre_analysis.py /dev/stdin $session_id $unit $stimulus $stimulus_blocks $code_dir/mre_analysis/$settings_file_name $two_timescales

$python_dir/python3 $code_dir/print_spike_times_for_neuron.py $session_id $unit $stimulus $stimulus_blocks $target_length $transient $spike_data_dir | $python_dir/python3 $code_dir/mre_analysis/mre_analysis.py /dev/stdin $session_id $unit $stimulus $stimulus_blocks $code_dir/mre_analysis/$settings_file_name $two_timescales
date
