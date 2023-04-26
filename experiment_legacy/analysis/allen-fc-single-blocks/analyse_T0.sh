#!/bin/bash
#below are no comments but flags!
#$ -S /bin/bash
#$ -N allenV2
#$ -q rostam.q
#$ -l h_vmem=6G                                         # job is killed if exceeding this
#$ -wd /scratch03.local/dgmarx/hdestimator              # run the job from this directory
#$ -o /scratch03.local/dgmarx/jobs/allen-fc-single-blocks/logs/       # write log file here
#$ -e /scratch03.local/dgmarx/jobs/allen-fc-single-blocks/errors/     # write log file here (errors)
#$ -t 1-11290


# avoid multithreading in numpy
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python_dir=/scratch03.local/dgmarx/miniconda3/bin
estimator_dir=/scratch03.local/dgmarx/hdestimator
analysis_dir=/scratch03.local/dgmarx/analysis/allen_fc_single_blocks/
neuron_list_name=allen_neuron_list_unmerged_analysis_v3.tsv
settings_file_name=settings_30.yaml

session_id=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $1 }')
# session_type=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $2 }')
unit=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $3 }')
structure_acronym=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $4 }')
stimulus=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $5 }')
stimulus_blocks=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $6 }')

target_length=840
transient=60

# print the start date to log
hostname
date

echo $python_dir/python3 $analysis_dir/print_spike_times_for_neuron.py $session_id $unit $stimulus $stimulus_blocks $target_length $transient \| $python_dir/python3 $estimator_dir/estimate.py /dev/stdin --settings-file $analysis_dir/$settings_file_name --label $session_id\;$unit\;$stimulus\;$stimulus_blocks\;$structure_acronym

$python_dir/python3 $analysis_dir/print_spike_times_for_neuron.py $session_id $unit $stimulus $stimulus_blocks $target_length $transient | $python_dir/python3 $estimator_dir/estimate.py /dev/stdin --settings-file $analysis_dir/$settings_file_name --label $session_id\;$unit\;$stimulus\;$stimulus_blocks\;$structure_acronym

date
