#!/bin/bash
#below are no comments but flags!
#$ -S /bin/bash
#$ -N hxSn
#$ -q rostam.q
#$ -l h_vmem=6G                                         # job is killed if exceeding this
#$ -wd /scratch03.local/dgmarx/hdestimator              # run the job from this directory
#$ -o /scratch03.local/dgmarx/jobs/hx_stdp_new2/logs/       # write log file here
#$ -e /scratch03.local/dgmarx/jobs/hx_stdp_new2/errors/     # write log file here (errors)
#$ -t 1-18750


# avoid multithreading in numpy
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python_dir=/scratch03.local/dgmarx/miniconda3/bin
estimator_dir=/scratch03.local/dgmarx/hdestimator
analysis_dir=/scratch03.local/dgmarx/analysis/hx_stdp_new2_static
neuron_list_name=neuron_list_25.tsv
settings_file_name=settings_30.yaml

kins=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $1 }')
filename=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $2 }')
seed=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $3 }')
neuronNum=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$neuron_list_name | awk '{ print $4 }')
scale=0

# print the start date to log
hostname
date

echo $python_dir/python3 $analysis_dir/print_spike_times.py $filename $neuronNum \| $python_dir/python3 $estimator_dir/estimate.py /dev/stdin --settings-file $analysis_dir/$settings_file_name --label $filename\;$neuronNum\;$kins\;$seed\;$scale

$python_dir/python3 $analysis_dir/print_spike_times.py $filename $neuronNum | $python_dir/python3 $estimator_dir/estimate.py /dev/stdin --settings-file $analysis_dir/$settings_file_name --label $filename\;$neuronNum\;$kins\;$seed\;$scale

date
