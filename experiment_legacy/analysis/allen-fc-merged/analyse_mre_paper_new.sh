#!/bin/bash
# below are no comments but flags!
#$ -S /bin/bash
#$ -N allenMREp
#$ -q rostam.q
#$ -l h_vmem=6G                                         # job is killed if exceeding this
#$ -wd /data.nst/lucas/history_dependence/signatures_of_temporal_processing_paper_code/cluster-jobs/allen-fc-merged              # run the job from this directory
#$ -o /scratch03.local/lucas/cluster_output/allen-fc-merged/logs-mre/       # write log file here
#$ -e /scratch03.local/lucas/cluster_output/allen-fc-merged/errors-mre/     # write log file here (errors)
# $ -t 1-5971 # for spontaneous activity 
# $ -t 1-5677 # for natural movie
# $ -t 1-5978 # for spontaneous activity 
#$ -t 1-5647 # for natural movie new 

# avoid multithreading in numpy
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python_dir=/home/lucas/anaconda2/envs/hierarchy_signatures/bin
analysis_dir=/scratch03.local/lucas/analysis/allen_fc_merged
code_dir=/data.nst/lucas/history_dependence/signatures_of_temporal_processing_paper_code/cluster-jobs/allen-fc-merged
data_dir=/data.nst/share/data/allen_visual_coding_neuropixels
valid_unit_list=valid_unit_list_allen_fc_movie.tsv # for natural movie new
# valid_unit_list=allen_neuron_list_merged_analysis_v3.tsv # for natural movie
# valid_unit_list=allen_neuron_list_merged_analysis_v3_sp.tsv # for spontaneous
settings_file_name=mre_settings.yaml

session_id=$(awk "NR==$(($SGE_TASK_ID + 2))" $code_dir/$valid_unit_list | awk '{ print $1 }')
# session_type=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$valid_unit_list | awk '{ print $3 }')
unit=$(awk "NR==$(($SGE_TASK_ID + 2))" $code_dir/$valid_unit_list | awk '{ print $3 }')
# structure_acronym=$(awk "NR==$(($SGE_TASK_ID + 1))" $analysis_dir/$valid_unit_list | awk '{ print $5 }')
stimulus=$(awk "NR==$(($SGE_TASK_ID + 2))" $code_dir/$valid_unit_list | awk '{ print $5 }')
stimulus_blocks=$(awk "NR==$(($SGE_TASK_ID + 2))" $code_dir/$valid_unit_list | awk '{ print $6 }')

target_length=1080
transient=360
tmin=05
tmax=20000

# print the start date to log
hostname
date

echo $python_dir/python3 $code_dir/mre_analysis.py $session_id $unit $stimulus $stimulus_blocks $target_length $transient $tmin $tmax $data_dir $code_dir/$settings_file_name

$python_dir/python3 $code_dir/mre_analysis.py $session_id $unit $stimulus $stimulus_blocks $target_length $transient $tmin $tmax $data_dir $code_dir/$settings_file_name
date
