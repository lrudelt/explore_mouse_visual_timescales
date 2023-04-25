#!/bin/bash

# example for SGE cluster
# submit from project directory !!!

#$ -S /bin/bash   # which shell to use
#$ -N its_bn       # name of the job
#$ -q rostam.q       # which queue to use
#$ -l h_vmem=25G  # job is killed if exceeding this
#$ -cwd           # workers cd into current working directory first
#$ -o ./log/      # location for log files. directory must exist
#$ -j y

# set up environment and prevent multithreading
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# debugging variables in the log
date
uname -n

# load settings from home, activate conda environment
source /home/pspitzner/.bashrc
conda activate its_bn

# each worker pulls the right line with parameters from our tsv
line=$(awk "NR==$(($SGE_TASK_ID + 1))" ./run/parameters.tsv)

# print to log
echo $line

# run the command, respecting escaped strings
eval $line

date
