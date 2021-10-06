#!/bin/bash

#GLOBAL PATHS
PY='/project/theorie/gmagni/miniconda3/envs/nnpdf/bin/python'

function submit_job () {

    # RUN SETUP
    RUNNAME=$1
    NCORES=$2
    WALLTIME=$3

    
    COMMAND=$PWD'/launch_'$RUNNAME'.sh'
    RUNNER_PATH=$PWD
    LOG_PATH=$PWD'/logs'
    
    LAUNCH=$PY' '$RUNNER_PATH'/evolve_backward.py > '$LOG_PATH'/output_'$RUNNAME'.log'


    [ -e $COMMAND ] && rm $COMMAND
    mkdir -p $LOG_PATH
    echo $LAUNCH >> $COMMAND
    chmod +x $COMMAND

    # submission
    qsub -q smefit -W group_list=smefit -l nodes=1:ppn=$NCORES  -l walltime=$WALLTIME $COMMAND # -l pvmem=16000mb
    # qsub -q long7  -W group_list=theorie -l nodes=1:ppn=$NCORES -l walltime=$WALLTIME $COMMAND
    # cleaning
    rm $COMMAND
}

# submit_job 'n3lo_matching_exact_long' '1' '96:00:00'
submit_job 'n3lo_matching_expanded_parallel_32' '32' '48:00:00'