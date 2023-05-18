#!/bin/bash

#$ -cwd
#$ -m bea -M martinmendez@mi.unc.edu.ar
#$ -N g100Rc1.5
#$ -j y
#$ -pe smp 8
#$ -q long@compute-0-26,long@compute-0-27,long@compute-0-22,long@compute-0-23,long@compute-0-24

./script_run.sh

# run command: qsub script.sh
