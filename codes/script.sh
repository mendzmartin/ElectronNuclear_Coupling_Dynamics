#!/bin/bash

#$ -cwd
#$ -m bea -M martinmendez@mi.unc.edu.ar
#$ -N g150Rc1.5
#$ -j y
#$ -pe smp 4
#$ -o output_Rc_1.5_grid150x150.log
#$ -l h=!compute-0-12&!compute-0-11&!compute-0-10&!compute-0-9&!compute-0-13
##$ -S /bin/bash
##$ -l mem_free=4.2G
##$ -v OMP_NUM_THREADS=4
##$ -R y
##$ -l h_rt=02:00:00
##export DMTCP_DL_PLUGIN=0
##$ -ckpt dmtcp
##$ -q long@compute-0-10.local

echo '+++++++++++++++++++++++++++++'
#conda init bash
#conda activate julia
#conda env list
julia -O3 -t 4 03_Code_JuliaPure.jl
echo '+++++++++++++++++++++++++++++'


# run command: qsub script.sh
