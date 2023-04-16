#!/bin/bash

#$ -cwd
#$ -m bea -M martinmendez@mi.unc.edu.ar
#$ -N g512Rc1.5
#$ -j y
#$ -pe smp 8
#$ -o output_Rc_1.5_grid512x512.log
#$ -q long@compute-0-26,long@compute-0-27,long@compute-0-22,long@compute-0-23,long@compute-0-24


##$ -l h=!compute-0-12&!compute-0-11&!compute-0-10&!compute-0-9&!compute-0-13&!compute-0-7&!compute-0-6&!compute-0-5
##$ -S /bin/bash
##$ -l mem_free=4.2G
##$ -v OMP_NUM_THREADS=8
##$ -R y
##$ -l h_rt=02:00:00
##export DMTCP_DL_PLUGIN=0
##$ -ckpt dmtcp
##$ -q long@compute-0-10.local

echo '+++++++++++++++++++++++++++++'
#conda init bash
#conda activate julia
#conda env list
julia -O3 -t 8 03_Code_JuliaPure.jl
echo '+++++++++++++++++++++++++++++'


# run command: qsub script.sh
