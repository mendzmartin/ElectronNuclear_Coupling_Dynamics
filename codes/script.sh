#!/bin/bash

#$ -cwd
#$ -m bea -M martinmendez@mi.unc.edu.ar
#$ -N julia100x100
#$ -j y
#$ -pe smp 4
#$ -o output_Rc_5.0_grid100x100.log
##$ -S /bin/bash
##$ -l mem_free=4.2G
##$ -v OMP_NUM_THREADS=4
##$ -R y
##$ -l h_rt=02:00:00
##$ -ckpt dmtcp
##$ -q long@compute-0-10.local

echo '+++++++++++++++++++++++++++++'
#conda init bash
#conda activate julia
conda env list
julia -O3 -t 4 03_Code_JuliaPure.jl
echo '+++++++++++++++++++++++++++++'
