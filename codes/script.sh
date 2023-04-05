#!/bin/bash

#$ -cwd
#$ -m bea -M martinmendez@mi.unc.edu.ar
#$ -N julia_parallel
#$ -j y
##$ -S /bin/bash
##$ -l mem_free=4.2G
#$ -pe smp 4
#$ -v OMP_NUM_THREADS=4
#$ -R y
#$ -l h_rt=05:00:00
##$ -ckpt dmtcp
#$ -o output_Rc_5.0_grid50x50.log
# Pide este nodo
##$ -q long@compute-0-10.local

echo '+++++++++++++++++++++++++++++'
julia -t 4 03_Code_JuliaPure.jl
echo '+++++++++++++++++++++++++++++'
