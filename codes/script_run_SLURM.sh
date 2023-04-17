#!/bin/bash

# +++ Metadatos para ejecutar con slurm
# +++ Nombre para identificar el trabajo. Por defecto es el nombre del script
#SBATCH --job-name=Rc1.5Grid180x180
#SBATCH --output=output_serafin_Rc_1.5_grid180x180.log
#SBATCH --partition=multi
# +++ Solicita N cores, por defecto se asignan consecutivamente dentro de un nodo. Si N excede la cantidad de cores continúa en otro nodo
#SBATCH --ntasks-per-node=1
# +++ Especifico en cuantos hilos correra cada uno de los procesos anteriores
#SBATCH --cpus-per-task=8
# +++ Especifico que no quiero a nadie más en el nodo cuando corra esta tarea
##SBATCH --exclusive
# +++ Envía un correo cuando el trabajo finaliza correctamente o por algún error
#SBATCH --mail-type=END
#SBATCH --mail-user=martinmendez@mi.unc.edu.ar
# +++ La linea siguiente es ignorada por Slurm porque empieza con ##
##SBATCH --mem-per-cpu=4G # cantidad de memoria por core

echo '++++++++++++++++++++++++++++++'
# srun conda env list
srun julia -O3 -t 8 03_Code_JuliaPure.jl
echo '++++++++++++++++++++++++++++++'


# run command: sbatch script.sh
# check squeue: squeue --me
