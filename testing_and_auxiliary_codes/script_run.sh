#!/bin/bash

dim=512
Rc_value=1
path=./Outputs/

namecode1="Testing_ElectronNuclearDynamics_BinaryIOData"
namecode2="Testing_ElectronNuclearDynamics_BinaryIOData_ScaleNuclearCoord"
namecode3="Testing_ElectronNuclearDynamics_BinaryIOData_v2"

# julia -O3 -t 4 ${namecode1}.jl > ${path}${dim}x${dim}_Rcval${Rc_value}_Output_${namecode1}.log
# julia -O3 -t 8 ${namecode2}.jl > ${path}${dim}x${dim}_Rcval${Rc_value}_Output_${namecode2}.log
julia -O3 -t 8 ${namecode3}.jl > ${path}${dim}x${dim}_Rcval${Rc_value}_Output_${namecode3}.log
