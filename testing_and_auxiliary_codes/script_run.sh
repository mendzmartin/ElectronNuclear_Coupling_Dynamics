#!/bin/bash

dim=400
Rc_value=1
path=./Outputs/

#julia -O3 -t 4 Testing_ElectronNuclearDynamics_BinaryIOData.jl > ${path}${dim}x${dim}_Rcval${Rc_value}_Output_Testing_ElectronNuclearDynamics_BinaryIOData.log
julia -O3 -t 8 Testing_ElectronNuclearDynamics_BinaryIOData_ScaleNuclearCoord.jl > ${path}${dim}x${dim}_Rcval${Rc_value}_Output_Testing_ElectronNuclearDynamics_BinaryIOData_ScaleNuclearCoord.log
