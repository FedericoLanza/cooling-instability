#!/bin/bash

# Job name:
#SBATCH --job-name=myfirstjob
# Slurm output file:
#SBATCH --output=munin_%j.out
# Number of tasks (processors):
#SBATCH --ntasks=64
# Memory (different units can be specified using the suffix K|M|G|T):
#SBATCH --mem=25G
# Maximum runtime, syntax is d-hh:mm:ss
#SBATCH --time=7-00:00:0

# Job to execute
mpirun python3 cooling.py 10.0 1.0 0.001 0.001 50.0 2.828427125 --rnd

# Finish the script
exit 0
