# Create .sh file for "cooling.py" and execute the job in it

import argparse
import subprocess
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--Pe', default=100, type=float, help='Value for Peclet number')
    parser.add_argument('--Gamma', default=1, type=float, help='Value for heat transfer ratio')
    parser.add_argument('--beta', default=1e-3, type=float, help='Value for viscosity ratio')
    parser.add_argument('--ueps', default=0.001, type=float, help='Value for amplitude of the perturbation')
    parser.add_argument('--Ly', default=2, type=float, help='Value for wavelength')
    parser.add_argument('--Lx', default=50, type=float, help='Value for system size')
    parser.add_argument('--dt', default=0.005, type=float, help='Value for time interval')
    parser.add_argument('--t_pert', default=0.1, type=float, help='Value for perturbation time')
    parser.add_argument('--t_end', default=20, type=float, help='Value for final time')
    parser.add_argument('--rnd',action='store_true', help='Flag for random velocity at inlet')
    parser.add_argument('--holdpert',action='store_true', help='Flag for maintaining the perturbation at all times')
    parser.add_argument('--many', action='store_true', help='Flag for starting multiple jobs varying a parameter')
    return parser.parse_args()
    
def create_script(Pe, Gamma, beta, ueps, Ly, Lx, rnd, holdpert, dt, t_pert, t_end):
    command_line = f"mpirun python3 cooling.py --Pe {Pe} --Gamma {Gamma} --beta {beta} --ueps {ueps} --Ly {Ly} --Lx {Lx} --dt {dt} --t_pert {t_pert} --t_end {t_end}"
    if rnd:
         command_line = command_line + " --rnd"
    if holdpert:
         command_line = command_line + " --holdpert"
    filename = f"Pe_{Pe}_Gamma_{Gamma}_beta_{beta}_ueps_{ueps}_Ly_{Ly}_Lx_{Lx}_rnd_{rnd}_holdpert_{holdpert}.sh"
    script = f"""#!/bin/bash

# Job name:
#SBATCH --job-name=cooling
# Slurm output file:
#SBATCH --output=Pe_{Pe}_Gamma_{Gamma}_beta_{beta}_ueps_{ueps}_Ly_{Ly}_Lx_{Lx}_rnd_{rnd}_holdpert_{holdpert}.out
# Number of tasks (processors):
#SBATCH --ntasks=64
# Memory (different units can be specified using the suffix K|M|G|T):
#SBATCH --mem=25G

# Job to execute
{command_line}

# Finish the script
exit 0
"""
# Maximum runtime, syntax is d-hh:mm:ss
#SBATCH --time=7-00:00:0

    with open(filename, 'w') as f:
        f.write(script)

if __name__ == "__main__":
    
    args = parse_args() # object containing the values of the parsed argument
    
    # Parameters
    Pe = args.Pe # Peclet number
    Gamma = args.Gamma # Heat transfer ratio
    beta = args.beta # Viscosity ratio ( nu(T) = beta^(-T) )
    ueps = args.ueps # amplitude of the perturbation
    Lx = args.Lx # x-lenght of domain (system size)
    Ly = args.Ly # y-lenght of domain (wavelength)
    dt = args.dt # time interval
    t_pert = args.t_pert # perturbation time
    t_end = args.t_end # final time
    rnd = args.rnd
    holdpert = args.holdpert
    many = args.many
    
    if many is True:
        Pe_ = [pow(10,a) for a in np.arange(0., 3.75, 0.5)] # List of multiple values
        for Pe_value in Pe_:
            filename = f"Pe_{Pe_value}_Gamma_{Gamma}_beta_{beta}_ueps_{ueps}_Ly_{Ly}_Lx_{Lx}_rnd_{rnd}_holdpert_{holdpert}.sh"
    
            # create file .sh
            create_script(Pe_value, Gamma, beta, ueps, Ly, Lx, rnd, holdpert, dt, t_pert, t_end)
            
            # submit the job
            subprocess.run(["sbatch", filename])
    else:
        filename = f"Pe_{Pe}_Gamma_{Gamma}_beta_{beta}_ueps_{ueps}_Ly_{Ly}_Lx_{Lx}_rnd_{rnd}_holdpert_{holdpert}.sh"
    
        # create file .sh
        create_script(Pe, Gamma, beta, ueps, Ly, Lx, rnd, holdpert, dt, t_pert, t_end)
            
        # submit the job
        subprocess.run(["sbatch", filename])
