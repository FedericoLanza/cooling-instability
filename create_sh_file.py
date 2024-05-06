import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('Pe', type=float, help='Value for Peclet number')
    parser.add_argument('Gamma', type=float, help='Value for heat transfer ratio')
    parser.add_argument('beta', type=float, help='Value for viscosity ratio')
    parser.add_argument('ueps', type=float, help='Value for amplitude of the perturbation')
    parser.add_argument('Ly', type=float, help='Value for wavelength')
    parser.add_argument('Lx', type=float, help='Value for system size')
    parser.add_argument('--rnd',action='store_true', help='Flag for random velocity at inlet')
    parser.add_argument('--holdpert',action='store_true', help='Flag for maintaining the perturbation at all times')
    # parser.add_argument("--show", action="store_true", help="Show") # optional argument: typing --show enables the "show" feature
    parser.add_argument("--snap", action="store_true", help="Snap") # optional argument: typing --snap enables the "snap" feature
    return parser.parse_args()
    
def create_script(Pe, Gamma, beta, ueps, Lx, Ly, rnd, holdpert):
    command_line = f"mpirun python3 cooling.py {Pe} {Gamma} {beta} {ueps} {Lx} {Ly}"
    if rnd:
         command_line = command_line + " --rnd"
    if holdpert:
         command_line = command_line + " --holdpert"
    filename = f"Pe_{Pe}_Gamma_{Gamma}_beta_{beta}_ueps_{ueps}_Lx_{Lx}_Ly_{Ly}_rnd_{rnd}_holdpert_{holdpert}.sh"
    script = f"""#!/bin/bash

# Job name:
#SBATCH --job-name=cooling
# Slurm output file:
#SBATCH --output=Pe_{Pe}_Gamma_{Gamma}_beta_{beta}_ueps_{ueps}_Lx_{Lx}_Ly_{Ly}_rnd_{rnd}_holdpert_{holdpert}.out
# Number of tasks (processors):
#SBATCH --ntasks=64
# Memory (different units can be specified using the suffix K|M|G|T):
#SBATCH --mem=25G
# Maximum runtime, syntax is d-hh:mm:ss
#SBATCH --time=7-00:00:0

# Job to execute
{command_line}

# Finish the script
exit 0
"""
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
    rnd = args.rnd
    holdpert = args.holdpert
    
    filename = f"Pe_{Pe}_Gamma_{Gamma}_beta_{beta}_ueps_{ueps}_Lx_{Lx}_Ly_{Ly}_rnd_{rnd}_holdpert_{holdpert}.sh"
    
    # create file .sh
    create_script(Pe, Gamma, beta, ueps, Lx, Ly, rnd, holdpert)
    
    # submit the job
    subprocess.run(["sbatch", filename])

    
    
