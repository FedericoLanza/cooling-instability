# Create .sh file for "run_multiple_linear_model.py" and execute the job in it

import argparse
import subprocess
import numpy as np
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument("--Pe", default=10, type=float, help="Peclet number")
    parser.add_argument("--Gamma", default=1.0, type=float, help="Heat conductivity")
    parser.add_argument("--beta", default=1e-1, type=float, help="Viscosity ratio")
    parser.add_argument("--eps", default=1e-3, type=float, help="Perturbation amplide")
    parser.add_argument("--tpert", default=0.1, type=float, help="Perturbation duration")
    parser.add_argument("--dt", default=0.005, type=float, help="Timestep")
    parser.add_argument("--nx", default=4000, type=int, help="Number of mesh points")
    parser.add_argument("--Lx", default=800.0, type=float, help="System size")
    parser.add_argument("--tmax", default=20.0, type=float, help="Total time")
    parser.add_argument('--tp', action='store_true', help='Flag for executing linear_model_tp.py instead of linear_model_tu.py')
    parser.add_argument('--aesthetic', action='store_true', help='Flag for generating data for the plots to present in the article')
    parser.add_argument('--multi', action='store_true', help='Flag for executing the code in a range of values of Pe and Gamma (the input --Pe and --Gamma are then ignored)')
    return parser.parse_args()
    
def create_script(Pe, Gamma, beta, eps, tpert, dt, nx, Lx, tmax, tp, aesthetic):
    command_line = f"python3 run_multiple_linear_model.py --Pe {Pe:.10g} --Gamma {Gamma:.10g} --beta {beta:.10g} --eps {eps} --tpert {tpert} --dt {dt} --nx {nx} --Lx {Lx} --tmax {tmax}"
    # command_line = f"python3 run_multiple_linear_model.py -Pe {Pe} -Gamma {Gamma} -eps {eps} -tpert {tpert} -dt {dt} -nx {nx} -Lx {Lx} -tmax {tmax}"
    tvar = []
    aestheticvar = []
    if tp:
        tvar = "tp"
        command_line += " --tp"
    else:
        tvar = "tu"
    if aesthetic:
        command_line += " --aesthetic"
        aestheticvar = "_aesthetic"
    filename = f"run_multiple_linear_model_{tvar}_Pe_{Pe}_Gamma_{Gamma}_beta_{beta}_eps_{eps}_tpert_{tpert}_dt_{dt}_nx_{nx}_Lx_{Lx}_tmax_{tmax}{aestheticvar}.sh"
    script = f"""#!/bin/bash

# Job name:
#SBATCH --job-name=lin_{tvar}
# Slurm output file:
#SBATCH --output=run_multiple_linear_model_{tvar}_Pe_{Pe}_Gamma_{Gamma}_beta_{beta}_eps_{eps}_tpert_{tpert}_dt_{dt}_nx_{nx}_Lx_{Lx}_tmax_{aestheticvar}.out
# Number of tasks (processors):
#SBATCH --ntasks=1
# Memory (different units can be specified using the suffix K|M|G|T):
#SBATCH --mem=4G

# Activate FEniCS Conda environment
#source ~/.bashrc  # Ensure Conda is available in batch jobs
#conda activate fenics

# Job to execute
{command_line}

# Finish the script
exit 0
"""
# Maximum runtime, syntax is d-hh:mm:ss
#SBATCH --time=7-00:00:0

    with open(filename, 'w') as f:
        f.write(script)
    
    return filename

if __name__ == "__main__":
    
    args = parse_args() # object containing the values of the parsed argument

    Pe = args.Pe
    Gamma = args.Gamma
    #beta = args.beta

    eps = args.eps
    tpert = args.tpert
    dt = args.dt
    nx = args.nx
    Lx = args.Lx
    tmax = args.tmax
    tp = args.tp
    aesthetic = args.aesthetic
    multi = args.multi
    
    if multi == False:
    
        Pe = args.Pe
        Gamma = args.Gamma
        beta = args.beta
        
        # create file .sh
        filename = create_script(Pe, Gamma, beta, eps, tpert, dt, nx, Lx, tmax, tp, aesthetic)
                
        # submit the job
        subprocess.run(["sbatch", filename])
        
    else:
        Pe_ = [Pe]
        Gamma_ = [Gamma]
        #beta_ = [beta]
        
        #Pe_ = [10**a for a in np.arange(0., 6.51, 0.5)]
        #Gamma_ = [10**a for a in np.arange(-1, 2.1, 1.)]
        beta_ = [10**a for a in np.arange(-4.75, 1., 0.5)]
        
        for Pe in Pe_:
            for Gamma in Gamma_:
                for beta in beta_:
                    
                    # create file .sh
                    filename = create_script(Pe, Gamma, beta, eps, tpert, dt, nx, Lx, tmax, tp, aesthetic)
                        
                    # submit the job
                    subprocess.run(["sbatch", filename])
                    
                    # wait one second
                    time.sleep(1)
                
                
