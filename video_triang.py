import argparse
import os
import math
import meshio
from utils import parse_xdmf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import tri

#def parse_args():
#    parser = argparse.ArgumentParser(description="")
#    parser.add_argument("folder", type=str, help="Folder") # positional argument: expects a string input (path to folder)
#    parser.add_argument("--show", action="store_true", help="Show") # optional argument: typing --show enables the "show" feature
#    parser.add_argument("--video", action="store_true", help="Video") # optional argument: typing --video enables the "video" feature
#    return parser.parse_args()

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
    return parser.parse_args()

if __name__ == "__main__":

    # Parse the command-line arguments
    args = parse_args() # object containing the values of the parsed argument
    
    Lx = args.Lx # x-lenght of domain (system size)
    Ly = args.Ly # y-lenght of domain (wavelength)
    
    # global parameters
    Pe = args.Pe # Peclet number
    Gamma = args.Gamma # Heat transfer ratio
    beta = args.beta # Viscosity ratio ( nu(T) = beta^(-T) )
    
    # inlet parameters
    ueps = args.ueps # amplitude of the perturbation
    u0 = 1.0 # base inlet velocity
    
    # base state parameters
    Deff = 1./Pe + 2*Pe*u0*u0/105 # effective constant diffusion for the base state
    lambda_ = (- u0 + math.sqrt(u0*u0 + 4*Deff*Gamma)) / (2*Deff) # decay constant for the base state
    
    # flags
    rnd = args.rnd
    holdpert = args.holdpert
    
    Pe_str = f"Pe_{Pe:.10g}"
    Gamma_str = f"Gamma_{Gamma:.10g}"
    beta_str = f"beta_{beta:.10g}"
    ueps_str = f"ueps_{ueps:.10g}"
    Ly_str = f"Ly_{Ly:.10g}"
    Lx_str = f"Lx_{Lx:.10g}"
    rnd_str = f"rnd_{rnd}"
    holdpert_str = f"holdpert_{holdpert}"
    
    out_dir = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str, holdpert_str]) + "_right_200/" # directoty for output
    
    # Create paths to the targeted files
    Tfile = os.path.join(out_dir, "T.xdmf")
    ufile = os.path.join(out_dir, "u.xdmf")
    pfile = os.path.join(out_dir, "p.xdmf")

    dsets_T, topology_address, geometry_address = parse_xdmf(Tfile, get_mesh_address=True) # extracts data from T.xdmf file
    dsets_T = dict(dsets_T) # converts data of T in a standard dictionary

    dsets_u = parse_xdmf(ufile, get_mesh_address=False) # extracts data from u.xdmf file
    dsets_u = dict(dsets_u)

    dsets_p = parse_xdmf(pfile, get_mesh_address=False) # extracts data from p.xdmf file
    dsets_p = dict(dsets_p)

    with h5py.File(topology_address[0], "r") as h5f:
        elems = h5f[topology_address[1]][:] # elements of triangular lattice
    
    with h5py.File(geometry_address[0], "r") as h5f:
        nodes = h5f[geometry_address[1]][:] # nodes of triangular lattice
    
    triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], elems) # triangular lattice

    t_ = np.array(sorted(dsets_T.keys())) # time array
    it_ = list(range(len(t_))) # iteration steps

    T_ = np.zeros_like(nodes[:, 0]) # temperature field
    p_ = np.zeros_like(T_) # pressure field
    u_ = np.zeros((len(elems), 2)) # velocity field

    levels = np.linspace(0, 1, 11) # levels of T

    xmax = dict([(level, np.zeros_like(t_)) for level in levels]) # max x-position of a level for all time steps, for all levels
    xmin = dict([(level, np.zeros_like(t_)) for level in levels]) # min x-position of a level for all time steps, for all levels
    umax = np.zeros_like(t_) # max velocity for all time steps
    
    #fig, (axT, axu) = plt.subplots(1, 2)
    fig, axT = plt.subplots(1, 1, figsize=(15, 5))
    def update(frame):
        t = t_[frame]  # time at step it
        print(f"it={frame} t={t}")
            
        #Clear previous steps
        axT.clear()
        #axu.clear()
            
        # Update data for T
        dset_T = dsets_T[t]  # T-dictionary at time t
        with h5py.File(dset_T[0], "r") as h5f:
            T_[:] = h5f[dset_T[1]][:, 0]  # takes values of T from the T-dictionary
        axT.tripcolor(triang, T_) # plot of colormap of T
        axT.set_xlabel("$x$")
        axT.set_ylabel("$y$")
        axT.set_title(f"$T(x,y)$")
            
        # Update data for ux
        #dset_u = dsets_u[t] # u-dictionary at time t
        #with h5py.File(dset_u[0], "r") as h5f:
        #    u_[:, 0] = h5f[dset_u[1]][:, 0]  # takes values of ux from the u-dictionary
        #axu.tripcolor(triang, u_[:, 0]) # plot of colormap of ux
        #axu.set_title(f"$u_x(x,y)$")
        #del u_  # Clear u_ from memory
            
        general_title = f"$t = {t:1.2f}$, "
        if t < 0.1 and holdpert is False:
            general_title += f" $u_x(x=0) = u_0 + \\epsilon sin(2\pi y/L_y)$"
        else:
            if holdpert is False:
                general_title += f" $u_x(x=0) = u_0$"
            else:
                general_title += f" $u_x(x=0) = u_0 + \\epsilon sin(2\pi y/L_y)$"
        fig.suptitle(general_title)
        
        #plt.cla()  # Clear the current axes
                
    num_frames = len(t_)
    animation = FuncAnimation(fig, update, frames=num_frames, blit=False)
    animation.save(out_dir + '/animation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
