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
    
    out_dir = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str, holdpert_str]) + "/" # directoty for output
    
    # Create paths to the targeted files
    Tfile = os.path.join(out_dir, "T.xdmf")
    ufile = os.path.join(out_dir, "u.xdmf")
    pfile = os.path.join(out_dir, "p.xdmf")

    dsets_T, topology_address, geometry_address = parse_xdmf(Tfile, get_mesh_address=True) # extracts dataset and paths for topology and geometry from T.xdmf file
    dsets_T = dict(dsets_T) # converts data of T in a standard dictionary

    dsets_u = parse_xdmf(ufile, get_mesh_address=False) # extracts data from u.xdmf file
    dsets_u = dict(dsets_u)

    dsets_p = parse_xdmf(pfile, get_mesh_address=False) # extracts data from p.xdmf file
    dsets_p = dict(dsets_p)
    
#    print('topology_address = ', topology_address)
#    print('type(topology_address) = ', type(topology_address))
#    print('geometry_address = ', geometry_address)
#    print('type(geometry_address) = ', type(geometry_address))
    
    with h5py.File(topology_address[0], "r") as h5f:
        elems = h5f[topology_address[1]][:] # elements of triangular lattice
    
    with h5py.File(geometry_address[0], "r") as h5f:
        nodes = h5f[geometry_address[1]][:] # nodes of triangular lattice
    
#    print('elems = ', elems)
#    print('type(elems) = ', type(elems))
#    print('shape(elems) = ', elems.shape)
#    print('nodes = ', nodes)
#    print('type(nodes) = ', type(nodes))
#    print('shape(nodes) = ', nodes.shape)
    
    # Prepare meshgrid
    x = nodes[:, 0]
    y = nodes[:, 1]
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    X, Y = np.meshgrid(np.unique(x), np.unique(y))
    
    print('len(np.unique(x)) = ', len(np.unique(x)))
    print('len(np.unique(y)) = ', len(np.unique(y)))
    
    # Sort indices of nodes array
    sort_indices = np.lexsort((nodes[:, 0], nodes[:, 1]))
    
    t_ = np.array(sorted(dsets_T.keys())) # time array
    it_ = list(range(len(t_))) # iteration steps
    
    t = t_[it_[-1]] # final time
    dset_T = dsets_T[t] # T-dictionary at final time
    
    T_ = np.zeros_like(nodes[:, 0]) # temperature field
    p_ = np.zeros_like(T_) # pressure field
    u_ = np.zeros((len(elems), 2)) # velocity field
    
    # Initialize overall minimum and maximum values
    overall_min_ux = float('inf')
    overall_max_ux = float('-inf')
    
    num_frames = len(t_)
    for frame in range(num_frames):
        t = t_[frame]  # time at step it
        
        dset_T = dsets_T[t]  # T-dictionary at time t
        with h5py.File(dset_T[0], "r") as h5f:
            T_[:] = h5f[dset_T[1]][:, 0]  # takes values of T from the T-dictionary
        T_sorted = T_[sort_indices]
        T_r = T_sorted.reshape((ny, nx))
        
        dset_p = dsets_p[t] # p-dictionary at time t
        with h5py.File(dset_p[0], "r") as h5f:
            p_[:] = h5f[dset_p[1]][:, 0] # takes values of p from the p-dictionary
        p_sorted = p_[sort_indices]
        p_r = p_sorted.reshape((ny, nx))
        
        grad_py, grad_px = np.gradient(p_r, np.unique(y), np.unique(x))
        ux_r = -beta**-T_r * grad_px
        
        overall_min_ux = min(overall_min_ux, np.min(ux_r))
        overall_max_ux = max(overall_max_ux, np.max(ux_r))
    
    # Prepare figures for videos
    figT, axT = plt.subplots(1, 1, figsize=(15, 5))
    axT.set_xlabel("$x$")
    axT.set_ylabel("$y$")
    axT.set_title("$T(x,y)$")
    im_T = axT.pcolormesh(X, Y, np.zeros_like(X), vmin=0., vmax=1.)
    cb_T = plt.colorbar(im_T, ax=axT) # colorbar
    
    figu, axu = plt.subplots(1, 1, figsize=(15, 5))
    axu.set_xlabel("$x$")
    axu.set_ylabel("$y$")
    axu.set_title("$u_x(x,y)$")
    im_ux = axu.pcolormesh(X, Y, np.zeros_like(X), vmin=overall_min_ux, vmax=overall_max_ux)
    cb_ux = plt.colorbar(im_ux, ax=axu) # colorbar
    
    def update_T(frame):
        t = t_[frame]  # time at step it
        print(f"it={frame} t={t}")
        
        #Clear previous steps
        axT.clear()
        
        #Update data for T
        dset_T = dsets_T[t]  # T-dictionary at time t
        with h5py.File(dset_T[0], "r") as h5f:
            T_[:] = h5f[dset_T[1]][:, 0]  # takes values of T from the T-dictionary
        T_sorted = T_[sort_indices]
        T_r = T_sorted.reshape((ny, nx))
        
        im_T = axT.pcolormesh(X, Y, T_r, vmin=0., vmax=1.) # plot of colormap of T
        
        general_title = f"$t = {t:1.2f}$, "
        if t < 0.1 and holdpert is False:
            general_title += f" $u_x(x=0) = u_0 + \\epsilon sin(2\pi y/L_y)$"
        else:
            if holdpert is False:
                general_title += f" $u_x(x=0) = u_0$"
            else:
                general_title += f" $u_x(x=0) = u_0 + \\epsilon sin(2\pi y/L_y)$"
        figT.suptitle(general_title)
        
        return im_T
        #plt.cla()  # Clear the current axes
    
    def update_ux(frame):
        t = t_[frame]  # time at step it
        print(f"it={frame} t={t}")
        
        #Clear previous steps
        axu.clear()
        
        # Update data for ux
        dset_T = dsets_T[t]  # T-dictionary at time t
        with h5py.File(dset_T[0], "r") as h5f:
            T_[:] = h5f[dset_T[1]][:, 0]  # takes values of T from the T-dictionary
        T_sorted = T_[sort_indices]
        T_r = T_sorted.reshape((ny, nx))
        
        dset_p = dsets_p[t] # p-dictionary at time t
        with h5py.File(dset_p[0], "r") as h5f:
            p_[:] = h5f[dset_p[1]][:, 0] # takes values of p from the p-dictionary
        p_sorted = p_[sort_indices]
        p_r = p_sorted.reshape((ny, nx))
        
        #print('len(p_r) = ', len(p_r))
        #print('len(np.unique(x)) = ', len(np.unique(x)))
        #print('len(np.unique(y)) = ', len(np.unique(y)))
        
        grad_py, grad_px = np.gradient(p_r, np.unique(y), np.unique(x))
        ux_r = -beta**-T_r * grad_px
        im_ux = axu.pcolormesh(X, Y, ux_r, vmin=overall_min_ux, vmax=overall_max_ux) # plot of colormap of ux
        
        general_title = f"$t = {t:1.2f}$, "
        if t < 0.1 and holdpert is False:
            general_title += f" $u_x(x=0) = u_0 + \\epsilon sin(2\pi y/L_y)$"
        else:
            if holdpert is False:
                general_title += f" $u_x(x=0) = u_0$"
            else:
                general_title += f" $u_x(x=0) = u_0 + \\epsilon sin(2\pi y/L_y)$"
        figu.suptitle(general_title)
        
        return im_ux
    
    print('making video for T')
    animation_T = FuncAnimation(figT, update_T, frames=num_frames, blit=False)
    animation_T.save(out_dir + '/T.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
    print('making video for ux')
    animation_ux = FuncAnimation(figu, update_ux, frames=num_frames, blit=False)
    animation_ux.save(out_dir + '/ux.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
