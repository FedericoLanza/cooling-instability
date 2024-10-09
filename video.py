import argparse
import os
import math
import meshio
from utils import parse_xdmf
import h5py
import numpy as np
import matplotlib as mpl
#mpl.rcParams['animation.ffmpeg_path'] = '/path/to/ffmpeg'  # Set this to the actual path of ffmpeg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import tri
from scipy.interpolate import RectBivariateSpline

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
    parser.add_argument('ny', type=float, help='Value for tile density along y')
    parser.add_argument('rtol', type=float, help='Value for error function')
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
    
    # resolution parameters
    ny = args.ny
    rtol = args.rtol
    
    # flags
    rnd = args.rnd
    holdpert = args.holdpert
    
    Pe_str = f"Pe_{Pe:.10g}"
    Gamma_str = f"Gamma_{Gamma:.10g}"
    beta_str = f"beta_{beta:.10g}"
    ueps_str = f"ueps_{ueps:.10g}"
    Ly_str = f"Ly_{Ly:.10g}"
    Lx_str = f"Lx_{Lx:.10g}"
    ny_str = f"ny_{ny:.10g}"
    rtol_str = f"rtol_{rtol:.10g}"
    rnd_str = f"rnd_{rnd}"
    holdpert_str = f"holdpert_{holdpert}"
    
    out_dir = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str, holdpert_str, ny_str, rtol_str]) + "_2periods/" # directoty for output
    
    # Create paths to the targeted files
    Tfile = os.path.join(out_dir, "T.xdmf")
    #ufile = os.path.join(out_dir, "u.xdmf")
    pfile = os.path.join(out_dir, "p.xdmf")

    dsets_T, topology_address, geometry_address = parse_xdmf(Tfile, get_mesh_address=True) # extracts dataset and paths for topology and geometry from T.xdmf file
    dsets_T = dict(dsets_T) # converts data of T in a standard dictionary

    #dsets_u = parse_xdmf(ufile, get_mesh_address=False) # extracts data from u.xdmf file
    #dsets_u = dict(dsets_u)

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
    x_sort = np.unique(x)
    y_sort = np.unique(y)
    nx = len(x_sort)
    ny = len(y_sort)
    X, Y = np.meshgrid(x_sort, y_sort)
    
    x_min = min(nodes[:, 0])
    x_max = max(nodes[:, 0])
    nx_high_res = 400
    x_high_res = np.linspace(x_min, x_max, nx_high_res)
    X_high_res, Y_high_res = np.meshgrid(x_high_res, y_sort)
    
    # Sort indices of nodes array
    sort_indices = np.lexsort((nodes[:, 0], nodes[:, 1]))
    
    t_ = np.array(sorted(dsets_T.keys())) # time array
    it_ = list(range(len(t_))) # iteration steps
    
    t = t_[it_[-1]] # final time
    dset_T = dsets_T[t] # T-dictionary at final time
    
    T_ = np.zeros_like(nodes[:, 0]) # temperature field
    p_ = np.zeros_like(T_) # pressure field
    #u_ = np.zeros((len(elems), 2)) # velocity field
    
    # Initialize overall minimum and maximum values
    overall_min_ux = float('inf')
    overall_max_ux = float('-inf')
    overall_min_uy = float('inf')
    overall_max_uy = float('-inf')
    
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
        uy_r = -beta**-T_r * grad_py
        
        f_ux = RectBivariateSpline(np.unique(y), np.unique(x), ux_r)
        ux_r_high_res = f_ux(Y_high_res[:, 0], X_high_res[0, :])
        
        overall_min_ux = min(overall_min_ux, np.min(ux_r_high_res))
        overall_max_ux = max(overall_max_ux, np.max(ux_r_high_res))
    
        overall_min_uy = min(overall_min_uy, np.min(uy_r))
        overall_max_uy = max(overall_max_uy, np.max(uy_r))
        
    # Prepare figures for videos
    figT, axT = plt.subplots(1, 1, figsize=(15, 5))
    axT.set_xlabel("$x$", fontsize=16)
    axT.set_ylabel("$y$", fontsize=16)
    #axT.set_title("$T(x,y)$")
    im_T = axT.pcolormesh(X_high_res, Y_high_res, np.zeros_like(X_high_res), vmin=0., vmax=1.)
    cb_T = plt.colorbar(im_T, ax=axT) # colorbar
    cb_T.ax.tick_params(labelsize=14)
    
    figux, axux = plt.subplots(1, 1, figsize=(15, 5))
    axux.set_xlabel("$x$", fontsize=16)
    axux.set_ylabel("$y$", fontsize=16)
    #axux.set_title("$u_x(x,y)$")
    im_ux = axux.pcolormesh(X_high_res, Y_high_res, np.zeros_like(X_high_res), vmin=overall_min_ux, vmax=overall_max_ux)
    cb_ux = plt.colorbar(im_ux, ax=axux) # colorbar
    cb_ux.ax.tick_params(labelsize=14)
    
    figuy, axuy = plt.subplots(1, 1, figsize=(15, 5))
    axuy.set_xlabel("$x$", fontsize=16)
    axuy.set_ylabel("$y$", fontsize=16)
    #axuy.set_title("$u_x(x,y)$")
    im_uy = axuy.pcolormesh(X, Y, np.zeros_like(X), vmin=overall_min_uy, vmax=overall_max_uy)
    cb_uy = plt.colorbar(im_uy, ax=axuy) # colorbar
    cb_uy.ax.tick_params(labelsize=14)
    
    def update_T(frame):
        t = t_[frame]  # time at step it
        print(f"it={frame} t={t}")
        
        #Clear previous steps
        axT.clear()
        axT.set_xlabel("$x$", fontsize=18)
        axT.set_ylabel("$y$", fontsize=18)
        
        axT.set_title("$T(x,y)$, " + f"$t = {t:1.2f}$", fontsize=20)
        #axT.set_title("$T(x,y)$", fontsize=21)
        
        #Update data for T
        dset_T = dsets_T[t]  # T-dictionary at time t
        with h5py.File(dset_T[0], "r") as h5f:
            T_[:] = h5f[dset_T[1]][:, 0]  # takes values of T from the T-dictionary
        T_sorted = T_[sort_indices]
        T_r = T_sorted.reshape((ny, nx))
        
        # Interpolate the data to higher resolution using RectBivariateSpline
        f = RectBivariateSpline(y_sort, x_sort, T_r)
        T_r_high_res = f(Y_high_res[:, 0], X_high_res[0, :])
        
        im_T = axT.pcolormesh(X_high_res, Y_high_res, T_r_high_res, vmin=0., vmax=1.) # plot of colormap of T
        axT.tick_params(axis='both', which='major', labelsize=14)
        
        general_title = "$T(x,y)$" + f"$t = {t:1.2f}$"
        #if t < 0.1 and holdpert is False:
        #    general_title += f" $u_x(x=0) = u_0 + \\epsilon sin(2\pi y/L_y)$"
        #else:
        #    if holdpert is False:
        #        general_title += f" $u_x(x=0) = u_0$"
        #    else:
        #        general_title += f" $u_x(x=0) = u_0 + \\epsilon sin(2\pi y/L_y)$"
        #figT.suptitle(general_title)
        
        return im_T
        #plt.cla()  # Clear the current axes
    
    def update_ux(frame):
        t = t_[frame]  # time at step it
        print(f"it={frame} t={t}")
        
        #Clear previous steps
        axux.clear()
        axux.set_xlabel("$x$", fontsize=18)
        axux.set_ylabel("$y$", fontsize=18)
        
        #axux.set_title("$u_x(x,y)$, " + f"$t = {t:1.2f}$", fontsize=20)
        axux.set_title("$u_x(x,y)$", fontsize=21)
        
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
        
        grad_py, grad_px = np.gradient(p_r, y_sort, x_sort)
        ux_r = -beta**-T_r * grad_px
        
        # Interpolate the data to higher resolution using RectBivariateSpline
        f = RectBivariateSpline(y_sort, x_sort, ux_r)
        ux_r_high_res = f(Y_high_res[:, 0], X_high_res[0, :])
        
        im_ux = axux.pcolormesh(X_high_res, Y_high_res, ux_r_high_res, vmin=overall_min_ux, vmax=overall_max_ux) # plot of colormap of ux
        axux.tick_params(axis='both', which='major', labelsize=14)
        
        general_title = "$u_x(x,y)$" + f"$t = {t:1.2f}$"
        #if t < 0.1 and holdpert is False:
        #    general_title += f" $u_x(x=0) = u_0 + \\epsilon sin(2\pi y/L_y)$"
        #else:
        #    if holdpert is False:
        #        general_title += f" $u_x(x=0) = u_0$"
        #    else:
        #        general_title += f" $u_x(x=0) = u_0 + \\epsilon sin(2\pi y/L_y)$"
        #figux.suptitle(general_title)
        
        return im_ux
        
    def update_uy(frame):
        t = t_[frame]  # time at step it
        print(f"it={frame} t={t}")
        
        #Clear previous steps
        axuy.clear()
        axuy.set_xlabel("$x$", fontsize=16)
        axuy.set_ylabel("$y$", fontsize=16)
        
        axuy.set_title("$u_y(x,y)$, " + f"$t = {t:1.2f}$", fontsize=20)
        
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
        
        grad_py, grad_px = np.gradient(p_r, y_sort, x_sort)
        uy_r = -beta**-T_r * grad_py
        
        # Interpolate the data to higher resolution using RectBivariateSpline
        f = RectBivariateSpline(y_sort, x_sort, uy_r)
        uy_r_high_res = f(Y_high_res[:, 0], X_high_res[0, :])
        
        im_uy = axuy.pcolormesh(X_high_res, Y_high_res, uy_r_high_res, vmin=overall_min_uy, vmax=overall_max_uy) # plot of colormap of ux
        axuy.tick_params(axis='both', which='major', labelsize=14)
        
        general_title = "$u_y(x,y)$" + f"$t = {t:1.2f}$"
        #if t < 0.1 and holdpert is False:
        #    general_title += f" $u_x(x=0) = u_0 + \\epsilon sin(2\pi y/L_y)$"
        #else:
        #    if holdpert is False:
        #        general_title += f" $u_x(x=0) = u_0$"
        #    else:
        #        general_title += f" $u_x(x=0) = u_0 + \\epsilon sin(2\pi y/L_y)$"
        # figuy.suptitle(general_title)
        
        return im_uy
    
    print('making video for T')
    animation_T = animation.FuncAnimation(figT, update_T, frames=num_frames, blit=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    animation_T.save(out_dir + '/T.mp4', writer=writer)
    #print('making video for ux')
    #animation_ux = FuncAnimation(figux, update_ux, frames=num_frames, blit=False)
    #animation_ux.save(out_dir + '/ux.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
    #print('making video for uy')
    #animation_uy = FuncAnimation(figuy, update_uy, frames=num_frames, blit=False)
    #animation_uy.save(out_dir + '/uy.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
