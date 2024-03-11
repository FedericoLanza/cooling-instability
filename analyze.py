import argparse
import os
import meshio
from utils import parse_xdmf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import tri

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("folder", type=str, help="Folder") # positional argument: expects a string input (path to folder)
    parser.add_argument("--show", action="store_true", help="Show") # optional argument: typing --show enables the "show" feature
    parser.add_argument("--video", action="store_true", help="Video") # optional argument: typing --video enables the "video" feature
    return parser.parse_args()

if __name__ == "__main__":

    # create paths to the targeted files
    args = parse_args() # object containing the values of the parsed argument (folder path and optional '--show' flag)
    Tfile = os.path.join(args.folder, "T.xdmf")
    ufile = os.path.join(args.folder, "u.xdmf")
    pfile = os.path.join(args.folder, "p.xdmf")

    dsets_T, topology_address, geometry_address = parse_xdmf(Tfile, get_mesh_address=True) # extracts data from T.xdmf file
    dsets_T = dict(dsets_T) # converts data of T in a standard dictionary

    dsets_u = parse_xdmf(ufile, get_mesh_address=False) # extracts data from u.xdmf file
    dsets_u = dict(dsets_u)

    dsets_p = parse_xdmf(pfile, get_mesh_address=False) # extracts data from p.xdmf file
    dsets_p = dict(dsets_p)

    with h5py.File(topology_address[0], "r") as h5f:
        elems = h5f[topology_address[1]][:] # elements of triangular lattice
    
    with h5py.File(geometry_address[0], "r") as h5f:
        nodes = h5f[geometry_address[1]][:] # modes of triangular lattice
    
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

    # Analyze final state
    
    if True:
        beta = 0.001 # viscosity ratio

        t = t_[it_[-1]] # total time
        dset = dsets_T[t] # T-dictionary at final time
        with h5py.File(dset[0], "r") as h5f:
            T_[:] = h5f[dset[1]][:, 0] # takes values of T from the T-dictionary

        with h5py.File(dsets_p[t][0], "r") as h5f:
            p_[:] = h5f[dsets_p[t][1]][:, 0] # takes values of p from the p-dictionary

        T_intp = tri.CubicTriInterpolator(triang, T_) # interpolator from T values on triang lattice
        p_intp = tri.CubicTriInterpolator(triang, p_) # interpolator from p values on triang lattice

        Nx = 100
        Ny = 50

        x = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), Nx) # array with x-coordinates of the new mesh
        y = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), Ny) # array with y-coordinates of the new mesh

        X, Y = np.meshgrid(x, y) # array representing the mesh coordinates

        T_vals_ = T_intp(X, Y) # values of T obtained from interpolation on the meshgrid
        p_vals_ = p_intp(X, Y) # values of p obtained from interpolation on the meshgrid
        px_vals_, py_vals_ = p_intp.gradient(X, Y) # gradient of p

        ux_vals_ = -beta**-T_vals_ * px_vals_ # velocity field (u = beta^(-T) \nabla p)

        T_max_ = T_vals_.max(axis=0) # max of T along y at fixed t
        ux_max_ = ux_vals_.max(axis=0) # max of u_x along y at fixed t

        # Plot T and u_x along y for fixed x
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for i in range(Nx)[::10]:
            ax[0].plot(y, T_vals_[:, i], label=f"$x={x[i]:1.2f}$") # plot T(y) for different x
            ax[1].plot(y, ux_vals_[:, i]) # plot u_x(y) for different x
        ax[0].set_ylabel("$T$")
        ax[1].set_ylabel("$u_x$")
        ax[0].legend()
        [axi.set_xlabel("$y$") for axi in ax]
        fig.savefig(args.folder + '/fx.png', dpi=300)
        
        # Plot T and u_x along x for fixed y
        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
        for i in range(Ny)[::10]:
            ax2[0].plot(x, T_vals_[i, :], label=f"$y={y[i]:1.2f}$") # plot T(x) for different y
            ax2[1].plot(x, ux_vals_[i, :]) # plot u_x(x) for different y
        ax2[0].set_ylabel("$T$")
        ax2[1].set_ylabel("$u_x$")
        ax2[0].legend()
        [axi.set_xlabel("$x$") for axi in ax2]
        fig2.savefig(args.folder + '/fy.png', dpi=300)
        
        # Plot max of T and u_x along x for fixed y
        fig3, ax3 = plt.subplots(1, 2, figsize=(12, 4))
        ax3[0].plot(x, T_max_)
        ax3[1].plot(x, ux_max_)
        ax3[0].set_ylabel("$T_{max}(x)$")
        ax3[1].set_ylabel("$u_{x,max}(x)$")
        [axi.set_xlabel("$x$") for axi in ax3]
        fig3.savefig(args.folder + '/maxfy.png', dpi=300)
        
        # plt.show()
        
        #exit(0)
    # Analyze evolution
    if (args.show):
        figT, axT = plt.subplots(1, 1)
        def update(frame):
            t = t_[frame]  # time at step it
            print(f"it={frame} t={t}")
            dset = dsets_T[t]  # T-dictionary at time t
            with h5py.File(dset[0], "r") as h5f:
                T_[:] = h5f[dset[1]][:, 0]  # takes values of T from the T-dictionary
            axT.tripcolor(triang, T_) # plot of colormap of T
            axT.set_title(f"t = {t:1.2f}")
        num_frames = 10
        animation = FuncAnimation(figT, update, frames=num_frames, blit=False)
        animation.save(args.folder + 'evolving_colormap.mp4', fps=2, extra_args=['-vcodec', 'libx264'])
    
    plt.close()
    
    #exit(0)
    
    Tvideo_folder = os.path.join(args.folder, "T")
    uvideo_folder = os.path.join(args.folder, "u")
    if not os.path.exists(Tvideo_folder):
        os.makedirs(Tvideo_folder)
    if not os.path.exists(uvideo_folder):
        os.makedirs(uvideo_folder)
        
    for it in it_:
        t = t_[it] # time at step it
        print(f"it={it} t={t}")

        dset = dsets_T[t] # T-dictionary at time t
        with h5py.File(dset[0], "r") as h5f:
            T_[:] = h5f[dset[1]][:, 0] # takes values of T from the T-dictionary

        with h5py.File(dsets_u[t][0], "r") as h5f:
            u_[:, :] = h5f[dsets_u[t][1]][:, :2] # takes values of u from the T-dictionary

        figT, axT = plt.subplots(1, 1)
        axT.tripcolor(triang, T_) # plot of colormap of T
        cs = axT.tricontour(triang, T_, colors="k", levels=levels) # plot of different levels on the colormap
        axT.set_aspect("equal")
        figT.set_tight_layout(True)
        if (args.show and it == it_[-1]):
            figT.savefig(args.folder + '/Tlevelmap.png', dpi=300)
            plt.show()
        #if (args.show): # if you type "--show" on the command line
        #    axT.set_title(f"t = {t:1.2f}")
        #    figT.savefig(Tvideo_folder + f"/T_{it}.jpg", dpi=300)
        plt.close()
            
        if (args.show and it == it_[-1]): # if you type "--show" on the command line
            figu, axu = plt.subplots(1, 2, figsize=(9, 3))
            axu[0].tripcolor(triang, u_[:, 0]) # plot colormap of u_x
            axu[1].tripcolor(triang, u_[:, 1]) # plot colormap of u_y
            axu[0].set_title("$u_x$")
            axu[1].set_title("$u_y$")
            [axi.set_xlabel("$x$") for axi in axu]
            [axi.set_ylabel("$y$") for axi in axu]
            figu.suptitle(f"t = {t:1.2f}")
            figu.savefig(uvideo_folder + f'/umap.jpg', dpi=300)
            plt.close()
            
        paths = [] # curves formed by each level
        for level, path in zip(cs.levels, cs.get_paths()):
            if len(path.vertices): # if the path has non-null lenght
                paths.append((level, path.vertices))
        paths = dict(paths)

        for level, verts in paths.items():
            xmax[level][it] = verts[:, 0].max() # max x-position of a level
            xmin[level][it] = verts[:, 0].min() # min x-position of a level

        umax[it] = np.linalg.norm(u_, axis=0).max() # max of |u| at step it

    figf, axf = plt.subplots(1, 5, figsize=(30, 3))
    figf.subplots_adjust(wspace=0.3)
    for level in levels[1:-1]:
        #axf[0].plot(t_, xmax[level], label=f"$T={level:1.2f}$") # plot xmax vs t for each level
        axf[0].plot(t_, xmax[level]) # plot xmax vs t for each level
        axf[1].plot(t_, xmin[level]) # plot xmin vs t for each level
        axf[2].plot(t_[:len(it_)], xmax[level][:len(it_)] - xmin[level][:len(it_)]) # plot span vs t for each level
        axf[3].plot(t_[:len(it_)], xmax[level][:len(it_)] - xmin[level][:len(it_)]) # plot span vs t for each level

    axf[0].legend()
    [axi.set_xlabel("$t$") for axi in ax]
    axf[0].set_ylabel("$x_{max}$")
    axf[1].set_ylabel("$x_{min}$")
    axf[2].set_ylabel("$x_{max}-x_{min}$")
    axf[2].semilogy()
    axf[3].set_xscale('log')
    axf[3].set_yscale('log')
    axf[4].plot(t_[1:], umax[1:])
    axf[4].set_ylabel("$u_{max}$")
    figf.savefig(args.folder + '/fingergrowth.png', dpi=300)
    plt.show()
