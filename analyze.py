import argparse
import h5py
import math
import matplotlib.pyplot as plt
import meshio
import numpy as np
import os
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib import tri
from scipy.interpolate import RectBivariateSpline
from sklearn.linear_model import LinearRegression
from utils import parse_xdmf

# ciao

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--Pe', default=100, type=float, help='Value for Peclet number')
    parser.add_argument('--Gamma', default=1, type=float, help='Value for heat transfer ratio')
    parser.add_argument('--beta', default=1e-3, type=float, help='Value for viscosity ratio')
    parser.add_argument('--ueps', default=0.001, type=float, help='Value for amplitude of the perturbation')
    parser.add_argument('--Ly', default=2, type=float, help='Value for wavelength')
    parser.add_argument('--Lx', default=50, type=float, help='Value for system size')
    parser.add_argument('--dt', default=0.005, type=float, help='Value for time interval')
    #parser.add_argument('ny', type=float, help='Value for tile density along y')
    #parser.add_argument('rtol', type=float, help='Value for error function')
    parser.add_argument('--rnd',action='store_true', help='Flag for random velocity at inlet')
    parser.add_argument('--holdpert',action='store_true', help='Flag for maintaining the perturbation at all times')
    parser.add_argument('--constDeltaP',action='store_true', help='Flag for imposing constant pressure, instead of constant flow rate, at the inlet boundary')
    parser.add_argument('--print_colormaps',action='store_true', help='Flag for printing 2d colormaps')
    parser.add_argument('--print_profiles',action='store_true', help='Flag for printing 1d profiles')
    parser.add_argument('--final', action='store_true', help='Flag for analyzing the final state')
    parser.add_argument('--latex', action='store_true', help='Flag for plotting in LaTeX style')
    # parser.add_argument("--show", action="store_true", help="Show") # optional argument: typing --show enables the "show" feature
    return parser.parse_args()

if __name__ == "__main__":

    cmap_space = sns.color_palette("mako", as_cmap=True)
    
    # Parse the command-line arguments
    args = parse_args() # object containing the values of the parsed argument
    
    Lx = args.Lx # x-lenght of domain (system size)
    Ly = args.Ly # y-lenght of domain (wavelength)
    dt = args.dt #time step
    
    # global parameters
    Pe = args.Pe # Peclet number
    Gamma = args.Gamma # Heat transfer ratio
    beta = args.beta # Viscosity ratio ( nu(T) = beta^(-T) )
    
    # inlet parameters
    ueps = args.ueps # amplitude of the perturbation
    u0 = 1.0 # base inlet velocity
    
    # base state parameters
    kappa_eff = 1./Pe + 2*Pe*u0*u0/105 # effective constant diffusion for the base state
    xi = ( - u0 + math.sqrt(u0*u0 + 4*kappa_eff*Gamma)) / (2*kappa_eff) # decay constant for the base state
    
    # resolution parameters
    #ny = args.ny
    #rtol = args.rtol
    
    # flags
    rnd = args.rnd
    holdpert = args.holdpert
    constDeltaP = args.constDeltaP
    final = args.final
    latex = args.latex
    
    if latex:
        plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 24,
        "axes.labelsize": 24,  # Axis labels (JFM ~8pt)
        "xtick.labelsize": 24,  # Tick labels
        "ytick.labelsize": 24,
        "legend.fontsize": 12,  # Legend size
        "lines.linewidth": 1.5,
        "lines.markersize": 8,
        "figure.subplot.wspace": 0.35,  # Horizontal spacing
        "figure.subplot.bottom": 0.15,  # Space for x-labels
        "axes.labelpad": 8,
        })
    
    Pe_str = f"Pe_{Pe:.10g}"
    Gamma_str = f"Gamma_{Gamma:.10g}"
    beta_str = f"beta_{beta:.10g}"
    ueps_str = f"ueps_{ueps:.10g}"
    Ly_str = f"Ly_{Ly:.10g}"
    Lx_str = f"Lx_{Lx:.10g}"
    #ny_str = f"ny_{ny:.10g}"
    #rtol_str = f"rtol_{rtol:.10g}"
    rnd_str = f"rnd_{rnd}"
    holdpert_str = f"holdpert_{holdpert}"
    
    out_dir = "results/"
    if constDeltaP:
        out_dir += "constDeltaP_"
    out_dir += "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str, holdpert_str]) + "/" # directoty for output
    
    # Create paths to the targeted files
    Tfile = os.path.join(out_dir, "T.xdmf")
    #ufile = os.path.join(out_dir, "u.xdmf")
    pfile = os.path.join(out_dir, "p.xdmf")

    dsets_T, topology_address, geometry_address = parse_xdmf(Tfile, get_mesh_address=True) # extracts data from T.xdmf file
    dsets_T = dict(dsets_T) # converts data of T in a standard dictionary

    # dsets_u = parse_xdmf(ufile, get_mesh_address=False) # extracts data from u.xdmf file
    # dsets_u = dict(dsets_u)

    dsets_p = parse_xdmf(pfile, get_mesh_address=False) # extracts data from p.xdmf file
    dsets_p = dict(dsets_p)

    with h5py.File(topology_address[0], "r") as h5f:
        elems = h5f[topology_address[1]][:] # elements of triangular lattice
    
    with h5py.File(geometry_address[0], "r") as h5f:
        nodes = h5f[geometry_address[1]][:] # modes of triangular lattice
    
    # Prepare meshgrid
    x = nodes[:, 0]
    y = nodes[:, 1]
    x_sort = np.unique(x)
    y_sort = np.unique(y)
    nx = len(x_sort)
    ny = len(y_sort)
    print ("nx = ", nx, ", ny = ",ny)
    X, Y = np.meshgrid(x_sort, y_sort)
    
    x_min = min(nodes[:, 0])
    x_max = max(nodes[:, 0])
    y_min = min(nodes[:, 1])
    y_max = max(nodes[:, 1])
    
    nx_high_res = 400
    ny_low_res = ny//2
    x_high_res = np.linspace(x_min, x_max, nx_high_res)
    y_low_res = np.linspace(y_min, y_max, ny_low_res)
    
    X_high_res, Y_high_res = np.meshgrid(x_high_res, y_sort)
    X_low_res, Y_low_res = np.meshgrid(x_sort, y_low_res)
    
    # Sort indices of nodes array
    sort_indices = np.lexsort((nodes[:, 0], nodes[:, 1]))

    t_ = np.array(sorted(dsets_T.keys())) # time array
    it_ = list(range(len(t_))) # iteration steps

    T_ = np.zeros_like(nodes[:, 0]) # temperature field
    p_ = np.zeros_like(T_) # pressure field
    # u_ = np.zeros((len(elems), 2)) # velocity field

    n_steps = len(it_)
    dump_intv = 10 # update manually
    dt_save = dt*dump_intv
    t_end = n_steps*dt_save
    print("t_end = ", t_end)
    levels = np.linspace(0, 1, 11) # levels of T
    xmax = dict([(level, np.zeros_like(t_)) for level in levels]) # max x-position of a level for all time steps, for all levels
    xmin = dict([(level, np.zeros_like(t_)) for level in levels]) # min x-position of a level for all time steps, for all levels
    umax = np.zeros_like(t_) # max velocity for all time steps
    Tprime_max = np.zeros_like(t_) # max velocity for all time steps
    
    #Tk = np.zeros(nx_high_res, n_steps//10)
    n_samples = 10
    Txt = np.zeros((n_samples, n_steps))
    uxt = np.zeros((n_samples, n_steps))
    
    # Analyze final state
    
    cmap = plt.cm.viridis
    
    if final == True:
        #beta = 0.001 # viscosity ratio

        t = t_[it_[8*n_steps//10]] # final time
        dset_T = dsets_T[t] # T-dictionary at final time
        with h5py.File(dset_T[0], "r") as h5f:
            T_[:] = h5f[dset_T[1]][:, 0] # takes values of T from the T-dictionary
        T_sorted = T_[sort_indices]
        T_r_ = T_sorted.reshape((ny, nx))
        
        with h5py.File(dsets_p[t][0], "r") as h5f:
            p_[:] = h5f[dsets_p[t][1]][:, 0] # takes values of p from the p-dictionary
        p_sorted = p_[sort_indices]
        p_r_ = p_sorted.reshape((ny, nx))

        grad_py, grad_px = np.gradient(p_r_, y_sort, x_sort) # gradient of p
        ux_r_ = -beta**-T_r_ * grad_px # x-component of velocity field (u = beta^(-T) \nabla p)

        # Interpolate the data to higher resolution using RectBivariateSpline
        f_T = RectBivariateSpline(y_sort, x_sort, T_r_)
        T_r = f_T(Y_high_res[:, 0], X_high_res[0, :])
        T_r_frame = f_T(Y_low_res[:, 0], X_low_res[0, :])
        
        if rnd:
            T_2 = np.zeros_like(nodes[:, 0]) # temperature field
            
            t2 = t_[it_[n_steps//2]] # final time
            dset_T2 = dsets_T[t2] # T-dictionary at final time
            with h5py.File(dset_T2[0], "r") as h5f:
                T_2[:] = h5f[dset_T2[1]][:, 0] # takes values of T from the T-dictionary
            T_sorted_2 = T_2[sort_indices]
            T_r_2 = T_sorted_2.reshape((ny, nx))
            
            f_T2 = RectBivariateSpline(y_sort, x_sort, T_r_2)
            T_r_early = f_T2(Y_high_res[:, 0], X_high_res[0, :])
            T_r_early_frame = f_T2(Y_low_res[:, 0], X_low_res[0, :])
        
        f_ux = RectBivariateSpline(y_sort, x_sort, ux_r_)
        ux_r = f_ux(Y_high_res[:, 0], X_high_res[0, :])
        
        Tprime_r = T_r - np.exp(-xi*X_high_res[:, :])
        uprime_r = ux_r - 1.
        
        T_max = T_r.max(axis=0) # max of T along y at fixed t
        ux_max = ux_r.max(axis=0) # max of u_x along y at fixed t
        Tprime_max = Tprime_r.max(axis=0) # max of T along y at fixed t
        uprime_max = uprime_r.max(axis=0) # max of u_x along y at fixed t

        if print_profiles:
        
            # Plot T and u_x along y for fixed x
            figx, axx = plt.subplots(2, 2, figsize=(7.5, 5))
            for i in range(nx_high_res)[::25]:
                color = cmap(i / (nx_high_res - 1))  # Adjust the color according to column index
                axx[0,0].plot(y_sort, T_r[:, i], label=f"$x={x_high_res[i]:1.2f}$", color=color) # plot T(y) for different x
                axx[1,0].plot(y_sort, ux_r[:, i], label=f"$x={x_high_res[i]:1.2f}$", color=color) # plot ux(y) for different x
                axx[0,1].plot(y_sort, Tprime_r[:, i], label=f"$x={x_high_res[i]:1.2f}$", color=color) # plot T(y) for different x
                axx[1,1].plot(y_sort, uprime_r[:, i], label=f"$x={x_high_res[i]:1.2f}$", color=color) # plot ux(y) for different x
            axx[0,0].set_ylabel("$T$")
            axx[1,0].set_ylabel("$u_x$")
            axx[0,1].set_ylabel("$T'$")
            axx[1,1].set_ylabel("$u'_x$")
            [axi.set_xlabel("$y$") for axi in axx]
            figx.tight_layout()
            figx.savefig(out_dir + '/fx.pdf', dpi=300)
            
            figxlog, axxlog = plt.subplots(1, 2, figsize=(12, 4))
            for i in range(nx_high_res)[::25]:
                color = cmap(i / (nx_high_res - 1))  # Adjust the color according to column index
                axxlog[0].plot(np.log(y_sort - Ly/4), np.log(1-(T_r[:, i]/T_max[i])), label=f"$x={x_high_res[i]:1.2f}$", color=color) # plot T(y) for different x
                axxlog[1].plot(np.log(y_sort - Ly/4), np.log(1-(ux_r[:, i]/ux_max[i])), color=color) # plot u_x(y) for different x
            for i in range(nx_high_res)[::400]:
                # color = cmap(i / (nx - 1))  # Adjust the color according to column index
                sigma = np.sqrt(0.0037*x[i])
                gaussT = [ 2*(y_) - (-5 + 0.0057*x[i]) for y_ in np.linspace(-5,0,40)]
                gaussux = [ 2*(y_) - (-5 + 0.0007*x[i]) for y_ in np.linspace(-5,0,40)]
                axxlog[0].plot(np.linspace(-5,0,40), gaussT, linestyle='dotted', color='black')
                axxlog[1].plot(np.linspace(-5,0,40), gaussux, linestyle='dotted', color='black')
            axxlog[0].set_ylabel("$\log(1 - T(y')/T_m)$")
            axxlog[1].set_ylabel("$\log(1 - u_x(y')/u_m)$")
            [axi.set_xlabel("$\log(y')$") for axi in axxlog]
            figxlog.savefig(out_dir + '/fx_loglog.pdf', dpi=300)
            
            # Plot T and u_x along x for fixed y
            
            figy, axy = plt.subplots(2, 2, figsize=(12, 4))
            axy[0].plot(x_high_res, [np.exp(-x*xi) for x in x_high_res], color='black', linestyle='dashed') # plot the base state T_0(x)
            for i in range(ny)[::25]:
                color = cmap(i / (ny - 1))  # Adjust the color according to column index
                axy[0,0].plot(x_high_res, T_r[i, :], label=f"$y={y_sort[i]:1.2f}$", color=color) # plot T(x) for different y
                axy[1,0].plot(x_high_res, ux_r[i, :], color=color) # plot u_x(x) for different y
                axy[0,1].plot(x_high_res, T_r[i, :], label=f"$y={y_sort[i]:1.2f}$", color=color) # plot T(x) for different y
                axy[1,1].plot(x_high_res, ux_r[i, :], color=color) # plot u_x(x) for different y
            axy[0,0].set_ylabel("$T$")
            axy[1,0].set_ylabel("$u_x$")
            axy[0,0].set_ylabel("$T'$")
            axy[1,0].set_ylabel("$u'_x$")
            axy[0].legend()
            [axi.set_xlabel("$x$") for axi in axy]
            figy.savefig(out_dir + '/fy.pdf', dpi=300)
            
            # Plot max of T and u_x along x for fixed y
            
            figmax, axmax = plt.subplots(1, 2, figsize=(12, 4))
            axmax[0].plot(x_high_res, [np.exp(-x*xi) for x in x_high_res], color='black', linestyle='dashed')
            axmax[0].plot(x_high_res, T_max)
            axmax[1].plot(x_high_res, ux_max)
            axmax[0].set_ylabel("$T_{max}(x)$")
            axmax[1].set_ylabel("$u_{x,max}(x)$")
            [axi.set_xlabel("$x$") for axi in ax3]
            fig3.savefig(out_dir + '/maxfy.pdf', dpi=300)
        
        
        # Calculate uy
        uy_r_ = -beta**-T_r_ * grad_py
        f_uy = RectBivariateSpline(y_sort, x_sort, uy_r_)
        uy_r = f_uy(Y_high_res[:, 0], X_high_res[0, :])
        
        if False:
            # Plot colormaps of ux and uy at final state
            figu, axu = plt.subplots(1, 2, figsize=(12, 4))
            
            im_ux = axu[0].pcolormesh(X_high_res, Y_high_res, ux_r) # plot of colormap of ux
            cb_ux = plt.colorbar(im_ux, ax=axu[0]) # colorbar
            axu[0].set_title("$u_x$")
        
            im_uy = axu[1].pcolormesh(X_high_res, Y_high_res, uy_r) # plot of colormap of ux
            cb_uy = plt.colorbar(im_uy, ax=axu[1]) # colorbar
            axu[1].set_title("$u_y$")
            
            [axi.set_ylabel("$y$") for axi in axu]
            [axi.set_xlabel("$x$") for axi in axu]
            
            figu.suptitle(f"Final state ($t = {t:1.2f}$)")
            figu.savefig(out_dir + f'/umap.pdf', dpi=300)
        
        if print_colormaps:
            if (rnd == False):
                # Plot colormaps of T with levels and streamlines at final state
                figTs, axTs = plt.subplots(1, 2, figsize=(15, 5))
                
                im_T = axTs[0].pcolormesh(X_low_res, Y_low_res, T_r_frame, vmin=0., vmax=1., cmap='plasma', alpha=0.7, edgecolors='none', linewidth=0, shading='gouraud') # plot of colormap of T
                cb_T = plt.colorbar(im_T, ax=axTs[0], location='right', orientation='vertical') # colorbar
                cb_T.set_label(r'$T(x,y)$', labelpad=10)
                cb_T.ax.xaxis.set_label_position('top')  # Move label to the top
                cs = axTs[0].contour(X_high_res, Y_high_res, T_r, levels=levels, linewidths=0.9, colors="k") # plot of different levels on the colormap
                
                speed = np.sqrt(ux_r**2 + uy_r**2) # Compute speed for coloring
                strm = axTs[1].streamplot(X_high_res, Y_high_res, ux_r, uy_r, density=2, linewidth=0.8, arrowsize=0.8, color=speed, cmap='plasma') # Draw streamlines on the ax object
                cbar = plt.colorbar(strm.lines, ax=axTs[1], label=r'$|\textbf{u}|(x,y)$')
                
                figTs.patch.set_facecolor('white')
                [axi.set_facecolor('white') for axi in axTs]
                
                [axi.set_xlabel("$x$") for axi in axTs]
                [axi.set_ylabel("$y$") for axi in axTs]
                
                ylab_xpos_l_b = axTs[0].yaxis.get_label().get_position()[0]  # horizontal position of y-label
                ylab_xpos_r_b = axTs[0].yaxis.get_label().get_position()[1]  # horizontal position of y-label
                figTs.text(ylab_xpos_l_b + 0.075, 0.98, "($a$)", verticalalignment='top', horizontalalignment='right')
                figTs.text(ylab_xpos_r_b + 0.025, 0.98, "($b$)", verticalalignment='top', horizontalalignment='right')
                
                Lxlim = 10 if rnd else 22.5
                [axi.set_xlim(0, Lxlim) for axi in axTs]
                [axi.set_ylim(0, Ly) for axi in axTs]
                [axi.set_aspect('auto') for axi in axTs]
                
                imgname = '/Tlevel_and_streamlines_sin.pdf'
                figTs.savefig(out_dir + imgname, dpi=400, bbox_inches="tight")
               
            else:
                figTT, axTT = plt.subplots(1, 2, figsize=(15., 5.))
                
                im_Te = axTT[0].pcolormesh(X_low_res, Y_low_res, T_r_early_frame, vmin=0., vmax=1., cmap='plasma', alpha=0.7, edgecolors='none', linewidth=0, shading='gouraud') # plot of colormap of T
                cb_Te = plt.colorbar(im_Te, ax=axTT[0], location='right', orientation='vertical') # colorbar
                cb_Te.set_label(r'$T(x,y)$', labelpad=10)
                #cb_T.ax.xaxis.set_label_position('top')  # Move label to the top
                cse = axTT[0].contour(X_high_res, Y_high_res, T_r_early, levels=levels, linewidths=0.9, colors="k") # plot of different levels on the colormap
            
                im_Tl = axTT[1].pcolormesh(X_low_res, Y_low_res, T_r_frame, vmin=0., vmax=1., cmap='plasma', alpha=0.7, edgecolors='none', linewidth=0, shading='gouraud') # plot of colormap of T
                cb_Tl = plt.colorbar(im_Tl, ax=axTT[1], location='right', orientation='vertical') # colorbar
                cb_Tl.set_label(r'$T(x,y)$', labelpad=10)
                csl = axTT[1].contour(X_high_res, Y_high_res, T_r, levels=levels, linewidths=0.8, colors="k") # plot of different levels on the colormap
                
                [axi.set_xlabel("$x$") for axi in axTT]
                [axi.set_ylabel("$y$") for axi in axTT]
                
                ylab_xpos_l_b = axTT[0].yaxis.get_label().get_position()[0]  # horizontal position of y-label
                ylab_xpos_r_b = axTT[0].yaxis.get_label().get_position()[1]  # horizontal position of y-label
                figTT.text(ylab_xpos_l_b + 0.075, 0.98, "($a$)", verticalalignment='top', horizontalalignment='right')
                figTT.text(ylab_xpos_r_b + 0.025, 0.98, "($b$)", verticalalignment='top', horizontalalignment='right')
                
                Lxlim = 10
                [axi.set_xlim(0, Lxlim) for axi in axTT]
                [axi.set_ylim(0, Ly) for axi in axTT]
                [axi.set_aspect('auto') for axi in axTT]
                
                figTT.savefig(out_dir + '/Tlevels_rnd.pdf', dpi=400, bbox_inches="tight")
        plt.show()
        plt.close()
        
    exit(0)
    # Analyze time evolution
    
    for it in it_:
        t = t_[it] # time at step it
        print(f"it={it} t={t}")

        # Load data
        dset_T = dsets_T[t] # T-dictionary at time t
        with h5py.File(dset_T[0], "r") as h5f:
            T_[:] = h5f[dset_T[1]][:, 0] # Takes values of T from the T-dictionary
        T_sorted = T_[sort_indices]
        T_r_ = T_sorted.reshape((ny, nx))

        with h5py.File(dsets_p[t][0], "r") as h5f:
            p_[:] = h5f[dsets_p[t][1]][:, 0] # Takes values of p from the p-dictionary
        p_sorted = p_[sort_indices]
        p_r_ = p_sorted.reshape((ny, nx))
        
        grad_py, grad_px = np.gradient(p_r_, y_sort, x_sort) # gradient of p
        ux_r_ = -beta**-T_r_ * grad_px # x-component of velocity field (u = beta^(-T) \nabla p)
        uy_r_ = -beta**-T_r_ * grad_py # y-component of velocity field (u = beta^(-T) \nabla p)
        
        # Interpolate the data to higher resolution using RectBivariateSpline
        f_T = RectBivariateSpline(y_sort, x_sort, T_r_)
        T_r = f_T(Y_high_res[:, 0], X_high_res[0, :])
        f_ux = RectBivariateSpline(y_sort, x_sort, ux_r_)
        ux_r = f_ux(Y_high_res[:, 0], X_high_res[0, :])
        f_uy = RectBivariateSpline(y_sort, x_sort, uy_r_)
        uy_r = f_uy(Y_high_res[:, 0], X_high_res[0, :])
        
        cs = plt.contour(X_high_res, Y_high_res, T_r, levels=levels, colors="k") # plot of different levels on the colormap
        paths = [] # curves formed by each level
        for level, path in zip(cs.levels, cs.get_paths()):
            if len(path.vertices): # if the path has non-null lenght
                paths.append((level, path.vertices))
        paths = dict(paths)

        for level, verts in paths.items():
            xmax[level][it] = verts[:, 0].max() # max x-position of a level
            xmin[level][it] = verts[:, 0].min() # min x-position of a level
            
        u_r = np.sqrt(ux_r**2 + uy_r**2) # |u| : absolute value of velocity field.
        umax[it] = u_r.max() # max of |u| at step it
        
        #Tprime_r = T_r - np.exp(-xi*X_high_res[:, :])
        #uprime_r = ux_r - 1.
        
        for i in range(0,n_samples):
            nxi = nx_high_res*(i+1)//int(Lx)
            nymax = np.argmax(T_r[:,nxi]) if rnd else ny//4
            nymin = np.argmin(T_r[:,nxi]) if rnd else 3*ny//4
            Txt[i][it] = T_r[nymax,nxi] - T_r[nymin,nxi]
            uxt[i][it] = ux_r[nymax,nxi] - ux_r[nymin,nxi]
    
    t_i = 5.
    t_f = 15.
    #t_i = 2 if rnd else 2.5 # initial time of growth rate measurement
    #t_f = 6.5 if rnd else 13.5 # final time of growth rate measurement
    
    n_i = int(n_steps * t_i/t_end)
    n_f = int(n_steps * t_f/t_end)
    
    gamma_ = np.zeros(n_samples)
    gamma_std_ = np.zeros(n_samples)
    for i in range(0,n_samples):
        model, cov = np.polyfit(t_[n_i:n_f], np.log(Txt[i][n_i:n_f]), 1, cov=True)
        gamma_[i] = model[0]
        gamma_std_[i] = np.sqrt(cov[0, 0])
        print("x = ", i+1, ", gamma_ = ", gamma_[i], ", gamma_std_ = ", gamma_std_[i])
    gamma_avg = np.average(gamma_, weights=1/gamma_std_**2)
    print(f"gamma_avg = {gamma_avg}")
    
    # Plot Tspan(x, t) and uxspan(x, t) vs t for different x
    figspan, axspan = plt.subplots(1, 2, figsize=(15., 5))
    
    n_evol = n_steps if rnd else n_steps//2
    
    for i in range(0,n_samples):
        color = cmap_space(1. - (i+1) / n_samples)
        axspan[0].plot(t_[0:n_evol], Txt[i][0:n_evol], label=fr'$x = {i:1f}$', color=color)
        axspan[1].plot(t_[0:n_evol], uxt[i][0:n_evol], label=fr'$x = {i:1f}$', color=color)
    aT = 8*1e-5 if rnd else 1.5*1e-4
    aux = 5*1e-4 if rnd else 1e-3
    axspan[0].plot(t_[n_i:n_f], aT*np.exp(gamma_avg*t_[n_i:n_f]), color='black', linestyle='dashed')
    axspan[1].plot(t_[n_i:n_f], aux*np.exp(gamma_avg*t_[n_i:n_f]), color='black', linestyle='dashed')
    text_idx = (n_i + n_f)//2
    text_fit = r"$\propto e^{\gamma_* t}$" if rnd else r"$\propto e^{\gamma t}$"
    axspan[0].text(t_[text_idx], aT*np.exp(gamma_avg*t_[text_idx]), text_fit, va="bottom", ha="right")
    axspan[1].text(t_[text_idx], aux*np.exp(gamma_avg*t_[text_idx]), text_fit, va="bottom", ha="right")
    
    [axi.semilogy() for axi in axspan]
    [axi.set_xlabel(r"$t$") for axi in axspan]
    axspan[0].set_ylabel(r"$T^{\rm span}(x,t)$")
    axspan[1].set_ylabel(r"$u_{x}^{\rm span}(x,t)$")
    
    ylab_xpos_l_b = axspan[0].yaxis.get_label().get_position()[0]  # horizontal position of y-label
    ylab_xpos_r_b = axspan[0].yaxis.get_label().get_position()[1]  # horizontal position of y-label
    figspan.text(ylab_xpos_l_b + 0.075, 0.98, "($a$)", verticalalignment='top', horizontalalignment='right')
    figspan.text(ylab_xpos_r_b + 0.025, 0.98, "($b$)", verticalalignment='top', horizontalalignment='right')
    
    if (rnd == False):
        axspan[0].set_ylim(np.sqrt(10)*1e-7, 2)
        axspan[1].set_ylim(np.sqrt(10)*1e-6, 20)
        
    imgnamespan = '/Tspan_and_uxspan_rnd.pdf' if rnd else '/Tspan_and_uxspan_sin.pdf'
    figspan.savefig(out_dir + imgnamespan, dpi=600, bbox_inches="tight")
        
    #plt.show()
    
    cmap = plt.cm.viridis
    #figf, axf = plt.subplots(1, 2, figsize=(15, 5))
    figf, axf = plt.subplots(1, 1)
    figf.subplots_adjust(wspace=0.3)
    
    figff, axff = plt.subplots(1, 1)
    
    gamma2_ = np.zeros(len(levels[1:-1])) # growth rate for each level
    tstat_ = np.zeros(len(levels[1:-1])) # time to reach the stationary state for each level
    i = 0
    
    for level in levels[1:-1]:
    
        # Plot xmax, xmin, xspan vs t
        color = cmap(level)
        xbase = -math.log(level) / xi
        #axf[0].plot(t_[:n_steps], xmax[level][:n_steps] - xmax[level][0], color=color) # plot xmax vs t for each level
        #axf[1].plot(t_[:n_steps], np.abs(xmin[level][:n_steps] - xmin[level][0]), color=color) # plot xmin vs t for each level
        axf.plot(t_[1:n_steps], (xmax[level][1:n_steps] - xmin[level][1:n_steps]), label=f"$T={level:1.2f}$", color=color) # plot span vs t for each level
        
        # Save xspan vs t in file .txt
        xspan_data =  np.column_stack(( t_[1:n_steps], xmax[level][1:n_steps] - xmin[level][1:n_steps] ))
        np.savetxt(out_dir + f'/xspan_T={level:1.2f}.txt', xspan_data, fmt='%1.9f')
        
        # Plot xspan vs t at growing stage and find gamma and tstat
        n_i = int(n_steps * t_i/t_end)
        n_f = int(n_steps * t_f/t_end)
        axff.plot(t_[n_i:n_f], np.log(xmax[level][n_i:n_f] - xmin[level][n_i:n_f]), label=f"$T={level:1.2f}$", color=color)
        model = LinearRegression().fit(t_[n_i:n_f].reshape((-1, 1)), np.log(xmax[level][n_i:n_f] - xmin[level][n_i:n_f]))
        xspan_sat = xmax[level][n_steps-1] - xmin[level][n_steps-1] # stationary value for xspan
        gamma2_[i] = model.coef_[0]
        tstat_[i] = (np.log(xspan_sat) - model.intercept_)/ model.coef_[0]
        i += 1
    gamma = np.mean(gamma2_)
    print('gamma = ', gamma)
    axf.plot(t_[n_i:n_f], 1e-3*np.exp(gamma*t_[n_i:n_f]), color='black', linestyle='dashed') # plot fitting line
    axf.text(t_[n_steps//2], 1e-3*np.exp(gamma*t_[n_steps//2]), r"$\propto e^{\gamma_{\max} t}$", va="bottom", ha="right")
    
    axf.set_xlabel("$t$")
    axf.set_ylabel("$x_{max}-x_{min}$")
    axf.semilogy()
    axf.tick_params(axis='both', which='major')
    figf.tight_layout()
    figf.savefig(out_dir + '/fingergrowth.pdf', dpi=300)
    
    #axff.legend(fontsize='small')
    axff.set_xlabel("$t$")
    axff.set_ylabel("$\log(x_{max}-x_{min})$")
    figff.savefig(out_dir + '/xmax_growing.pdf', dpi=300)
    
    gamma_data = np.column_stack(( levels[1:-1], gamma2_ ))
    np.savetxt(out_dir + f'/growth_rates.txt', gamma_data, fmt='%1.9f')
    tstat_data = np.column_stack(( levels[1:-1], tstat_ ))
    np.savetxt(out_dir + f'/tstat.txt', tstat_data, fmt='%1.9f')
    
    plt.show()
