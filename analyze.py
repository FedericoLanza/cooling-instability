import argparse
import h5py
import math
import matplotlib.pyplot as plt
import meshio
import numpy as np
import os
from matplotlib.animation import FuncAnimation
from matplotlib import tri
from scipy.interpolate import RectBivariateSpline
from sklearn.linear_model import LinearRegression
from utils import parse_xdmf

# ciao

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('Pe', type=float, help='Value for Peclet number')
    parser.add_argument('Gamma', type=float, help='Value for heat transfer ratio')
    parser.add_argument('beta', type=float, help='Value for viscosity ratio')
    parser.add_argument('ueps', type=float, help='Value for amplitude of the perturbation')
    parser.add_argument('Ly', type=float, help='Value for wavelength')
    parser.add_argument('Lx', type=float, help='Value for system size')
    parser.add_argument('dt', type=float, help='Value for time interval')
    #parser.add_argument('ny', type=float, help='Value for tile density along y')
    #parser.add_argument('rtol', type=float, help='Value for error function')
    parser.add_argument('--rnd',action='store_true', help='Flag for random velocity at inlet')
    parser.add_argument('--holdpert',action='store_true', help='Flag for maintaining the perturbation at all times')
    parser.add_argument('--constDeltaP',action='store_true', help='Flag for imposing constant pressure, instead of constant flow rate, at the inlet boundary')
    # parser.add_argument("--show", action="store_true", help="Show") # optional argument: typing --show enables the "show" feature
    parser.add_argument("--snap", action="store_true", help="Snap") # optional argument: typing --snap enables the "snap" feature
    return parser.parse_args()

if __name__ == "__main__":

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
    nx_high_res = 400
    x_high_res = np.linspace(x_min, x_max, nx_high_res)
    X_high_res, Y_high_res = np.meshgrid(x_high_res, y_sort)
    
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
    Tkx = np.zeros_like(t_)
    Tk_max = np.zeros_like(t_)
    
    # Analyze final state
    
    cmap = plt.cm.viridis
    
    if True:
        #beta = 0.001 # viscosity ratio

        t = t_[it_[-1]] # final time
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
        f_ux = RectBivariateSpline(y_sort, x_sort, ux_r_)
        ux_r = f_ux(Y_high_res[:, 0], X_high_res[0, :])
        
        T_max = T_r.max(axis=0) # max of T along y at fixed t
        ux_max = ux_r.max(axis=0) # max of u_x along y at fixed t

        # Plot T and u_x along y for fixed x
        
        fig1, ax1 = plt.subplots(1, 2, figsize=(12, 4))
        for i in range(nx_high_res)[::25]:
            color = cmap(i / (nx_high_res - 1))  # Adjust the color according to column index
            ax1[0].plot(y_sort, T_r[:, i], label=f"$x={x_high_res[i]:1.2f}$", color=color) # plot T(y) for different x
            ax1[1].plot(y_sort, ux_r[:, i], color=color) # plot u_x(y) for different x
        #for i in range(nx)[::20]:
            #gauss = [ np.exp( -( y_ - Lx/4)**2/(0.0017*x[i]) ) for y_ in y_sort]
            #ax1[1].plot(y_sort, gauss, linestyle='dotted', color='black')
        ax1[0].set_ylabel("$T$")
        ax1[1].set_ylabel("$u_x$")
        #ax1[0].legend(fontsize='small')
        [axi.set_xlabel("$y$") for axi in ax1]
        fig1.savefig(out_dir + '/fx.png', dpi=300)
        
        fig1a, ax1a = plt.subplots(1, 2, figsize=(12, 4))
        for i in range(nx_high_res)[::25]:
            color = cmap(i / (nx_high_res - 1))  # Adjust the color according to column index
            ax1a[0].plot(np.log(y_sort - Ly/4), np.log(1-(T_r[:, i]/T_max[i])), label=f"$x={x_high_res[i]:1.2f}$", color=color) # plot T(y) for different x
            ax1a[1].plot(np.log(y_sort - Ly/4), np.log(1-(ux_r[:, i]/ux_max[i])), color=color) # plot u_x(y) for different x
        for i in range(nx_high_res)[::400]:
            # color = cmap(i / (nx - 1))  # Adjust the color according to column index
            sigma = np.sqrt(0.0037*x[i])
            gaussT = [ 2*(y_) - (-5 + 0.0057*x[i]) for y_ in np.linspace(-5,0,40)]
            gaussux = [ 2*(y_) - (-5 + 0.0007*x[i]) for y_ in np.linspace(-5,0,40)]
            #cauchyT = [ 1/((1 + (y_/sigma)**2 )) for y_ in y_sort]
            #cauchyux = [ 1/((1 + (y_/sigma)**2 )) for y_ in y_sort]
            ax1a[0].plot(np.linspace(-5,0,40), gaussT, linestyle='dotted', color='black')
            ax1a[1].plot(np.linspace(-5,0,40), gaussux, linestyle='dotted', color='black')
        ax1a[0].set_ylabel("$\log(1 - T(y')/T_m)$")
        ax1a[1].set_ylabel("$\log(1 - u_x(y')/u_m)$")
        #ax1a[0].semilogy()
        #ax1a[1].semilogy()
        #ax1a[0].loglog()
        #ax1a[1].loglog()
        #ax1a[0].legend(fontsize='small')
        [axi.set_xlabel("$\log(y')$") for axi in ax1a]
        fig1a.savefig(out_dir + '/fx_loglog.png', dpi=300)
        
        # Plot T and u_x along x for fixed y
        
        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
        ax2[0].plot(x_high_res, [np.exp(-x*xi) for x in x_high_res], color='black', linestyle='dashed') # plot the base state T_0(x)
        for i in range(ny)[::25]:
            color = cmap(i / (ny - 1))  # Adjust the color according to column index
            ax2[0].plot(x_high_res, T_r[i, :], label=f"$y={y_sort[i]:1.2f}$", color=color) # plot T(x) for different y
            ax2[1].plot(x_high_res, ux_r[i, :], color=color) # plot u_x(x) for different y
        ax2[0].set_ylabel("$T$")
        ax2[1].set_ylabel("$u_x$")
        ax2[0].legend()
        [axi.set_xlabel("$x$") for axi in ax2]
        fig2.savefig(out_dir + '/fy.png', dpi=300)
        
        # Plot max of T and u_x along x for fixed y
        
        fig3, ax3 = plt.subplots(1, 2, figsize=(12, 4))
        ax3[0].plot(x_high_res, [np.exp(-x*xi) for x in x_high_res], color='black', linestyle='dashed')
        ax3[0].plot(x_high_res, T_max)
        ax3[1].plot(x_high_res, ux_max)
        ax3[0].set_ylabel("$T_{max}(x)$")
        ax3[1].set_ylabel("$u_{x,max}(x)$")
        [axi.set_xlabel("$x$") for axi in ax3]
        fig3.savefig(out_dir + '/maxfy.png', dpi=300)
        
        if (args.snap):
        
            # Plot colormaps of T at final state with levels
            figT, axT = plt.subplots(1, 1, figsize=(15, 3))
            
            im_T = axT.pcolormesh(X_high_res, Y_high_res, T_r, vmin=0., vmax=1.) # plot of colormap of T
            cb_T = plt.colorbar(im_T, ax=axT, shrink=0.5) # colorbar
            cs = axT.contour(X_high_res, Y_high_res, T_r, levels=levels, colors="k") # plot of different levels on the colormap
            axT.set_aspect("equal")
            axT.set_xlabel("$x$")
            axT.set_ylabel("$y$")
            figT.suptitle(f"Final state ($t = {t:1.2f}$)")
            figT.savefig(out_dir + '/Tlevelmap.png', dpi=300)
            plt.show()
            plt.close()
        
            # Calculate uy
            uy_r_ = -beta**-T_r_ * grad_py
            f_uy = RectBivariateSpline(y_sort, x_sort, uy_r_)
            uy_r = f_uy(Y_high_res[:, 0], X_high_res[0, :])
            
            # Plot colormaps of ux and uy at final state
            figu, axu = plt.subplots(1, 2, figsize=(9, 3))
            
            im_ux = axu[0].pcolormesh(X_high_res, Y_high_res, ux_r) # plot of colormap of ux
            cb_ux = plt.colorbar(im_ux, ax=axu[0]) # colorbar
            axu[0].set_title("$u_x$")
            [axi.set_xlabel("$x$") for axi in axu]
            
            im_uy = axu[1].pcolormesh(X_high_res, Y_high_res, uy_r) # plot of colormap of ux
            cb_uy = plt.colorbar(im_uy, ax=axu[1]) # colorbar
            axu[1].set_title("$u_y$")
            [axi.set_ylabel("$y$") for axi in axu]
            
            figu.suptitle(f"Final state ($t = {t:1.2f}$)")
            figu.savefig(out_dir + f'/umap.jpg', dpi=300)
        
        plt.show()
        plt.close()
    
    # Analyze time evolution
    
    #figTiki, axTiki = plt.subplots(1, 1)
    #axTiki.set_xlabel("$t$")
    #axTiki.set_ylabel("$ok$")
    
    figTk, axTk = plt.subplots(1, 2, figsize=(15, 5))
    axTk[0].set_xlabel(r"$x$")
    axTk[0].set_ylabel(r"$T_k(x)$")
    axTk[1].set_xlabel(r"$x$")
    axTk[1].set_ylabel(r"$T_k(x)/T_k^{\rm max}$")
    
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
        
        Tprime_r = T_r - np.exp(- xi*X_high_res[:, :])
        Tprime_max[it] = Tprime_r.max() # max of T' at step it
        
        Tk_max[it] = np.max(Tprime_r[ny//4]) # max of Tk at step it
        
        Tkx[it] = Tprime_r[ny//4][nx_high_res*5//int(Lx)] #T_k(x=5)
        
        #if (it <= n_steps//5 and it % 20 == 0 ):
        #    color = cmap(it / (n_steps//5 - 1))
        #    axTk[0].plot(x_high_res, Tprime_r[ny//4][:], label=f"$t={t:1.2f}$", color=color) # plot u_max vs t
        #    axTk[1].plot(x_high_res, Tprime_r[ny//4][:]/Tprime_r.max(), label=f"$t={t:1.2f}$", color=color) # plot u_max vs t
    
    t_i = 2.5 # initial time of growth rate measurement
    t_f = 6 # final time of growth rate measurement
    
    n_i = int(n_steps * t_i/t_end)
    n_f = int(n_steps * t_f/t_end)
    model_max, cov_max = np.polyfit( t_[n_i:n_f], np.log(Tk_max[n_i:n_f]), 1, cov=True )
    model_5, cov_5 = np.polyfit( t_[n_i:n_f], np.log(Tkx[n_i:n_f]), 1, cov=True )
    gamma_max = model_max[0]
    gamma_max_std = cov_max[0, 0]
    gamma_5 = model_5[0]
    gamma_5_std = cov_5[0, 0]
    
    print(f"gamma_max = {gamma_max}")
    print(f"gamma_max_std = {gamma_max_std}")
    print(f"gamma_5 = {gamma_5}")
    print(f"gamma_5_std = {gamma_5_std}")
        
    axTk[0].legend(loc='center right')
    axTk[0].set_xlim(0,20)
    axTk[1].set_xlim(0,20)
    figTk.savefig(out_dir + f'/Tk.jpg', dpi=300)
    
    # Plot Tprime_max vs t
    figTpmax, axTpmax = plt.subplots(1, 1)
    axTpmax.plot(t_[1:n_steps//5], Tprime_max[1:n_steps//5])
    axTpmax.set_xlabel(r"$t$")
    axTpmax.set_ylabel(r"$T'_{\rm max}$")
    axTpmax.semilogy()
    figTpmax.savefig(out_dir + f'/Tpmax.jpg', dpi=300)
    
    # Plot T_k^max vs t
    figTkmax, axTkmax = plt.subplots(1, 1)
    axTkmax.plot(t_[1:n_steps//5], Tk_max[1:n_steps//5])
    axTkmax.set_xlabel(r"$t$")
    axTkmax.set_ylabel(r"$T_{k}^{\rm max}$")
    axTkmax.semilogy()
    figTkmax.savefig(out_dir + f'/Tk_max.jpg', dpi=300)
    
    # Plot T_k(x=5) vs t
    figTkx, axTkx = plt.subplots(1, 1)
    axTkx.plot(t_[1:n_steps//5], Tkx[1:n_steps//5])
    axTkx.set_xlabel(r"$t$")
    axTkx.set_ylabel(r"$T_{k}(x=5)$")
    axTkx.semilogy()
    figTkx.savefig(out_dir + f'/Tkx5.jpg', dpi=300)
    
    # Plot umax vs t
    figumax, axumax = plt.subplots(1, 1)
    axumax.plot(t_[1:n_steps], umax[1:n_steps] - 1.)
    axumax.set_xlabel(r"$t$")
    axumax.set_ylabel(r"$u'_{max}$")
    axumax.semilogy()
    figumax.savefig(out_dir + f'/umax.jpg', dpi=300)
    
    # Save umax vs t in file .txt
    umax_data =  np.column_stack(( t_[1:], umax[1:] ))
    np.savetxt(out_dir + f'/umax.txt', umax_data, fmt='%1.9f')
    
    #plt.show()
    
    cmap = plt.cm.viridis
    figf, axf = plt.subplots(1, 3, figsize=(30, 5))
    #figf, axf = plt.subplots(1, 1)
    figf.subplots_adjust(wspace=0.3)
    figff, axff = plt.subplots(1, 1)
    
    gamma_ = np.zeros(len(levels[1:-1])) # growth rate for each level
    tstat_ = np.zeros(len(levels[1:-1])) # time to reach the stationary state for each level
    i = 0
    
    for level in levels[1:-1]:
    
        # Plot xmax, xmin, xspan vs t
        color = cmap(level)
        xbase = -math.log(level) / xi
        axf[0].plot(t_[:n_steps], xmax[level][:n_steps] - xmax[level][0], color=color) # plot xmax vs t for each level
        axf[1].plot(t_[:n_steps], np.abs(xmin[level][:n_steps] - xmin[level][0]), color=color) # plot xmin vs t for each level
        axf[2].plot(t_[1:n_steps], (xmax[level][1:n_steps] - xmin[level][1:n_steps]), label=f"$T={level:1.2f}$", color=color) # plot span vs t for each level
        
        # Save xspan vs t in file .txt
        xspan_data =  np.column_stack(( t_[1:n_steps], xmax[level][1:n_steps] - xmin[level][1:n_steps] ))
        np.savetxt(out_dir + f'/xspan_T={level:1.2f}.txt', xspan_data, fmt='%1.9f')
        
        # Plot xspan vs t at growing stage and find gamma and tstat
        n_i = int(n_steps * t_i/t_end)
        n_f = int(n_steps * t_f/t_end)
        axff.plot(t_[n_i:n_f], np.log(xmax[level][n_i:n_f] - xmin[level][n_i:n_f]), label=f"$T={level:1.2f}$", color=color)
        model = LinearRegression().fit(t_[n_i:n_f].reshape((-1, 1)), np.log(xmax[level][n_i:n_f] - xmin[level][n_i:n_f]))
        xspan_sat = xmax[level][n_steps-1] - xmin[level][n_steps-1] # stationary value for xspan
        gamma_[i] = model.coef_[0]
        tstat_[i] = (np.log(xspan_sat) - model.intercept_)/ model.coef_[0]
        i += 1
    
    axf[2].legend(fontsize='large')
    [axi.set_xlabel("$t$", fontsize=16) for axi in axf]
    #axf.set_xlabel("$t$", fontsize=16)
    axf[0].set_ylabel("$x_{max}(t) - x_{max}(t=0)$", fontsize=16)
    axf[1].set_ylabel("$x_{min}(t) -  x_{min}(t=0)$", fontsize=16)
    #axf.set_ylabel("$x_{max}-x_{min}$", fontsize=16)
    axf[2].set_ylabel("$x_{max}-x_{min}$", fontsize=16)
    axf[0].semilogy()
    axf[1].semilogy()
    #axf.semilogy()
    axf[2].semilogy()
    axf[2].tick_params(axis='both', which='major', labelsize=14)
    figf.savefig(out_dir + '/fingergrowth.png', dpi=300)
    #axf.tick_params(labelsize=14)
    axf[2].tick_params(labelsize=14)
    
    #axff.legend(fontsize='small')
    axff.set_xlabel("$t$")
    axff.set_ylabel("$\log(x_{max}-x_{min})$")
    figff.savefig(out_dir + '/xmax_growing.png', dpi=300)
    
    gamma_data = np.column_stack(( levels[1:-1], gamma_ ))
    np.savetxt(out_dir + f'/growth_rates.txt', gamma_data, fmt='%1.9f')
    tstat_data = np.column_stack(( levels[1:-1], tstat_ ))
    np.savetxt(out_dir + f'/tstat.txt', tstat_data, fmt='%1.9f')
    
    #plt.show()
