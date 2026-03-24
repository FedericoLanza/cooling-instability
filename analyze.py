import argparse
import h5py
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import meshio
import numpy as np
import os
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib import tri
from scipy.interpolate import RectBivariateSpline
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from utils import parse_xdmf

# ciao

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--Pe', default=1000, type=float, help='Value for Peclet number')
    parser.add_argument('--Gamma', default=1e-5, type=float, help='Value for heat transfer ratio')
    parser.add_argument('--beta', default=1e-3, type=float, help='Value for viscosity ratio')
    parser.add_argument('--ueps', default=0.001, type=float, help='Value for amplitude of the perturbation')
    parser.add_argument('--Ly', default=140000, type=float, help='Value for wavelength')
    parser.add_argument('--Lx', default=800000, type=float, help='Value for system size')
    parser.add_argument('--ny', default=300, type=float, help='Value for wavelength')
    parser.add_argument('--nx', default=800, type=float, help='Value for system size')
    parser.add_argument('--dt', default=200, type=float, help='Value for time interval')
    #parser.add_argument('rtol', type=float, help='Value for error function')
    parser.add_argument('--rnd', action='store_true', help='Flag for random velocity at inlet')
    parser.add_argument('--holdpert', action='store_true', help='Flag for maintaining the perturbation at all times')
    parser.add_argument('--Tpert', action='store_true', help='Flag for perturbing T, instead of u, at the inlet')
    parser.add_argument('--twoperiods', action='store_true', help='Flag for performing simulations with 2 wavelenghts per spatial period')
    parser.add_argument('--tp', action='store_true', help='Flag for imposing constant pressure, instead of constant flow rate, at the inlet boundary')
    parser.add_argument('--final', action='store_true', help='Flag for analyzing the final state')
    parser.add_argument('--print_colormaps', action='store_true', help='Flag for printing 2d colormaps')
    parser.add_argument('--print_profiles', action='store_true', help='Flag for printing 1d profiles')
    parser.add_argument('--growth', action='store_true', help='Flag for studying the growth')
    parser.add_argument('--latex', action='store_true', help='Flag for plotting in LaTeX style')
    # parser.add_argument("--show", action="store_true", help="Show") # optional argument: typing --show enables the "show" feature
    return parser.parse_args()

if __name__ == "__main__":

    cmap_space = sns.color_palette("mako", as_cmap=True)
    cmap_time = sns.color_palette("rocket", as_cmap=True)
    
    # Parse the command-line arguments
    args = parse_args() # object containing the values of the parsed argument
    
    Lx = args.Lx # x-lenght of domain (system size)
    Ly = args.Ly # y-lenght of domain (wavelength)
    nx = args.nx # x-lenght of domain (system size)
    ny = args.ny # y-lenght of domain (wavelength)
    dt = args.dt #time step
    
    # global parameters
    Pe = args.Pe # Peclet number
    Gamma = args.Gamma # Heat transfer ratio
    beta = args.beta # Viscosity ratio ( nu(T) = beta^(-T) )
    
    # inlet parameters
    ueps = args.ueps # amplitude of the perturbation
    u0 = 1.0 # base inlet velocity
    k = 2*np.pi/Ly
    # gamma = 0.52
    ym = Ly/4
    
    # base state parameters
    kappa = 1./Pe # thermal diffusivity
    kappa_eff = kappa + 2*Pe*u0*u0/105 # effective constant diffusion for the base state
    xi = ( - u0 + math.sqrt(u0*u0 + 4*kappa_eff*Gamma)) / (2*kappa_eff) # decay constant for the base state
    #Lambda = ( - u0 + math.sqrt(u0*u0 + 4*kappa_eff*(Gamma + kappa*k*k + gamma))) / (2*kappa_eff) # decay constant for the base state
    
    # resolution parameters
    #rtol = args.rtol
    
    # flags
    rnd = args.rnd
    Tpert = args.Tpert
    holdpert = args.holdpert
    twoperiods = args.twoperiods
    tp = args.tp
    final = args.final
    print_colormaps = args.print_colormaps
    print_profiles = args.print_profiles
    latex = args.latex
    growth = args.growth
    
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
    dt_str = f"dt_{dt:.10g}"
    ny_str = f"ny_{ny:.10g}"
    nx_str = f"nx_{nx:.10g}"
    #rtol_str = f"rtol_{rtol:.10g}"
    rnd_str = f"rnd_{rnd}"
    Tpert_str = f"Tpert_{Tpert}"
    holdpert_str = f"holdpert_{holdpert}"
    
    out_dir = "results/"
    if tp:
        out_dir += "constDeltaP_"
    out_dir += "_".join([Pe_str, Gamma_str, beta_str, Ly_str, Lx_str, dt_str, ny_str, nx_str, rnd_str, Tpert_str, holdpert_str]) # directoty for output
    #out_dir += "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str, holdpert_str]) # directoty for output
    if twoperiods:
        out_dir += "_twoperiods"
    out_dir += "/"
    
    out_dir_2 = "results/"
    if tp == False:
        out_dir_2 += "output"
    else:
        out_dir_2 += "outppt"
        
    out_dir_2 = out_dir_2 + "_tfull_mix/"
    if os.path.exists(out_dir_2) == False:
        os.mkdir(out_dir_2)
        
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
    Nx = len(x_sort) - 1
    Ny = len(y_sort) - 1
    print ("Nx = ", Nx, ", Ny = ",Ny)
    X, Y = np.meshgrid(x_sort, y_sort)
    
    x_min = min(nodes[:, 0])
    x_max = max(nodes[:, 0])
    y_min = min(nodes[:, 1])
    y_max = max(nodes[:, 1])
    
    Ny_low_res = Ny//2
    x_high_res = x_sort
    y_low_res = np.linspace(y_min, y_max, Ny_low_res)
    
    X_high_res, Y_high_res = np.meshgrid(x_sort, y_sort)
    X_low_res, Y_low_res = np.meshgrid(x_sort, y_low_res)
    
    # Sort indices of nodes array
    sort_indices = np.lexsort((nodes[:, 0], nodes[:, 1]))

    t_ = np.array(sorted(dsets_T.keys())) # time array
    it_ = list(range(len(t_))) # iteration steps

    T_ = np.zeros_like(nodes[:, 0]) # temperature field
    p_ = np.zeros_like(T_) # pressure field
    # u_ = np.zeros((len(elems), 2)) # velocity field

    n_steps = len(it_)
    print("n_steps = ", n_steps)
    dump_intv = 10 # update manually
    dt_save = dt*dump_intv
    t_end = n_steps*dt_save
    print("t_end = ", t_end)
    levels = np.linspace(0, 1, 11) # levels of T
    xmax = dict([(level, np.zeros_like(t_)) for level in levels]) # max x-position of a level for all time steps, for all levels
    xmin = dict([(level, np.zeros_like(t_)) for level in levels]) # min x-position of a level for all time steps, for all levels
    
    n_samples = 10
    Txt = np.zeros((n_samples, n_steps))
    uxt = np.zeros((n_samples, n_steps))
    Ttx = np.zeros((n_samples, Nx))
    utx = np.zeros((n_samples, Nx))
    
    
    if twoperiods:
        Txt_2 = np.zeros((n_samples, n_steps))
        uxt_2 = np.zeros((n_samples, n_steps))
    
    # Analyze final state
    
    cmap = plt.cm.viridis
    
    if final:
        #beta = 0.001 # viscosity ratio

        t = t_[it_[14*n_steps//16]] # final time
        #t = t_[it_[n_steps-1]] # final time
        dset_T = dsets_T[t] # T-dictionary at final time
        with h5py.File(dset_T[0], "r") as h5f:
            T_[:] = h5f[dset_T[1]][:, 0] # takes values of T from the T-dictionary
        T_sorted = T_[sort_indices]
        T_r_ = T_sorted.reshape((Ny + 1, Nx + 1))
        
        with h5py.File(dsets_p[t][0], "r") as h5f:
            p_[:] = h5f[dsets_p[t][1]][:, 0] # takes values of p from the p-dictionary
        p_sorted = p_[sort_indices]
        p_r_ = p_sorted.reshape((Ny + 1, Nx + 1))
        
        grad_py, grad_px = np.gradient(p_r_, y_sort, x_sort) # gradient of p
        ux_r_ = -beta**-T_r_ * grad_px # x-component of velocity field (u = beta^(-T) \nabla p)
        
        # Interpolate the data to higher resolution using RectBivariateSpline
        f_T = RectBivariateSpline(y_sort, x_sort, T_r_)
        T_r = f_T(Y_high_res[:, 0], X_high_res[0, :])
        T_r_frame = f_T(Y_low_res[:, 0], X_low_res[0, :])
        
        if rnd:
            T_2 = np.zeros_like(nodes[:, 0]) # temperature field
            
            t2 = t_[it_[11*n_steps//16]] # time selected for the first frame
            dset_T2 = dsets_T[t2] # T-dictionary at final time
            with h5py.File(dset_T2[0], "r") as h5f:
                T_2[:] = h5f[dset_T2[1]][:, 0] # takes values of T from the T-dictionary
            T_sorted_2 = T_2[sort_indices]
            T_r_2 = T_sorted_2.reshape((Ny + 1, Nx + 1))
            
            f_T2 = RectBivariateSpline(y_sort, x_sort, T_r_2)
            T_r_early_frame = f_T2(Y_low_res[:, 0], X_low_res[0, :])
        
        f_ux = RectBivariateSpline(y_sort, x_sort, ux_r_)
        ux_r = f_ux(Y_high_res[:, 0], X_high_res[0, :])
        
        Tprime_r = T_r - np.exp(-xi*X_high_res[:, :])
        uxprime_r = ux_r - 1.
        
        #T_max = T_r.max(axis=0) # max of T along y at fixed t
        #ux_max = ux_r.max(axis=0) # max of u_x along y at fixed t
        #Tprime_max = Tprime_r.max(axis=0) # max of T along y at fixed t
        #uxprime_max = uxprime_r.max(axis=0) # max of u_x along y at fixed t
        
        print("y_sort = ", y_sort)
        print("len(y_sort) = ", len(y_sort))
        y_max_where = np.where(abs(y_sort - ym) < 1e-3)[0][0]
        #y_max_where = 1750
        print("y_max_where = ", y_max_where)
        
        T_max = T_r[y_max_where,:]
        ux_max = ux_r[y_max_where,:]
        Tprime_max = Tprime_r[y_max_where,:]
        uxprime_max = uxprime_r[y_max_where,:]
        
        if print_profiles:
        
            def poly1sqrt2(x, a, b, c):
                return a / np.sqrt(b - c*x)
                
            def poly1(x, a, b,):
                return a*x + b
            
            #y_fit = y_sort[y_max_where-5:y_max_where+5]
        
            # Define sone intervals of action
            Tprime_maxmax = Tprime_max.max()
            Tprime_maxmax_where = Tprime_max.argmax()
            x_Tprime_maxmax = x_high_res[Tprime_maxmax_where]
            
            uxprime_maxmax = uxprime_max.max()
            uxprime_maxmax_where = uxprime_max.argmax()
            x_uxprime_maxmax = x_high_res[uxprime_maxmax_where]
            ux_maxmax = uxprime_maxmax + 1.
            x_ux_maxmax = x_uxprime_maxmax
            
            mask_Tbump = (x_high_res > (1./2)*x_Tprime_maxmax) & (T_max > 0.05)
            mask_uxbump = (x_high_res > (10/9)*x_uxprime_maxmax) & (uxprime_max > ux_max.max()/16)
            mask_tail = T_max < 1e-3
            
            x_high_res_Tbump = x_high_res[mask_Tbump]
            x_high_res_uxbump = x_high_res[mask_uxbump]
            x_high_res_tail = x_high_res[mask_tail]
            T_max_bump = T_max[mask_Tbump]
            ux_max_bump = ux_max[mask_uxbump]
            
            mask_wT = Tprime_r[:,Tprime_maxmax_where]/Tprime_max[Tprime_maxmax_where] > 0.7
            y_sort_wT = y_sort[mask_wT]
            Tprime_r_wT = Tprime_r[mask_wT,Tprime_maxmax_where]/Tprime_max[Tprime_maxmax_where]
            Topt, Tcov = np.polyfit(y_sort_wT, Tprime_r_wT, 2, cov=True)
            wT_fit = (-2*Topt[0])**(-0.5)
            wT_fit_sigma = (0.5*(-2*Topt[0])**(-1.5) * np.sqrt(Tcov[0,0]))
            
            mask_wT_direct = Tprime_r[:,Tprime_maxmax_where]/Tprime_max[Tprime_maxmax_where] > np.exp(-1./2)
            y_sort_wT_direct = y_sort[mask_wT_direct]
            wT_direct = (y_sort_wT_direct[-1] - y_sort_wT_direct[0])/2
            
            mask_wux = uxprime_r[:,uxprime_maxmax_where]/uxprime_max[uxprime_maxmax_where] > 0.7
            y_sort_wux = y_sort[mask_wux]
            uxprime_r_wux = uxprime_r[mask_wux,uxprime_maxmax_where]/uxprime_max[uxprime_maxmax_where]
            uxopt, uxcov = np.polyfit(y_sort_wux, uxprime_r_wux, 2, cov=True)
            wux_fit = (-2*uxopt[0])**(-0.5)
            wux_fit_sigma = (0.5*(-2*uxopt[0])**(-1.5) * np.sqrt(uxcov[0,0]))
            
            mask_wux_direct = uxprime_r[:,uxprime_maxmax_where]/uxprime_max[uxprime_maxmax_where] > np.exp(-1./2)
            y_sort_wux_direct = y_sort[mask_wux_direct]
            wux_direct = (y_sort_wux_direct[-1] - y_sort_wux_direct[0])/2
            
            length_Tm = x_high_res_Tbump[-1]
            length_uxm = x_high_res_uxbump[-1]
            
            aT_ = []
            au_ = []
            aTsigma_ = []
            ausigma_ = []
            
            # Plot T and u_x along y for fixed x
            figx, axx = plt.subplots(2, 2, figsize=(12, 10))

            i_start = np.where(x_high_res == x_Tprime_maxmax)[0][0]
            i_end = np.where((T_max - 0.05) < 1e-2)[0][0]
            print('i_start = ', i_start)
            print('i_end = ', i_end)
            #for i in range(15,150,15):
            for i in range(i_start, i_end, (i_end - i_start)//10):
                color = cmap((i - i_start)/(i_end - i_start))  # Adjust the color according to column index
                axx[0,0].plot(y_sort, T_r[:, i], label=f"$x={x_high_res[i]:1.2f}$", color=color) # plot T(y) for different x
                axx[0,1].plot(y_sort, ux_r[:, i], label=f"$x={x_high_res[i]:1.2f}$", color=color) # plot ux(y) for different x
                axx[1,0].plot(y_sort, Tprime_r[:, i]/Tprime_max[i], label=f"$x={x_high_res[i]:1.2f}$", color=color) # plot T(y) for different x
                axx[1,1].plot(y_sort, uxprime_r[:, i]/uxprime_max[i], label=f"$x={x_high_res[i]:1.2f}$", color=color) # plot ux(y) for different x
                
                mask_fit_Tm = Tprime_r[:,i]/Tprime_max[i] > 0.7
                y_sort_fit_Tm = y_sort[mask_fit_Tm]
                Tprime_r_fit = Tprime_r[mask_fit_Tm,i]/Tprime_max[i]
                
                mask_fit_uxm = uxprime_r[:,i]/uxprime_max[i] > 0.7
                y_sort_fit_uxm = y_sort[mask_fit_uxm]
                uxprime_r_fit = uxprime_r[mask_fit_uxm,i]/uxprime_max[i]
                
                # quadratic fit around the maximum
                Topt, Tcov = np.polyfit(y_sort_fit_Tm, Tprime_r_fit, 2, cov=True)
                uopt, ucov = np.polyfit(y_sort_fit_uxm, uxprime_r_fit, 2, cov=True)
                
                a0T = Topt[2]
                a1T = Topt[1]
                a2T = Topt[0]
                a2T_var = Tcov[0,0]
                
                a0u = uopt[2]
                a1u = uopt[1]
                a2u = uopt[0]
                a2u_var = ucov[0,0]
                
                print('a0T = ', a0T, 'a1T = ', a1T, 'a2T = ', a2T)
                axx[1,0].plot(y_sort_fit_Tm, [a2T*y**2 + a1T*y + a0T for y in y_sort_fit_Tm], color='black', linestyle='dotted')
                axx[1,1].plot(y_sort_fit_uxm, [a2u*y**2 + a1u*y + a0u for y in y_sort_fit_uxm], color='black', linestyle='dotted')
                aT_.append( -a2T )
                au_.append( -a2u )
                aTsigma_.append(np.sqrt(a2T_var))
                ausigma_.append(np.sqrt(a2u_var))
                
            axx[0,0].set_ylabel("$T$")
            axx[0,1].set_ylabel("$u_x$")
            axx[1,0].set_ylabel("$T'$")
            axx[1,1].set_ylabel("$u'_x$")
            [[axx[i,j].set_xlabel("$y$") for i in [0,1]] for j in [0,1]]
            axx[0,0].legend()
            
            figx.tight_layout()
            figx.savefig(out_dir + 'fx.pdf', dpi=300)
            
            # Plot wT and wux for different x
            figw, axw = plt.subplots(1, 2, figsize=(15, 5))
            aTopt, aTcov = curve_fit(poly1, [x_high_res[i] for i in range(i_start, i_end, (i_end - i_start)//10)], aT_, p0=[-1.0, 1.0])
            auopt, aucov = curve_fit(poly1, [x_high_res[i] for i in range(i_start, i_end, (i_end - i_start)//10)], au_, p0=[-1.0, 1.0])
            #c_wT = wTopt[2]
            #b_wT = wTopt[1]
            #a_wT = wTopt[0]
            #c_wu = wuopt[2]
            #b_wu = wuopt[1]
            #a_wu = wuopt[0]
            
            axw[0].errorbar( [x_high_res[i] for i in range(i_start, i_end, (i_end - i_start)//10)], aT_, yerr=aTsigma_, capsize=3, fmt='o')
            axw[1].errorbar( [x_high_res[i] for i in range(i_start, i_end, (i_end - i_start)//10)], au_, yerr=ausigma_, capsize=3, fmt='o')
            axw[0].plot(x_high_res_Tbump, poly1(x_high_res_Tbump, *aTopt), color='black', linestyle='dotted')
            axw[1].plot(x_high_res_uxbump, poly1(x_high_res_uxbump, *auopt), color='black', linestyle='dotted')
            
            axw[0].set_ylabel("$a_T$")
            axw[1].set_ylabel("$w_T$")
            
            figw.savefig(out_dir + 'wx.pdf', dpi=300)
            
            
            figxlog, axxlog = plt.subplots(2, 2, figsize=(12, 10))
            for i in range(15,150,15):
                color = cmap(i / (150 - 1))  # Adjust the color according to column index
                axxlog[0,0].plot(np.log(y_sort - ym), np.log(1 - T_r[:, i]/T_max[i]), label=f"$x={x_high_res[i]:1.2f}$", color=color) # plot T(y) for different x
                axxlog[0,1].plot(np.log(y_sort - ym), np.log(1 - ux_r[:, i]/ux_max[i]), color=color) # plot u_x(y) for different x
                axxlog[1,0].plot(np.log(y_sort - ym), np.log(1 - Tprime_r[:, i]/Tprime_max[i]), label=f"$x={x_high_res[i]:1.2f}$", color=color) # plot T(y) for different x
                axxlog[1,1].plot(np.log(y_sort - ym), np.log(1 - uxprime_r[:, i]/uxprime_max[i]), color=color) # plot u_x(y) for different x
                
            for i in range(Nx_high_res)[::400]:
                # color = cmap(i / Nx)  # Adjust the color according to column index
                gaussT = [2*(y_) + 0.057*x[i] for y_ in np.linspace(-5, 0, 40)]
                gaussux = [2*(y_) + 0.007*x[i] for y_ in np.linspace(-5, 0, 40)]
                
            axxlog[0,0].set_ylabel("$\log(1 - T(y-y_m)/T_m)$")
            axxlog[0,1].set_ylabel("$\log(1 - u_x(y-y_m)/u_m)$")
            axxlog[1,0].set_ylabel("$\log(1 - T'(y-y_m)/T'_m)$")
            axxlog[1,1].set_ylabel("$\log(1 - u'_x(y-y_m)/u'_m)$")
            [[axxlog[i,j].set_xlabel("$\log(y-y_m)$") for i in [0,1]] for j in [0,1]]
            axxlog[0,0].legend()
            figxlog.savefig(out_dir + 'fx_loglog.pdf', dpi=300)
            
            # Plot T and u_x along x for fixed y
            
            figy, axy = plt.subplots(2, 2, figsize=(12, 10))
            axy[0,0].plot(x_high_res, [np.exp(-x*xi) for x in x_high_res], color='black', linestyle='dashed') # plot the base state T_0(x)
            for i in range(Ny)[::25]:
                color = cmap(i / Ny)  # Adjust the color according to column index
                axy[0,0].plot(x_high_res, T_r[i, :], label=f"$y={y_sort[i]:1.2f}$", color=color) # plot T(x) for different y
                axy[0,1].plot(x_high_res, ux_r[i, :], color=color) # plot u_x(x) for different y
                axy[1,0].plot(x_high_res, Tprime_r[i, :], label=f"$y={y_sort[i]:1.2f}$", color=color) # plot T(x) for different y
                axy[1,1].plot(x_high_res, uxprime_r[i, :], color=color) # plot u_x(x) for different y
            axy[0,0].set_ylabel("$T$")
            axy[0,1].set_ylabel("$u_x$")
            axy[1,0].set_ylabel("$T'$")
            axy[1,1].set_ylabel("$u'_x$")
            axy[0,0].legend()
            [[axy[i,j].set_xlabel("$x$") for i in [0,1]] for j in [0,1]]
            figy.savefig(out_dir + 'fy.pdf', dpi=300)
            
            # Plot max of T and u_x along x for fixed y
            figmax, axmax = plt.subplots(2, 2, figsize=(12, 10))
                    
            def poly1(x, a, b):
                    return b - a*x
            
            axmax[0,0].plot(x_high_res, [np.exp(-x*xi) for x in x_high_res], color='black', linestyle='dashed')
            axmax[1,0].plot(x_high_res_tail, [1e2*np.exp(-x*xi) for x in x_high_res_tail], color='green', linestyle='dashed', label="$exp(-xi x)$")
            #axmax[1,0].plot(x_high_res_tail, [1e3*np.exp(-x*Lambda) for x in x_high_res_tail], color='black', linestyle='dotted', label="$exp(-Lambda x)$")
            axmax[1,1].plot(x_high_res_tail, [1e3*np.exp(-x*xi) for x in x_high_res_tail], color='green', linestyle='dashed', label="$exp(-xi x)$")
            
            Tmpar, Tmcov = curve_fit(poly1, x_high_res_Tbump, T_max_bump, p0=[1.0, 1.0])
            uxmpar, uxmcov = curve_fit(poly1, x_high_res_uxbump, ux_max_bump, p0=[1.0, 1.0])
            a_Tm = Tmpar[0]
            b_Tm = Tmpar[1]
            a_Tm_sigma = np.sqrt(Tmcov[0,0])
            b_Tm_sigma = np.sqrt(Tmcov[1,1])
            a_uxm = uxmpar[0]
            b_uxm = uxmpar[1]
            a_uxm_sigma = np.sqrt(uxmcov[0,0])
            b_uxm_sigma = np.sqrt(uxmcov[1,1])
            axmax[0,0].plot(x_high_res_Tbump, poly1(x_high_res_Tbump, *Tmpar), color='black', linestyle='dashed')
            axmax[0,1].plot(x_high_res_uxbump, poly1(x_high_res_uxbump, *uxmpar), color='black', linestyle='dashed')
            
            axmax[0,0].plot(x_high_res, T_max)
            axmax[0,1].plot(x_high_res, ux_max)
            axmax[1,0].plot(x_high_res, Tprime_max)
            axmax[1,1].plot(x_high_res, uxprime_max)
            
            axmax[0,0].set_ylabel("$T_{max}(x)$")
            axmax[0,1].set_ylabel("$u_{x,max}(x)$")
            axmax[1,0].set_ylabel("$T'_{max}(x)$")
            axmax[1,1].set_ylabel("$u'_{x,max}(x)$")
            
            [[axmax[i,j].set_xlabel("$x$") for i in [0,1]] for j in [0,1]]
            
            axmax[0,0].legend()
            #axmax[0,0].semilogy()
            #axmax[0,1].semilogy()
            #axmax[1,0].semilogy()
            #axmax[1,1].semilogy()
            
            # Print measured values
            with open(out_dir_2 + 'values.txt', 'a') as output_file:
                output_file.write(f'{Pe}\t{Gamma}\t{beta}\t{k}\t') # 0-3: input parameters
                output_file.write(f'{a_Tm}\t{a_Tm_sigma}\t{b_Tm}\t{b_Tm_sigma}\t{a_uxm}\t{a_uxm_sigma}\t{b_uxm}\t{b_uxm_sigma}\t') # 4-11: coeff. linear fit
                output_file.write(f'{length_Tm}\t{length_uxm}\t') # 12-13: finger length
                output_file.write(f'{x_ux_maxmax}\t{ux_maxmax}\t{x_Tprime_maxmax}\t{Tprime_maxmax}\t') # 14-17: max pos. and value
                output_file.write(f'{wT_direct}\t{wux_direct}\t{wT_fit}\t{wT_fit_sigma}\t{wux_fit}\t{wux_fit_sigma}\n') # 18-23: peak width
            figmax.savefig(out_dir + 'maxfy.pdf', dpi=300)
        
        
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
            figu.savefig(out_dir + f'umap.pdf', dpi=300)
        
        if print_colormaps:
            if (rnd == False): # sin perturbation
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
                
                Lxlim = 7.1e5
                [axi.set_xlim(0, Lxlim) for axi in axTs]
                [axi.set_ylim(0, Ly) for axi in axTs]
                [axi.set_aspect('auto') for axi in axTs]
                
                # Create a ScalarFormatter
                formatter = ticker.ScalarFormatter(useOffset=True, useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-3, 3)) # Example: scientific notation for exponents outside -3 to 4

                # Apply the formatter to the x-axis
                [axi.yaxis.set_major_formatter(formatter) for axi in axTs]
                [axi.xaxis.set_major_formatter(formatter) for axi in axTs]
                
                imgname = '/Tlevel_and_streamlines_sin.pdf'
                figTs.savefig(out_dir + imgname, dpi=400, bbox_inches="tight")
               
            else: # rnd perturbation
                figTT, axTT = plt.subplots(1, 2, figsize=(15., 5.))
                
                im_Te = axTT[0].pcolormesh(X_low_res, Y_low_res, T_r_early_frame, vmin=0., vmax=1., cmap='plasma', alpha=0.7, edgecolors='none', linewidth=0, shading='gouraud') # plot of colormap of T
                cb_Te = plt.colorbar(im_Te, ax=axTT[0], location='right', orientation='vertical') # colorbar
                cb_Te.set_label(r'$T(x,y)$', labelpad=10)
                #cb_T.ax.xaxis.set_label_position('top')  # Move label to the top
                cse = axTT[0].contour(X_low_res, Y_low_res, T_r_early_frame, levels=levels, linewidths=0.9, colors="k") # plot of different levels on the colormap
            
                im_Tl = axTT[1].pcolormesh(X_low_res, Y_low_res, T_r_frame, vmin=0., vmax=1., cmap='plasma', alpha=0.7, edgecolors='none', linewidth=0, shading='gouraud') # plot of colormap of T
                cb_Tl = plt.colorbar(im_Tl, ax=axTT[1], location='right', orientation='vertical') # colorbar
                cb_Tl.set_label(r'$T(x,y)$', labelpad=10)
                csl = axTT[1].contour(X_low_res, Y_low_res, T_r_frame, levels=levels, linewidths=0.8, colors="k") # plot of different levels on the colormap
                
                [axi.set_xlabel("$x$") for axi in axTT]
                [axi.set_ylabel("$y$") for axi in axTT]
                
                ylab_xpos_l_b = axTT[0].yaxis.get_label().get_position()[0]  # horizontal position of y-label
                ylab_xpos_r_b = axTT[0].yaxis.get_label().get_position()[1]  # horizontal position of y-label
                figTT.text(ylab_xpos_l_b + 0.075, 0.98, "($a$)", verticalalignment='top', horizontalalignment='right')
                figTT.text(ylab_xpos_r_b + 0.025, 0.98, "($b$)", verticalalignment='top', horizontalalignment='right')
                
                Lxlim = 7e5
                [axi.set_xlim(0, Lxlim) for axi in axTT]
                [axi.set_ylim(0, Ly) for axi in axTT]
                [axi.set_aspect('auto') for axi in axTT]
                
                # Create a ScalarFormatter
                formatter = ticker.ScalarFormatter(useOffset=True, useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-3, 3)) # Example: scientific notation for exponents outside -3 to 4

                # Apply the formatter to the x-axis
                [axi.yaxis.set_major_formatter(formatter) for axi in axTT]
                [axi.xaxis.set_major_formatter(formatter) for axi in axTT]
                
                
                figTT.savefig(out_dir + '/Tlevels_rnd.pdf', dpi=300, bbox_inches="tight")
                
        plt.show()
        plt.close()
    
    if (growth==False):
        exit(0)
    # Analyze time evolution
    
    nn = 2 if twoperiods else 1
    Nymax_2 = 5*Ny//(nn*4)
    Nymin_2 = 7*Ny//(nn*4)
    
    for it in it_:
        t = t_[it] # time at step it
        print(f"it={it} t={t}")

        # Load data
        dset_T = dsets_T[t] # T-dictionary at time t
        with h5py.File(dset_T[0], "r") as h5f:
            T_[:] = h5f[dset_T[1]][:, 0] # Takes values of T from the T-dictionary
        T_sorted = T_[sort_indices]
        T_r_ = T_sorted.reshape((Ny + 1, Nx + 1))

        with h5py.File(dsets_p[t][0], "r") as h5f:
            p_[:] = h5f[dsets_p[t][1]][:, 0] # Takes values of p from the p-dictionary
        p_sorted = p_[sort_indices]
        p_r_ = p_sorted.reshape((Ny + 1, Nx + 1))
        
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
        
        #Tprime_r = T_r - np.exp(-xi*X_high_res[:, :])
        #uxprime_r = ux_r - 1.
        
        for i in range(0,n_samples):
        
            Nxi = int(Nx*0.5*(i+1)/(n_samples+1))
            if it < 1:
                print('Nxi = ', Nxi)
            Nymax = np.argmax(T_r[:,Nxi]) if rnd else Ny//(nn*2)
            Nymin = np.argmin(T_r[:,Nxi]) if rnd else 0
            Txt[i][it] = T_r[Nymax,Nxi] - T_r[Nymin,Nxi]
            uxt[i][it] = ux_r[Nymax,Nxi] - ux_r[Nymin,Nxi]
            
            if twoperiods:
                Txt_2[i][it] = T_r[Nymax_2,Nxi] - T_r[Nymin_2,Nxi]
                uxt_2[i][it] = ux_r[Nymax_2,Nxi] - ux_r[Nymin_2,Nxi]
                
        n_in = n_steps//5
        n_out = 3*n_steps//5
        if ((it > n_in) and (it < n_out+1) and ((it - n_in) % ((n_out - n_in)//(n_samples)) == 0)):
        
            i = (it - n_in) * (n_samples) // (n_out - n_in) - 1
            print("new time sample: i = ", i)
            Nymax = Ny//(nn*2)
            Nymin = 0
            T_r_max = np.max(T_r[Nymax,:] - T_r[Nymin,:])
            ux_r_max = np.max(ux_r[Nymax,:] - ux_r[Nymin,:])
            print('T_r_max = ', T_r_max)
            print('ux_r_max = ', ux_r_max)
            for j in range(0,Nx):
                Ttx[i][j] = (T_r[Nymax,j] - T_r[Nymin,j]) / T_r_max
                utx[i][j] = (ux_r[Nymax,j] - ux_r[Nymin,j]) / ux_r_max
            utx[i][0] = 0
    
    #n_i = int(n_steps * t_i/t_end)
    #n_f = int(n_steps * t_f/t_end)
    n_i = 6*n_steps//16
    n_f = 10*n_steps//16
    
    x_i = 8*Nx//16
    x_f = 13*Nx//16
    
    gamma_ = np.zeros(n_samples)
    gamma_std_ = np.zeros(n_samples)
    
    Lambda_ = np.zeros(n_samples)
    Lambda_std_ = np.zeros(n_samples)
    
    if twoperiods:
        gamma2_ = np.zeros(n_samples)
        gamma2_std_ = np.zeros(n_samples)
    
    for i in range(0,n_samples):
        model, cov = np.polyfit(t_[n_i:n_f], np.log(Txt[i][n_i:n_f]), 1, cov=True)
        gamma_[i] = model[0]
        gamma_std_[i] = np.sqrt(cov[0, 0])
        print("x = ", i+1, ", gamma_ = ", gamma_[i], ", gamma_std_ = ", gamma_std_[i])
        
        model_tx, cov_tx = np.polyfit(x_sort[x_i:x_f], np.log(Ttx[i][x_i:x_f]), 1, cov=True)
        Lambda_[i] = -model_tx[0]
        Lambda_std_[i] = np.sqrt(cov_tx[0, 0])
        print("t = ", n_in*dt + i*(n_out-n_in)*dt//n_samples, ", Lambda_ = ", Lambda_[i], ", Lambda_std_ = ", Lambda_std_[i])
        
    gamma_avg = np.average(gamma_, weights=1/gamma_std_**2)
    print(f"gamma_Tspan = {gamma_avg}")
    
    Lambda_avg = np.average(Lambda_, weights=1/Lambda_std_**2)
    print(f"Lambda_Tspan = {Lambda_avg}")
    
    # Plot Tspan(x, t) and uxspan(x, t) vs t for different x
    
    figspan, axspan = plt.subplots(1, 2, figsize=(15., 5))
        
    for i in range(0,n_samples):
        color = cmap_space(1. - (i+1) / n_samples)
        axspan[0].plot(t_[0:n_steps], Txt[i][0:n_steps], label=fr'$x = {i:1f}$', color=color)
        axspan[1].plot(t_[0:n_steps], uxt[i][0:n_steps], label=fr'$x = {i:1f}$', color=color)

    aT = 2*1e-9 if rnd else 1.5*1e-5
    aux = 9*1e-9 if rnd else 1e-4
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
    
    if (not rnd):
        #axspan[0].set_ylim(1e-9, 2)
        #axspan[1].set_ylim(1e-8, 200)
        #[axspani.set_xlim(-5e4, 2e6) for axspani in axspan]
    #else:
        axspan[0].set_ylim(np.sqrt(10)*1e-7, 2)
        axspan[1].set_ylim(np.sqrt(10)*1e-6, 20)
        [axspani.set_xlim(-5e4, 1.1e6) for axspani in axspan]
    
    imgnamespan = '/Tspan_and_uxspan_rnd.pdf' if rnd else '/Tspan_and_uxspan_sin.pdf'
    figspan.savefig(out_dir + imgnamespan, dpi=600, bbox_inches="tight")
    
    # Plot Tspan(x, t)/Tmax(t) and uxspan(x, t)/umax(t) vs x for different t
    
    figspan2, axspan2 = plt.subplots(1, 2, figsize=(15., 5))
    
    for i in range(0,n_samples):
        color = cmap_time(1. - (i+1) / n_samples)
        axspan2[0].plot(x_sort[0:Nx], Ttx[i][0:Nx], label=fr'$x = {i:1f}$', color=color)
        axspan2[1].plot(x_sort[0:Nx], utx[i][0:Nx], label=fr'$x = {i:1f}$', color=color)
        
    bT = 2*1e-9 if rnd else 0.5*1e3
    bux = 9*1e-9 if rnd else 0.5*1e3
    axspan2[0].plot(x_sort[x_i:x_f], bT*np.exp(-Lambda_[n_samples-1]*x_sort[x_i:x_f]), color='black', linestyle='dashed')
    axspan2[1].plot(x_sort[x_i:x_f], bux*np.exp(-Lambda_[n_samples-1]*x_sort[x_i:x_f]), color='black', linestyle='dashed')
    text_idx_2 = (x_i + x_f)//2
    text_fit_2 = r"$\propto e^{-\Lambda t}$"
    axspan2[0].text(x_sort[text_idx_2], bT*np.exp(-Lambda_[n_samples-1]*x_sort[text_idx_2]), text_fit_2, va="bottom", ha="left")
    axspan2[1].text(x_sort[text_idx_2], bux*np.exp(-Lambda_[n_samples-1]*x_sort[text_idx_2]), text_fit_2, va="bottom", ha="left")
    
    [axi.semilogy() for axi in axspan2]
    [axi.set_xlabel(r"$x$") for axi in axspan2]
    axspan2[0].set_ylabel(r"$T^{\rm span}(x,t)/T^{\rm max}(t)$")
    axspan2[1].set_ylabel(r"$u_{x}^{\rm span}(x,t)/T^{\rm max}(t)$")
    
    ylab_xpos_l_b_2 = axspan2[0].yaxis.get_label().get_position()[0]  # horizontal position of y-label
    ylab_xpos_r_b_2 = axspan2[0].yaxis.get_label().get_position()[1]  # horizontal position of y-label
    figspan2.text(ylab_xpos_l_b_2 + 0.075, 0.98, "($c$)", verticalalignment='top', horizontalalignment='right')
    figspan2.text(ylab_xpos_r_b_2 + 0.025, 0.98, "($d$)", verticalalignment='top', horizontalalignment='right')
    
    axspan2[0].set_ylim(1e-9, 2)
    axspan2[1].set_ylim(np.sqrt(10)*1e-9, 2)
    [axspani.set_xlim(-np.sqrt(10)*1e4, 7e5) for axspani in axspan2]
    
    # Create a ScalarFormatter
    formatter = ticker.ScalarFormatter(useOffset=True, useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3)) # Example: scientific notation for exponents outside -3 to 4
    
    # Apply the formatter to the x-axis
    [axi.xaxis.set_major_formatter(formatter) for axi in axspan2]
    
    imgnamespan2 = '/Tspan_and_uxspan_2_rnd.pdf' if rnd else '/Tspan_and_uxspan_2_sin.pdf'
    figspan2.savefig(out_dir + imgnamespan2, dpi=600, bbox_inches="tight")
    
    if twoperiods:
    
        for i in range(0,n_samples):
            model2, cov2 = np.polyfit(t_[n_i:n_f], np.log(Txt_2[i][n_i:n_f]), 1, cov=True)
            gamma2_[i] = model2[0]
            gamma2_std_[i] = np.sqrt(cov2[0, 0])
        gamma2_avg = np.average(gamma2_, weights=1/gamma_std_**2)
        print(f"gamma_Tspan_2 = {gamma2_avg}")
    
        figspan2, axspan2 = plt.subplots(1, 2, figsize=(15., 5))
        
        for i in range(0,n_samples):
            color = cmap_space(1. - (i+1) / n_samples)
            axspan2[0].plot(t_[0:n_evol], Txt_2[i][0:n_evol], label=fr'$x = {i:1f}$', color=color)
            axspan2[1].plot(t_[0:n_evol], uxt_2[i][0:n_evol], label=fr'$x = {i:1f}$', color=color)

        axspan2[0].plot(t_[n_i:n_f], aT*np.exp(gamma2_avg*t_[n_i:n_f]), color='black', linestyle='dashed')
        axspan2[1].plot(t_[n_i:n_f], aux*np.exp(gamma2_avg*t_[n_i:n_f]), color='black', linestyle='dashed')
        
        text_fit = r"$\propto e^{\gamma_2 t}$"
        axspan2[0].text(t_[text_idx], aT*np.exp(gamma_avg*t_[text_idx]), text_fit, va="bottom", ha="right")
        axspan2[1].text(t_[text_idx], aux*np.exp(gamma_avg*t_[text_idx]), text_fit, va="bottom", ha="right")
        
        [axi.semilogy() for axi in axspan2]
        [axi.set_xlabel(r"$t$") for axi in axspan2]
        axspan2[0].set_ylabel(r"$T^{\rm span}(x,t)$")
        axspan2[1].set_ylabel(r"$u_{x}^{\rm span}(x,t)$")
        
        imgnamespan = '/Tspan_and_uxspan_sin_2periods.pdf'
        figspan2.savefig(out_dir + imgnamespan, dpi=600, bbox_inches="tight")
        
    cmap = plt.cm.viridis
    figf, axf = plt.subplots(1, 3, figsize=(20, 5))
    #figf, axf = plt.subplots(1, 1)
    figf.subplots_adjust(wspace=0.3)
    
    figff, axff = plt.subplots(1, 1)
    
    gamma2_ = np.zeros(len(levels[1:-1])) # growth rate for each level
    tstat_ = np.zeros(len(levels[1:-1])) # time to reach the stationary state for each level
    i = 0
    
    for level in levels[1:-1]:
    
        # Plot xmax, xmin, xspan vs t
        color = cmap(level)
        xbase = -math.log(level) / xi
        axf[0].plot(t_[:n_steps], xmax[level][:n_steps], color=color) # plot xmax vs t for each level
        axf[1].plot(t_[:n_steps], np.abs(xmin[level][:n_steps]), color=color) # plot xmin vs t for each level
        axf[2].plot(t_[1:n_steps], (xmax[level][1:n_steps] - xmin[level][1:n_steps]), label=f"$T={level:1.2f}$", color=color) # plot span vs t for each level
        
        # Save xspan vs t in file .txt
        xspan_data =  np.column_stack(( t_[1:n_steps], xmax[level][1:n_steps] - xmin[level][1:n_steps] ))
        #np.savetxt(out_dir + f'/xspan_T={level:1.2f}.txt', xspan_data, fmt='%1.9f')
        
        # Plot xspan vs t at growing stage and find gamma and tstat
        axff.plot(t_[n_i:n_f], np.log(xmax[level][n_i:n_f] - xmin[level][n_i:n_f]), label=f"$T={level:1.2f}$", color=color)
        model = LinearRegression().fit(t_[n_i:n_f].reshape((-1, 1)), np.log(xmax[level][n_i:n_f] - xmin[level][n_i:n_f]))
        xspan_sat = xmax[level][n_steps-1] - xmin[level][n_steps-1] # stationary value for xspan
        gamma2_[i] = model.coef_[0]
        tstat_[i] = (np.log(xspan_sat) - model.intercept_)/ model.coef_[0]
        i += 1
        
    gamma_deltax = np.mean(gamma2_)
    print('gamma_deltax = ', gamma_deltax)
    axf[2].plot(t_[n_i:n_f], 1e-3*np.exp(gamma_deltax*t_[n_i:n_f]), color='black', linestyle='dashed') # plot fitting line
    axf[2].text(t_[n_steps//2], 1e-3*np.exp(gamma_deltax*t_[n_steps//2]), r"$\propto e^{\gamma_{\max} t}$", va="bottom", ha="right")
    
    [axfi.set_xlabel("$t$") for axfi in axf]
    axf[0].set_ylabel("$x_{max}$")
    axf[1].set_ylabel("$x_{min}$")
    axf[2].set_ylabel("$x_{max}-x_{min}$")
    axf[2].semilogy()
    [axfi.tick_params(axis='both', which='major') for axfi in axf]
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
