import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from mpi4py import MPI
mpi_size = MPI.COMM_WORLD.Get_size()
mpi_rank = MPI.COMM_WORLD.Get_rank()
if mpi_size > 1:
    if mpi_rank == 0:
        print("This script only works in serial. You are better off  \n"
              "simply running the parameter scan in parallel instead.")
    exit()

class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < df.DOLFIN_EPS

class Right(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] > df.DOLFIN_EPS

def parse_args():
    parser = argparse.ArgumentParser(description="Solve the linearised model")
    parser.add_argument("--Pe", default=100, type=float, help="Peclet number")
    parser.add_argument("--k", default=3.1415926536, type=float, help="Wavelength") #1.5707963268
    parser.add_argument("--Gamma", default=1.0, type=float, help="Heat conductivity")
    parser.add_argument("--beta", default=1e-3, type=float, help="Viscosity ratio")
    parser.add_argument("--eps", default=1e-3, type=float, help="Perturbation amplide")
    parser.add_argument("--tpert", default=0.1, type=float, help="Perturbation duration")
    parser.add_argument("--dt", default=0.01, type=float, help="Timestep")
    parser.add_argument("--nx", default=4000, type=int, help="Number of mesh points")
    parser.add_argument("--Lx", default=25.0, type=float, help="System size")
    parser.add_argument("--tmax", default=20.0, type=float, help="Total time")
    parser.add_argument('--plot', action='store_true', help='Flag for plotting the eigenfunctions')
    parser.add_argument('--latex', action='store_true', help='Flag for plotting in LaTeX style')
    parser.add_argument('--savegamma', action='store_true', help='Flag for saving the growth rate in a .txt file')
    parser.add_argument('--savexmax', action='store_true', help='Flag for saving xmax in a .txt file')
    parser.add_argument('--aesthetic', action='store_true', help='Flag for saving data in gamma_linear_plot.txt')
    parser.add_argument('--notaylor', action='store_true', help='Flag for removing Taylor dispersion')
    return parser.parse_args()

if __name__ == "__main__":

    cmap_space = sns.color_palette("mako", as_cmap=True)
    cmap_time = sns.color_palette("rocket", as_cmap=True)
    #cmap_space = plt.cm.cividis
    #cmap_time = plt.cm.cividis
    
    args = parse_args()

    dt = args.dt
    nx = args.nx
    Lx = args.Lx
    tmax = args.tmax

    Pe = args.Pe
    kk = args.k
    Gamma = args.Gamma
    beta = args.beta
    eps = args.eps
    tpert = args.tpert

    plot = args.plot
    latex = args.latex
    savegamma = args.savegamma
    savexmax = args.savexmax
    aesthetic = args.aesthetic
    notaylor = args.notaylor
    
    if latex:
        plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 24,  # Default text size (JFM uses ~8pt for labels)
        "axes.labelsize": 24,  # Axis labels (JFM ~8pt)
        "xtick.labelsize": 24,  # Tick labels
        "ytick.labelsize": 24,
        "legend.fontsize": 12,  # Legend size
        "figure.figsize": (5.5, 3),  # Keep plots compact (JFM prefers small plots)
        "lines.linewidth": 1.5,  # Thin lines
        "lines.markersize": 8,  # Small but visible markers
        "figure.subplot.wspace": 0.35,  # Horizontal spacing
        "figure.subplot.bottom": 0.15,  # Space for x-labels
        "axes.labelpad": 8, #default is 5
    })
    
    
    plot_intv = 100

    #k = df.Constant(2*np.pi/lam)
    k = df.Constant(kk)
    
    kappa = 1./Pe
    if (notaylor == True):
        kappa_par = 0.
    else:
        kappa_par = 2.0 / 105 * Pe # equivalent to (2.0 / 105) * Pe
    kappa_eff = kappa + kappa_par
    psi = -np.log(beta)
    xi = (- 1 + np.sqrt(1 + 4*kappa_eff*Gamma))/(2*kappa_eff)

    #print('kappa_eff = ', kappa_eff, ', xi = ', xi, ', 1/xi = ',1.0/xi)

    # define mesh
    mesh = df.UnitIntervalMesh(nx)
    mesh.coordinates()[:, 0] = mesh.coordinates()[:, 0]**4
    mesh.coordinates()[:] *= Lx

    S_el = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W_el = df.MixedElement([S_el, S_el])

    # define function spaces
    W = df.FunctionSpace(mesh, W_el)
    S = W.sub(0).collapse()
    
    # define trial and test functions
    T, u = df.TrialFunctions(W)
    Q, v = df.TestFunctions(W)

    # define function that will host the solution
    w_ = df.Function(W)
    T_, u_ = df.split(w_)
    
    # define x-variable as a function
    x = df.Function(S)
    x.interpolate(df.Expression("x[0]", degree=1))
    x_values = x.vector()[:]
    
    list_i = []
    x_sample_ = [n for n in np.arange(1., 11., 1.)]
    len_x_sample_ = len(x_sample_)
    for x_sample in x_sample_:
        i = np.argmin(np.abs(x_values - x_sample))  # Index of the closest value
        list_i.append(i)
    
    #print("list_i = ", list_i)
    #for i in list_i:
    #    print("x_values[", i ,"] = ", x_values[i])
    #exit(0)
    
    # define base state function
    T0 = df.Function(S)
    T0.vector()[:] = np.exp(-xi*x_values)

    betamT0 = df.Function(S)
    betamT0.vector()[:] = beta**-T0.vector()[:]

    # mesh function to mark different boundaries of the mesh
    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    subd.set_all(0) # initialize the markers to 0
    left = Left()
    left.mark(subd, 1) # mark the left part of the border with label 1
    right = Right()
    right.mark(subd, 2) # mark the right part of the border with label 2

    # define a measure ds to integrate over the marked subdomains of the boundary
    ds = df.Measure("ds", domain=mesh, subdomain_data=subd)

    onoff = df.Constant(eps)

    # define equations in variational form
    
    FT = (T - T_) / dt * Q * df.dx \
        + T.dx(0) * Q * df.dx \
        + kappa_eff * T.dx(0) * Q.dx(0) * df.dx \
        + (kappa * k**2 + Gamma) * T * Q * df.dx \
        + xi * T0 * ( -(1 + 2 * kappa_par * xi) * u + kappa_par * u.dx(0) ) * Q * df.dx

    Fu = - u.dx(0) * v.dx(0) * df.dx \
        + psi * xi * T0 * u.dx(0) * v * df.dx \
        - k**2 * u * v * df.dx \
        + k**2 * psi * T * v * df.dx

    F = FT + Fu

    # define boundary conditions
    
    bc_T_l = df.DirichletBC(W.sub(0), 0., subd, 1) # apply T = 0 to the left boundary
    #bc_T_r = df.DirichletBC(W.sub(0), 0., subd, 2) # apply T = 0 to the right boundary
    bc_u_l = df.DirichletBC(W.sub(1), onoff, subd, 1) # apply u = onoff to the left boundary

    bcs = [bc_T_l, bc_u_l]

    # split equation in left and right member
    a, L = df.lhs(F), df.rhs(F)
    
    datamax = []
    dataxmax = []
    data = []
    
    t = 0.0
    it = 0
    
    #print(x_values[300:900])

    fig, axa = plt.subplots(1, 2, figsize=(15., 5.))
    fig, axb = plt.subplots(1, 2, figsize=(15., 5.))
    
    while t < tmax:
        print('t = ', t)
        #print('before solving: w_.vector() = ', w_.vector()[:])

        if t > tpert:
            onoff.assign(0.)

        df.solve(a == L, w_, bcs=bcs)
        
        t += dt

        T__, u__ = w_.split(deepcopy=True)
        
        T_values = T__.vector()[:]
        u_values = u__.vector()[:]

        Tmax = np.max(T_values)
        umax = np.max(u_values)
        datamax.append([t, Tmax, umax]) # collect the (t, Tmax, pmax) values
        
        idx = np.abs(T_values - Tmax).argmin()
        xmax = T_values[idx]
        dataxmax.append([xmax])
        
        data.append([t])
        for i in list_i:
            data[it].extend([T_values[i], u_values[i]])
            
        it += 1
        
        if (it % plot_intv == 0 and plot):
            color = cmap_time(1. - 0.75*t/tmax)
            axa[0].plot(x_values, T_values, label=f"$t={t:1.2f}$", color=color) # plot T vs x
            axa[1].plot(x_values, u_values, label=f"$t={t:1.2f}$", color=color) # plot ux vs x
            
            axb[0].plot(x_values, T_values/Tmax, label=f"$t={t:1.2f}$", color=color) # plot T/Tmax vs x
            axb[1].plot(x_values, u_values/umax, label=f"$t={t:1.2f}$", color=color) # plot ux/ux_max vs x
            #axb[0].set_xlim(0, 20)
            #axb[1].set_xlim(0, 20)
            
    datamax = np.array(datamax)
    data = np.array(data)
    n_steps = len(datamax[:, 0])
    istart = int(tmax/dt)//2
    #print("istart = ", istart)

    # Find the growth rate gamma
    
    popt, pcov = np.polyfit(datamax[istart:, 0], np.log(datamax[istart:, 1]), 1, cov=True)
    gamma = popt[0]
    gamma_variance = pcov[0, 0]
    gamma_standard_error = np.sqrt(gamma_variance)
    
    print(f"gamma = {gamma}")
    print(f"gamma_standard_error = {gamma_standard_error}")
    
    # Write the values in the related files
    
    Pe_str = f"Pe_{Pe:.10g}"
    Gamma_str = f"Gamma_{Gamma:.10g}"
    beta_str = f"beta_{beta:.10g}"

    if savegamma == True:
        output_folder = f"results/output_"
        if notaylor:
            output_folder += "notaylor_"
        output_folder += "_".join([Pe_str, Gamma_str, beta_str]) + "/"
        os.makedirs(output_folder, exist_ok=True)
        aesth = "_plot" if aesthetic else ""
        output_file = output_folder + f"gamma_linear{aesth}.txt"
        with open(output_file, "a") as file:
            file.write(f"\n{kk}\t{gamma}\t{gamma_standard_error}")
        
    if savexmax == True:
        k_str = f"k_{kk:.10g}"
        idx = np.abs(T_values - Tmax).argmin()
        xmax = x_values[idx]
        with open("results/output_mix/xmax_vs_Pe_" + "_".join([Gamma_str, beta_str, k_str]) + ".txt", "a") as file:
            file.write(f"{Pe}\t{xmax}\t{Tmax}\n")
    
    # Plot results
    
    if plot:
        
        # Plot T_max vs time and the semi-log fit
        
        fig_, ax_ = plt.subplots(1, 1, figsize=(7.5, 5.))
        ax_.plot(datamax[:, 0], datamax[:, 1]) # plot Tmax vs time
        #ax_.plot(datamax[istart//2:9*n_steps//10, 0], 1e-4*np.exp(gamma*datamax[istart//2:9*n_steps//10, 0]), color='black', linestyle='dotted', label=r"fit") # plot fitting line
        ax_.plot(datamax[istart:, 0], 1e-4*np.exp(gamma*datamax[istart:, 0]), color='black', linestyle='dotted', label=r"fit") # plot fitting line
        ax_.semilogy()
        ax_.set_xlabel(r"$t$")
        ax_.set_ylabel(r"$T_{\rm max}$")
        ax_.legend()
        fig_.subplots_adjust(left=0.15)
        
        # Plot T and ux vs time
        
        figt, axt = plt.subplots(1, 2, figsize=(15., 5.))
        i = 0
        while i < len_x_sample_:
            color = cmap_space(1. - i / len_x_sample_)
            x_sample = x_sample_[i]
            axt[0].plot(data[:, 0], data[:, 2*i + 1], label=f"$x={x_sample:1.1f}$", color=color) # plot T(x=x0) vs time
            axt[1].plot(data[:, 0], data[:, 2*i + 2], label=f"$x={x_sample:1.1f}$", color=color) # plot ux(x=x0) vs time
            i += 1
        n_start = istart//3
        n_end = 9*n_steps//10
        axt[0].plot(datamax[n_start:n_end, 0], 1e-4*np.exp(gamma*datamax[n_start:n_end, 0]), color='black', linestyle='dashed') # plot fitting line
        axt[1].plot(datamax[n_start:n_end, 0], 1e-3*np.exp(gamma*datamax[n_start:n_end, 0]), color='black', linestyle='dashed')
        text_idx = (n_end + n_start)//2
        axt[0].text(datamax[text_idx, 0], 1e-4*np.exp(gamma*datamax[text_idx, 0]), r"$\propto e^{\gamma t}$", va="bottom", ha="right")
        axt[1].text(datamax[text_idx, 0], 1e-3*np.exp(gamma*datamax[text_idx, 0]), r"$\propto e^{\gamma t}$", va="bottom", ha="right")
        [axti.semilogy() for axti in axt]
        [axti.set_xlabel(r"$t$") for axti in axt]
        #axt[0].legend(fontsize=12)
        axt[0].set_ylabel(r"$T_k(x,t)$")
        axt[1].set_ylabel(r"$u_k(x,t)$")
        axt[0].set_ylim(np.sqrt(10)*1e-7, 200) # for imgs
        axt[1].set_ylim(np.sqrt(10)*1e-6, 2000) # for imgs
        ylab_xpos_l = axt[0].yaxis.get_label().get_position()[0]  # horizontal position of y-label
        ylab_xpos_r = axt[0].yaxis.get_label().get_position()[1]  # horizontal position of y-label
        figt.text(ylab_xpos_l + 0.075, 0.98, "($a$)", verticalalignment='top', horizontalalignment='right')
        figt.text(ylab_xpos_r + 0.025, 0.98, "($b$)", verticalalignment='top', horizontalalignment='right')
        
        # Plot T and ux vs x
        
        [axai.set_xlabel(r"$x$") for axai in axa]
        axa[0].set_ylabel(r"$T_k(x)$")
        axa[1].set_ylabel(r"$u_k(x)$")
        #for axi in axa:
        #    xmin, xmax = 0, 20
        #    padding = 0.05 * (xmax - xmin)  # 5% padding
        #    axi.set_xlim(xmin - padding, xmax + padding)
        #axa[0].legend()
        
        G = gamma + kk*kappa + Gamma
        Lambda_k = (-1 + np.sqrt(1 + 4*G*kappa_eff) )/(2*kappa_eff) # decay rate of the solution of the ODE for T0 = 0
        print("Lambda_k = ", Lambda_k)
        
        # Plot T/T_max and ux/ux_max vs x
        
        n_start_x = 200
        n_end_x = 1000
        axb[0].plot(x_values[n_start_x:n_end_x], [100 * np.exp(-Lambda_k * x_) for x_ in x_values[n_start_x:n_end_x]], color='black', linestyle='dashed', label="$\sim e^{-\Lambda x}$")
        #axb[1].plot(x_values[n_start_x:n_end_x], [250 * (np.exp(- Lambda_k * x_) + np.exp(- kk * x_)) for x_ in x_values[n_start_x:n_end_x]], color='black', linestyle='dotted', label="$\sim Ae^{-\Lambda*x} + Be^{-k*x}$")
        text_idx_x = (n_end_x + n_start_x)//2
        axb[0].text(x_values[text_idx_x], 100*np.exp(-Lambda_k*x_values[text_idx_x]), r"$\propto e^{-\Lambda x}$", va="bottom", ha="left")
        axb[0].set_ylim(3*1e-10, 1e1) # for imgs
        axb[1].set_ylim(8*1e-7, 5) # for imgs
        ylab_xpos_l_b = axb[0].yaxis.get_label().get_position()[0]  # horizontal position of y-label
        ylab_xpos_r_b = axb[0].yaxis.get_label().get_position()[1]  # horizontal position of y-label
        fig.text(ylab_xpos_l_b + 0.075, 0.98, "($a$)", verticalalignment='top', horizontalalignment='right')
        fig.text(ylab_xpos_r_b + 0.025, 0.98, "($b$)", verticalalignment='top', horizontalalignment='right')
        [axbi.set_xlabel(r"$x$") for axbi in axb]
        axb[0].set_ylabel(r"$T_k(x,t)/T_{\max}(t)$")
        axb[1].set_ylabel(r"$u_k(x,t)/u_{\max}(t)$")
        for axi in axb: # for imgs
            xmin, xmax = 0, 20
            padding = 0.05 * (xmax - xmin)  # 5% padding
            axi.set_xlim(xmin - padding, xmax + padding)
        #axb[0].legend()
        axb[0].semilogy()
        axb[1].semilogy()
        
        outpute_imgs_folder = "results/imgs"
        fig.savefig(outpute_imgs_folder + "/T_and_u_vs_x.pdf", dpi=600, bbox_inches="tight")
        #fig_.savefig(outpute_imgs_folder + "/Tmax_vs_t.pdf", dpi=600, bbox_inches="tight")
        figt.savefig(outpute_imgs_folder + "/T_and_u_vs_t.pdf", dpi=600, bbox_inches="tight")
        
        plt.show()
