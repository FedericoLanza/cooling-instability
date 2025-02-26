import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
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
    parser.add_argument("-Pe", default=1e5, type=float, help="Peclet number")
    parser.add_argument("-k", default=1, type=float, help="Wavelength")
    parser.add_argument("-Gamma", default=1.0, type=float, help="Heat conductivity")
    parser.add_argument("-beta", default=1e-1, type=float, help="Viscosity ratio")
    parser.add_argument("-eps", default=1e-3, type=float, help="Perturbation amplide")
    parser.add_argument("-tpert", default=0.1, type=float, help="Perturbation duration")
    parser.add_argument("-dt", default=0.01, type=float, help="Timestep")
    parser.add_argument("-nx", default=6000, type=int, help="Number of mesh points")
    parser.add_argument("-Lx", default=300.0, type=float, help="System size")
    parser.add_argument("-tmax", default=10.0, type=float, help="Total time")
    return parser.parse_args()

if __name__ == "__main__":

    cmap = plt.cm.viridis
    
    args = parse_args()

    dt = args.dt
    nx = args.nx
    Lx = args.Lx
    tmax = args.tmax # 10.0

    Pe = args.Pe # 10**0.75 #100.0
    kk = args.k # 2.0 # 4.0
    Gamma = args.Gamma #1.0
    beta = args.beta # 0.001
    eps = args.eps # 1e-1
    tpert = args.tpert # 0.1

    plot_intv = 100
    
    #k = df.Constant(2*np.pi/lam)
    k = df.Constant(kk)

    kappa = 1.0 / Pe
    kappa_par = 2.0 / 105 * Pe # equivalent to (2.0 / 105) * Pe
    kappa_eff = kappa + kappa_par
    psi = -np.log(beta)
    xi = (- 1 + np.sqrt(1 + 4*kappa_eff*Gamma))/(2*kappa_eff)
    
    #print('Pe = ', Pe, ', beta = ', beta, ', Gamma = ', Gamma)
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
    T, p = df.TrialFunctions(W)
    U, q = df.TestFunctions(W)

    # define function that will host the solution
    w_ = df.Function(W)
    T_, p_ = df.split(w_)
    
    # define x-variable as a function
    x = df.Function(S)
    x.interpolate(df.Expression("x[0]", degree=1))

    # define base state function
    T0 = df.Function(S)
    T0.vector()[:] = np.exp(-xi*x.vector()[:])

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
    
    FT = (T - T_) / dt * U * df.dx \
        + T.dx(0) * U * df.dx \
        + kappa_eff * T.dx(0) * U.dx(0) * df.dx \
        + (kappa * k**2 + Gamma - (2*kappa_par*xi + 1) * psi * xi * T0) * T * U * df.dx \
        - betamT0 * xi * T0 * ( kappa_par * k**2 * p - (2*kappa_par*xi + 1) * p.dx(0) ) * U * df.dx

    Fp = + betamT0 * p.dx(0) * q.dx(0) * df.dx \
        + betamT0 * k**2 * p * q * df.dx \
        + psi * T.dx(0) * q * df.dx
        # - onoff * q * ds(1)

    F = FT + Fp

    # define boundary conditions
    
    bc_T_l = df.DirichletBC(W.sub(0), 0., subd, 1) # apply T = 0 to the left boundary
    bc_T_r = df.DirichletBC(W.sub(0), 0., subd, 2) # apply T = 0 to the right boundary
    bc_p_l = df.DirichletBC(W.sub(1), onoff, subd, 1) # apply p = onoff to the right boundary
    # bc_p_r = df.DirichletBC(W.sub(1), 0., subd, 2) # apply p = 0 to the right boundary

    bcs = [bc_T_l, bc_p_l] #, bc_T_r, bc_p_r]

    # split equation in left and right member
    a, L = df.lhs(F), df.rhs(F)

    #xdmff_T = df.XDMFFile(mesh.mpi_comm(), "T.xdmf")
    #xdmff_p = df.XDMFFile(mesh.mpi_comm(), "p.xdmf")

    plot = True
        
    data = []
    
    t = 0.0
    it = 0
        
    fig, axp = plt.subplots(1, 2, figsize=(15, 5))
    fig, axT = plt.subplots(1, 2, figsize=(15, 5))
        
    while t < tmax:
        print('t = ', t)
        #print('before solving: w_.vector() = ', w_.vector()[:])
        it += 1

        if t > tpert:
            onoff.assign(0.)

        df.solve(a == L, w_, bcs=bcs)
            
        t += dt
            
        #print('after solving: w_.vector() = ', w_.vector()[:])
        T__, p__ = w_.split(deepcopy=True)
            
        #print(f'T__.vector() = {T__.vector()[:]}')
        #print(f'p__.vector() = {p__.vector()[:]}')
            
        #T__.rename("T", "T")
        #p__.rename("p", "p")
        #xdmff_T.write(T__, t)
        #xdmff_p.write(p__, t)

        Tmax = np.max(T__.vector()[:])
        pmax = np.max(p__.vector()[:])
        data.append([t, Tmax, pmax]) # collect the (t, Tmax, pmax) values

        if (it % plot_intv == 0 and plot):
            color = cmap(t / tmax)
            axp[0].plot(x.vector()[:], p__.vector()[:], label=f"$t={t:1.2f}$", color=color) # plot p vs x
            axp[1].plot(x.vector()[:], p__.vector()[:]/pmax, label=f"$t={t:1.2f}$", color=color) # plot p/pmax vs x
            # ax[1].plot(x.vector()[:], p__.vector()[:]/pmax)
            #axp[0].set_xlim(0, 20)
            #axp[1].set_xlim(0, 20)
            
            axT[0].plot(x.vector()[:], T__.vector()[:], label=f"$t={t:1.2f}$", color=color) # plot T vs x
            axT[1].plot(x.vector()[:], T__.vector()[:]/Tmax, label=f"$t={t:1.2f}$", color=color) # plot T/Tmax vs x
            #axT[0].set_xlim(0, 20)
            #axT[1].set_xlim(0, 20)
        

    data = np.array(data)
    n_steps = len(data[:, 0])
    istart = int(tmax/dt)//2
    print("istart = ", istart)

    # find the growth rate gamma
    popt, pcov = np.polyfit(data[istart:, 0], np.log(data[istart:, 1]), 1, cov=True)
    gamma = popt[0]
    gamma_variance = pcov[0, 0]
    gamma_standard_error = np.sqrt(gamma_variance)
    
    print(f"gamma = {gamma}")
    print(f"gamma_standard_error = {gamma_standard_error}")
    
    # write the gamma value in the related file
    Pe_str = f"Pe_{Pe:.10g}"
    Gamma_str = f"Gamma_{Gamma:.10g}"
    beta_str = f"beta_{beta:.10g}"
    output_folder = f"results/outppt_" + "_".join([Pe_str, Gamma_str, beta_str]) + "/"
    os.makedirs(output_folder, exist_ok=True)
    #with open(output_folder + "gamma_linear.txt", "a") as file:
    #    file.write(f"\n{kk}\t{gamma}\t{gamma_standard_error}")
        
    # xdmff_T.close()
    # xdmff_p.close()
    
    #exit(0)
    
    # Plot results
    
    fig_, ax_ = plt.subplots(1, 1)
    
    ax_.plot(data[:, 0], data[:, 1]) # plot Tmax vs time
    #ax_.plot(data[:, 0], data[:, 2], label=r"$p_{\rm max}$")  # plot pmax vs time
    ax_.plot(data[istart:, 0], 0.001*np.exp(gamma*data[istart:, 0]), label=r"fit") # plot fitting line
    ax_.semilogy()
    ax_.set_xlabel(r"$t$")
    ax_.set_ylabel(r"$T_{\rm max}$")
    ax_.legend()
    
    kk = 2
    G = gamma + kk*kappa + Gamma
    Lambda_k = (-1 + np.sqrt(1 + 4*G*kappa_eff) )/(2*kappa_eff)
    
    #axp[1].plot(x.vector()[:], [x_ * np.exp(- Lambda_k * x_) for x_ in x.vector()[:]], label="$x*e^{-xii*x}$", color='black')
    axp[1].plot(x.vector()[:], [100*np.exp(- Lambda_k * x_) for x_ in x.vector()[:]], label="$100*e^{-\Lambda*x}$", color='black', linestyle='dashed')
    axp[1].plot(x.vector()[:], [50*(np.exp(- Lambda_k * x_) + np.exp(- kk * x_)) for x_ in x.vector()[:]], label="$50*(e^{-\Lambda*x}+e^{-k*x})$", color='black', linestyle='dotted')
    
    #axT[1].plot(x.vector()[:], [x_ * np.exp(- Lambda_k * x_) for x_ in x.vector()[:]], label="$x*e^{-xii*x}$", color='black')
    #axT[1].plot(x.vector()[:], [100*np.exp(- Lambda_k * x_) for x_ in x.vector()[:]], label="$100*e^{-\Lambda*x}$", color='black', linestyle='dashed')
    #axT[1].plot(x.vector()[:], [50*(np.exp(- Lambda_k * x_) + np.exp(- kk * x_)) for x_ in x.vector()[:]], label="$50*(e^{-\Lambda*x}+e^{-k*x})$", color='black', linestyle='dotted')

    [axpi.set_xlabel(r"$x$") for axpi in axp]
    [axTi.set_xlabel(r"$x$") for axTi in axT]
        
    axp[0].set_ylabel(r"$p_k(x)$")
    axp[1].set_ylabel(r"$p_k(x)/pmax$")
    axT[1].set_ylabel(r"$T_k(x)/Tmax$")
    
    axp[0].set_title("p(x)")
    axp[1].set_title("p(x)/p_max")
    axT[1].set_title("T(x)/T_max")
    
    axp[1].semilogy()
    axT[1].semilogy()
    axp[0].legend()
    axT[0].legend(fontsize=8)
    
    plt.show()
