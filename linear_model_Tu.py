import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import argparse
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
    parser.add_argument("-Pe", default=100.0, type=float, help="Peclet number")
    parser.add_argument("-k", default=0.0, type=float, help="Wavelength")
    parser.add_argument("-Gamma", default=1.0, type=float, help="Heat conductivity")
    parser.add_argument("-beta", default=0.1, type=float, help="Viscosity ratio")
    parser.add_argument("-eps", default=1e-3, type=float, help="Perturbation amplide")
    parser.add_argument("-tpert", default=0.1, type=float, help="Perturbation duration")
    parser.add_argument("-dt", default=0.01, type=float, help="Timestep")
    parser.add_argument("-nx", default=1000, type=int, help="Number of mesh points")
    parser.add_argument("-Lx", default=50.0, type=float, help="System size")
    parser.add_argument("-tmax", default=10.0, type=float, help="Total time")
    return parser.parse_args()

if __name__ == "__main__":

    cmap = plt.cm.viridis
    
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

    plot_intv = 100

    #k = df.Constant(2*np.pi/lam)
    k = df.Constant(kk)
    
    kappa = 1/Pe
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
    bc_T_r = df.DirichletBC(W.sub(0), 0., subd, 2) # apply T = 0 to the right boundary
    bc_u_l = df.DirichletBC(W.sub(1), onoff, subd, 1) # apply u = onoff to the left boundary

    bcs = [bc_T_l, bc_u_l]

    # split equation in left and right member
    a, L = df.lhs(F), df.rhs(F)
        
    data = []
        
    t = 0.0
    it = 0
        
    fig, axu = plt.subplots(1, 2, figsize=(15, 5))
    fig, axT = plt.subplots(1, 2, figsize=(15, 5))
        
    while t < tmax:
        print('t = ', t)
        #print('before solving: w_.vector() = ', w_.vector()[:])
        it += 1

        if t > tpert:
            onoff.assign(0.)

        df.solve(a == L, w_, bcs=bcs)
        
        t += dt

        T__, u__ = w_.split(deepcopy=True)

        Tmax = np.max(T__.vector()[:])
        umax = np.max(u__.vector()[:])
        data.append([t, Tmax, umax]) # collect the (t, Tmax, pmax) values

        if (it % plot_intv == 0):
            color = cmap(t / tmax)
            axu[0].plot(x.vector()[:], u__.vector()[:], label=f"$t={t:1.2f}$", color=color) # plot ux vs x
            axu[1].plot(x.vector()[:], u__.vector()[:]/umax, label=f"$t={t:1.2f}$", color=color) # plot ux/ux_max vs x
            axu[0].set_xlim(0, 20)
            axu[1].set_xlim(0, 20)
            
            axT[0].plot(x.vector()[:], T__.vector()[:], label=f"$t={t:1.2f}$", color=color) # plot T vs x
            axT[1].plot(x.vector()[:], T__.vector()[:]/Tmax, label=f"$t={t:1.2f}$", color=color) # plot T/Tmax vs x
            axT[0].set_xlim(0, 20)
            axT[1].set_xlim(0, 20)
        
    data = np.array(data)
    n_steps = len(data[:, 0])
    istart = int(tmax/dt)//2
    #print("istart = ", istart)

    # find the growth rate gamma
    popt, pcov = np.polyfit(data[istart:, 0], np.log(data[istart:, 1]), 1, cov=True)
    gamma = popt[0]
    gamma_variance = pcov[0, 0]
    gamma_standard_error = np.sqrt(gamma_variance)
    
    #print(f"gamma = {gamma}")
    #print(f"gamma_standard_error = {gamma_standard_error}")
    
    # write the gamma value in the related file
    Pe_str = f"Pe_{Pe:.10g}"
    Gamma_str = f"Gamma_{Gamma:.10g}"
    beta_str = f"beta_{beta:.10g}"
    output_folder = f"results/output_" + "_".join([Pe_str, Gamma_str, beta_str]) + "/"
    with open(output_folder + "gamma_linear.txt", "a") as file:
        file.write(f"\n{kk}\t{gamma}\t{gamma_standard_error}")
    
    exit(0)
    
    # Plot results
    
    fig_, ax_ = plt.subplots(1, 1)
    
    ax_.plot(data[:, 0], data[:, 1]) # plot Tmax vs time
    #ax_.plot(data[:, 0], data[:, 2], label=r"$p_{\rm max}$")  # plot pmax vs time
    ax_.plot(data[istart:, 0], np.exp(gamma*data[istart:, 0]), label=r"fit") # plot fitting line
    ax_.semilogy()
    ax_.set_xlabel(r"$t$")
    ax_.set_ylabel(r"$T_{\rm max}$")
    ax_.legend()
        
    G = gamma + k*kappa + Gamma
    Lambda_k = (-1 + np.sqrt(1 + 4*G*kappa_eff) )/(2*kappa_eff)
    print('Lambda_k = ',Lambda_k)
    
    axu[1].plot(x.vector()[:], [100 * np.exp(- Lambda_k * x_) for x_ in x.vector()[:]], label="$\sim e^{-\Lambda*x}$", color='black', linestyle='dashed')
    axu[1].plot(x.vector()[:], [50 * (np.exp(- Lambda_k * x_) + np.exp(- k * x_)) for x_ in x.vector()[:]], label="$\sim Ae^{-\Lambda*x} + Be^{-k*x}$", color='black', linestyle='dotted')
    #axu[1].plot(x.vector()[:], [2 * x_ * (np.exp(- Lambda_k * x_)) for x_ in x.vector()[:]], label="$2x*e^{-\Lambda*x}$", color='black', linestyle='dotted')
    
    axT[1].plot(x.vector()[:], [100 * np.exp(- Lambda_k * x_) for x_ in x.vector()[:]], label="$\sim e^{-\Lambda*x}$", color='black', linestyle='dashed')
    #axT[1].plot(x.vector()[:], [2 * x_ * np.exp(- Lambda_k * x_) for x_ in x.vector()[:]], label="2x*e^{-\Lambda*x}$", color='black', linestyle='dotted')

    [axui.set_xlabel(r"$x$") for axui in axu]
    [axTi.set_xlabel(r"$x$") for axTi in axT]
    
    axu[0].set_ylabel(r"$u_k(x)$")
    axu[1].set_ylabel(r"$u_k(x)/umax$")
    axT[0].set_ylabel(r"$T_k(x)$")
    axT[1].set_ylabel(r"$T_k(x)/Tmax$")
    
    axu[0].set_title("u(x)")
    axu[1].set_title("u(x)/u_max")
    axT[0].set_title("T(x)")
    axT[1].set_title("T(x)/T_max")
    
    axu[1].semilogy()
    axT[1].semilogy()
    axu[0].legend()
    axT[0].legend()
    
    plt.show()
