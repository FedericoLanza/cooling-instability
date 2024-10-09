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
    parser.add_argument("-lam", default=2, type=float, help="Wavelength")
    parser.add_argument("-Gamma", default=1.0, type=float, help="Heat conductivity")
    parser.add_argument("-beta", default=0.001, type=float, help="Viscosity ratio")
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
    tmax = args.tmax # 10.0

    Pe = args.Pe # 10**0.75 #100.0
    lam = args.lam # 2.0 # 4.0
    Gamma = args.Gamma #1.0
    beta = args.beta # 0.001
    eps = args.eps # 1e-1
    tpert = args.tpert # 0.1

    plot_intv = 100

    kappa = 1/Pe
    kappa_par = 2.0 / 105 * Pe # equivalent to (2.0 / 105) * Pe
    k = df.Constant(2*np.pi/lam)

    kappa_eff = kappa + kappa_par
    psi = -np.log(beta)
    xi = (- 1 + np.sqrt(1 + 4*kappa_eff*Gamma))/(2*kappa_eff)

    print('kappa_eff = ', kappa_eff, ', xi = ', xi, ', 1/xi = ',1.0/xi)

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

    onoff = df.Constant(1.0)

    # define equations in variational form
    
    FT = (T - T_) / dt * Q * df.dx \
        + T.dx(0) * Q * df.dx \
        + kappa_eff * T.dx(0) * Q.dx(0) * df.dx \
        + (kappa * k**2 + Gamma) * T * Q * df.dx \
        + ((- xi - 2*kappa_par * xi**2) * T0 * u + kappa_par * xi * T0 * u.dx(0) ) * Q * df.dx

    Fu = - u.dx(0) * v.dx(0) * df.dx \
        - onoff * eps * v * ds(1) \
        + psi * xi * T0 * u.dx(0) * v * df.dx \
        + k**2 * u * v * df.dx \
        + k**2 * psi * T * v * df.dx

    F = FT + Fu

    # define boundary conditions
    
    bc_T_l = df.DirichletBC(W.sub(0), 0., subd, 1) # apply T = 0 to the left boundary
    bc_T_r = df.DirichletBC(W.sub(0), 0., subd, 2) # apply T = 0 to the right boundary
    bc_p_r = df.DirichletBC(W.sub(1), 0., subd, 2) # apply p = 0 to the right boundary

    bcs = [bc_T_l] #, bc_T_r, bc_p_r]

    # split equation in left and right member
    a, L = df.lhs(F), df.rhs(F)

    #xdmff_T = df.XDMFFile(mesh.mpi_comm(), "T.xdmf")
    #xdmff_u = df.XDMFFile(mesh.mpi_comm(), "u.xdmf")

    plot = True

    #for lam_ in [pow(2,a) for a in np.arange(4., 8., 0.5)]:
    for k_ in np.arange(2, 2.25, 0.25):
        #k_ = 2*np.pi/lam_
        k.assign(k_)
        
        data = []
        
        t = 0.0
        it = 0
        
        print('before resetting: w_.vector() = ', w_.vector()[:])
        # Resetting all components of the mixed function to zero
        w_.vector().zero()
        print('after resetting: w_.vector() = ', w_.vector()[:])
        
        fig, axu = plt.subplots(1, 2, figsize=(15, 5))
        fig, axT = plt.subplots(1, 2, figsize=(15, 5))
        
        while t < tmax:
            print('t = ', t)
            print('before solving: w_.vector() = ', w_.vector()[:])
            it += 1

            if t > tpert:
                onoff.assign(0.)

            df.solve(a == L, w_, bcs=bcs)
            
            t += dt
            
            #print('after solving: w_.vector() = ', w_.vector()[:])

            T__, u__ = w_.split(deepcopy=True)
            
            print(f'T__.vector() = {T__.vector()[:]}')
            print(f'u__.vector() = {u__.vector()[:]}')
            
            #T__.rename("T", "T")
            #u__.rename("u", "u")
            #xdmff_T.write(T__, t)
            #xdmff_u.write(u__, t)

            Tmax = np.max(T__.vector()[:])
            umax = np.max(u__.vector()[:])
            data.append([t, Tmax, umax]) # collect the (t, Tmax, pmax) values

            if (it % plot_intv == 0 and plot):
                color = cmap(t / tmax)
                axu[0].plot(x.vector()[:], u__.vector()[:], label=f"$t={t:1.2f}$", color=color) # plot ux vs x
                axu[1].plot(x.vector()[:], u__.vector()[:]/umax, label=f"$t={t:1.2f}$", color=color) # plot ux/ux_max vs x
                # ax[1].plot(x.vector()[:], p__.vector()[:]/pmax)
                axu[0].set_xlim(0, 20)
                axu[1].set_xlim(0, 20)
            
                axT[0].plot(x.vector()[:], T__.vector()[:], label=f"$t={t:1.2f}$", color=color) # plot T vs x
                axT[1].plot(x.vector()[:], T__.vector()[:]/Tmax, label=f"$t={t:1.2f}$", color=color) # plot T/Tmax vs x
                axT[0].set_xlim(0, 20)
                axT[1].set_xlim(0, 20)
        

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
        output_folder = f"results/output_" + "_".join([Pe_str, Gamma_str, beta_str]) + "/"
        #with open(output_folder + "gamma_linear.txt", "a") as file:
        #    file.write(f"\n{k_}\t{gamma}\t{gamma_standard_error}")
        
    # xdmff_T.close()
    # xdmff_u.close()
    
    # exit(0)
    
    # Plot results
    
        fig_, ax_ = plt.subplots(1, 1)
    
        ax_.plot(data[:, 0], data[:, 1]) # plot Tmax vs time
        #ax_.plot(data[:, 0], data[:, 2], label=r"$p_{\rm max}$")  # plot pmax vs time
        ax_.plot(data[istart:, 0], np.exp(gamma*data[istart:, 0]), label=r"fit") # plot fitting line
        ax_.semilogy()
        ax_.set_xlabel(r"$t$")
        ax_.set_ylabel(r"$T_{\rm max}$")
        ax_.legend()
        
    G = gamma + 7.25*kappa + Gamma
    Lambda_k = (-1 + np.sqrt(1 + 4*G*kappa_eff) )/(2*kappa_eff)
    
    #axp[1].plot(x.vector()[:], [x_ * np.exp(- Lambda_k * x_) for x_ in x.vector()[:]], label="$x*e^{-xii*x}$", color='black')
    axu[1].plot(x.vector()[:], [100*np.exp(- Lambda_k * x_) for x_ in x.vector()[:]], label="$100*e^{-\Lambda*x}$", color='black', linestyle='dotted')
    
    #axT[1].plot(x.vector()[:], [x_ * np.exp(- Lambda_k * x_) for x_ in x.vector()[:]], label="$x*e^{-xii*x}$", color='black')
    axT[1].plot(x.vector()[:], [100*np.exp(- Lambda_k * x_) for x_ in x.vector()[:]], label="$100*e^{-\Lambda*x}$", color='black', linestyle='dotted')

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
    axu[1].legend()
    axT[1].legend()
    
    plt.show()
