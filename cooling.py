from mpi4py import MPI
import argparse
import os
import time
import math

import ufl
import dolfinx as dfx
from dolfinx_mpc import LinearProblem, MultiPointConstraint
import numpy as np
from dolfinx.mesh import create_unit_square, CellType, locate_entities_boundary
from basix.ufl import element
from utils import get_next_subfolder, mpi_comm, mpi_rank, mpi_size


parser = argparse.ArgumentParser(description='Process some parameters.')

# Define the command-line arguments
parser.add_argument('--Pe', default=100, type=float, help='Value for Peclet number')
parser.add_argument('--Gamma', default=1, type=float, help='Value for heat transfer ratio')
parser.add_argument('--beta', default=1e-3, type=float, help='Value for viscosity ratio')
parser.add_argument('--ueps', default=0.001, type=float, help='Value for amplitude of the perturbation')
parser.add_argument('--Ly', default=2, type=float, help='Value for wavelength')
parser.add_argument('--Lx', default=50, type=float, help='Value for system size')
parser.add_argument('--dt', default=0.005, type=float, help='Value for time interval')
parser.add_argument('--t_pert', default=0.1, type=float, help='Value for perturbation time')
parser.add_argument('--t_end', default=20, type=float, help='Value for final time')
#parser.add_argument('ny', type=float, help='Value for tile density along y')
#parser.add_argument('rtol', type=float, help='Value for error function')
parser.add_argument('--rnd',action='store_true', help='Flag for random velocity at inlet')
parser.add_argument('--holdpert',action='store_true', help='Flag for maintaining the perturbation at all times')
parser.add_argument('--constDeltaP',action='store_true', help='Flag for imposing constant pressure, instead of constant flow rate, at the inlet boundary')

# Parse the command-line arguments
args = parser.parse_args()

tol = 1e-7
Lx = args.Lx # x-lenght of domain (system size)
Ly = args.Ly # y-lenght of domain (wavelength)

# inlet at x = 0
def inlet_boundary(x):
    return np.isclose(x[0], 0, atol=tol)

# outlet at x = Lx
def outlet_boundary(x):
    return np.isclose(x[0], Lx, atol=tol)
    
# wall at y = Ly
def periodic_boundary(x):
    return np.isclose(x[1], Ly, atol=tol)

# periodic relation ( f(x,y) = f(x,y - Ly) )
def periodic_relation(x):
    out_x = np.zeros_like(x)
    out_x[0] = x[0]
    out_x[1] = x[1] - Ly
    return out_x
    
# Generate Gaussian noise on a given mesh. (scale: The scale (standard deviation) of the Gaussian noise.)
#def generate_gaussian_noise(mesh, scale=1.0):
    #dim = mesh.geometry().dim()
   #noise = scale * np.random.randn(mesh.num_vertices()) # Generate random Gaussian noise
    #V = ufl.FunctionSpace(mesh, 'CG', 1) # Define a FunctionSpace for the noise
    #eta = ufl.Expression('scale * noise', degree=1, scale=scale, noise=noise, element=V.ufl_element()) # Create a UFL Expression representing the Gaussian noise field
 #def eta(x, sigma=1):
    #return sigma * np.random.randn()
    
def create_script(elapsed_time, Nx, Ny, rtol, dt, dump_intv, t_end, t_pert, out_dir):
    script = f"""Elapsed Time: {elapsed_time} seconds
    Nx = {Nx}, Ny = {Ny}
    rtol = {rtol}

    dt = {dt}
    dump_intv = {dump_intv}
    t_pert = {t_pert}
    t_end = {t_end}
    """
    with open(out_dir + "notes.txt", 'w') as f:
        f.write(script)


start_time = time.time()

if __name__ == "__main__":
    
    nx = 2 # tile density along x
    ny = 50 # tile density along y
    #ny = args.ny # tile density along y
    
    Nx = int(nx*Lx) # number of tiles along x ( = Number of divisions along the x-axis)
    Ny = int(ny*Ly) # number of tiles along y ( = Number of divisions along the y-axis. Proportional to the length Ly to be coherent in resolution between simulations with different Ly)
    
    # Global parameters
    Pe = args.Pe # Peclet number
    Gamma = args.Gamma # Heat transfer ratio
    beta = args.beta # Viscosity ratio ( nu(T) = beta^(-T) )
    
    # Inlet parameters
    ueps = args.ueps # amplitude of the perturbation
    u0 = 1.0 # base inlet velocity
    
    # Base state parameters
    Deff = 1./Pe + 2*Pe*u0*u0/105 # effective constant diffusion for the base state
    xi = (- u0 + math.sqrt(u0*u0 + 4*Deff*Gamma)) / (2*Deff) # decay constant for the base state
    
    # Flags
    rnd = args.rnd
    holdpert = args.holdpert
    constDeltaP = args.constDeltaP
    
    dt = args.dt # time interval (default value: 0.005)
    t = 0. # starting time
    t_pert = args.t_pert # perturbation time (default value: 0.1)
    t_end = args.t_end # final time (default value: 50.01)
    dump_intv = 10 # saving interval
    
    rtol = 1e-14 # tolerance for solving linear problem
    #rtol = args.rtol # tolerance for solving linear problem
    
    # Generate mesh
    def mesh_warp_x(x): # function for non-constant length of grid along x
        x0 = 0.6 # percentage of tiles after which you change function for length
        y0 = 0.2 # parameter to adjust
        ids_less = x < x0
        ids_more = np.logical_not(ids_less)
        x_out = np.zeros_like(x)
        x_out[ids_less] = (y0/x0) * x[ids_less]
        x_out[ids_more] = (1.-y0)/(1.-x0) * (x[ids_more]-1) + 1
        return x_out
    def mesh_warp_y(x): # function for non-constant length of grid along y
        x0 = 0.75 # percentage of tiles after which you change function for length
        y0 = 0.375 # parameter to adjust
        ids_less = x < x0
        ids_more = np.logical_not(ids_less)
        x_out = np.zeros_like(x)
        x_out[ids_less] = (y0/x0) * x[ids_less]
        x_out[ids_more] = (1.-y0)/(1.-x0) * (x[ids_more]-1) + 1
        return x_out
    alpha = 2
    #mesh = create_unit_square(MPI.COMM_WORLD, Nx, Ny, diagonal=dfx.cpp.mesh.DiagonalType.right)
    mesh = create_unit_square(MPI.COMM_WORLD, Nx, Ny, cell_type=CellType.quadrilateral, diagonal=dfx.cpp.mesh.DiagonalType.left)
    mesh.geometry.x[:, 0] = mesh_warp_x(mesh.geometry.x[:, 0]) * Lx
    #mesh.geometry.x[:, 0] *= Lx
    mesh.geometry.x[:, 1] *= Ly
    
    # Define the finite element function spaces
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    v_cg0 = element("DG", mesh.topology.cell_name(), 0, shape=(mesh.geometry.dim, ))
    S = dfx.fem.functionspace(mesh, s_cg1)
    V = dfx.fem.functionspace(mesh, v_cg0)
    #S = dfx.fem.functionspace(mesh, ("Lagrange", 1))
    #V = dfx.fem.FunctionSpace(mesh, ufl.VectorElement("DG", "quadrilateral", 0))
    # V = dfx.fem.FunctionSpace(mesh, ufl.VectorElement("DG", "triangle", 0))
    x = ufl.SpatialCoordinate(mesh)

    # Create Dirichlet boundary condition
    inlet_facets = locate_entities_boundary(mesh, 1, inlet_boundary)
    outlet_facets = locate_entities_boundary(mesh, 1, outlet_boundary)
    
    inlet_dofs = dfx.fem.locate_dofs_topological(S, 1, inlet_facets)
    outlet_dofs = dfx.fem.locate_dofs_topological(S, 1, outlet_facets)

    bc_T_inlet = dfx.fem.dirichletbc(1., inlet_dofs, S) # T = 1 at x = 0
    bcs_T = [bc_T_inlet] # Dirichlet boundary condition for T problem

    bc_p_outlet = dfx.fem.dirichletbc(0., outlet_dofs, S) # P = 0 at x = L
    if (constDeltaP == True):
        
        # Get DOF coordinates
        dof_coords = S.tabulate_dof_coordinates()
        
        # Check the shape of dof_coords
        print(f"Shape of dof_coords: {dof_coords.shape}")
        
        # If it's a 1D array, reshape it to 2D (ensure correct number of columns for the mesh geometry)
        if dof_coords.ndim == 1:
            dof_coords = dof_coords.reshape(-1, mesh.geometry.dim)
            
        # Now dof_coords should be reshaped correctly
        print(f"Reshaped dof_coords: {dof_coords.shape}")
        
        # Compute boundary values at x=0
        bc_values = np.array([1.0 + ueps*np.sin(2.0 * np.pi * coord[1] / Ly) for coord in dof_coords[inlet_dofs]])
        
        # Now, we will create a Function in the space `S` to store the boundary condition values
        bc_constant = dfx.fem.Constant(mesh, bc_values[0])
        
        # Define Dirichlet BC using NumPy array
        bc_p_inlet = dfx.fem.dirichletbc(bc_constant, inlet_dofs, S) # P = 1. + (2*pi/lambda)*sin(y) at x = 0.
        bcs_p = [bc_p_inlet, bc_p_outlet]
        
        bc_p_inlet_2 = dfx.fem.dirichletbc(1., inlet_dofs, S) # P = 1. at x = 0.
        bcs_p2 = [bc_p_inlet_2, bc_p_outlet]
    else:
        bcs_p = [bc_p_outlet] # Dirichlet boundary condition for Darcy problem
        bcs_p2 = [bc_p_outlet]

    # Create periodic boundary conditions
    mpc_T = MultiPointConstraint(S)
    mpc_T.create_periodic_constraint_geometrical(S, periodic_boundary, periodic_relation, bcs_T)
    mpc_T.finalize()

    mpc_p = MultiPointConstraint(S)
    mpc_p.create_periodic_constraint_geometrical(S, periodic_boundary, periodic_relation, bcs_p)
    mpc_p.finalize()

    mpc_u = MultiPointConstraint(V)
    mpc_u.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation, [])
    mpc_u.finalize()

    # Define the domain of the surface integral
    facet_indices = dfx.mesh.locate_entities(mesh, mesh.topology.dim-1, inlet_boundary)
    facet_markers = np.full_like(facet_indices, 1)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = dfx.mesh.meshtags(mesh, mesh.topology.dim-1, facet_indices[sorted_facets], facet_markers[sorted_facets])
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

    # Define trial and test functions
    T = ufl.TrialFunction(S) # scalar trial function (for temperature and pressure)
    b = ufl.TestFunction(S) # scalar test function

    u = ufl.TrialFunction(V) # vectorial trial function (for velocity)
    v = ufl.TestFunction(V) # vectorial test function

    # Define physical functions
    T_ = dfx.fem.Function(mpc_T.function_space, name="T") # temperature
    T_n = dfx.fem.Function(mpc_T.function_space, name="T") # temperature at previous step
    T_n.interpolate(lambda x: np.exp(-xi*x[0]))
    mob_ = beta**-T_n # mobility

    p_ = dfx.fem.Function(mpc_p.function_space, name="p") # pressure (before: p = dfx.fem.Function(S, name="p"))
    u_ = - mob_ * ufl.grad(p_) # velocity
    
    # velocity at inlet with perturbation
    ux0 = dfx.fem.Function(mpc_p.function_space, name="ux0")
    if rnd is False:
        #ux0 = u0 + ueps*ufl.sin(2*ufl.pi*x[1]/Ly) # velocity at inlet (before: u0 = ufl.as_vector((1.0 + 1.0*ufl.sin(2*ufl.pi*x[1]), 0.)))
        ux0 = u0 + ueps*ufl.sin(2*ufl.pi*x[1]/Ly)
    else:
        ux0.x.array[:] = u0 + ueps*np.random.randn()
    
    # velocity at inlet without perturbation
    ux0_2 = dfx.fem.Function(mpc_p.function_space, name="ux0_2")
    ux0_2.x.array[:] = u0

    
    # Problem for p during perturbation
    # constant inlet flow rate: nabla u = nabla ( - mob * nabla P) = 0, u = ux0 at x = 0
    # constant pressure drop: nabla u = nabla ( - mob * nabla P) = 0
    if (constDeltaP == True):
        F_p = mob_ * ufl.dot(ufl.grad(T), ufl.grad(b)) * ufl.dx - dfx.fem.Constant(mesh, 0.0) * b * ufl.dx
    else:
        F_p = mob_ * ufl.dot(ufl.grad(T), ufl.grad(b)) * ufl.dx - ux0 * b * ds(1)
    a_p = ufl.lhs(F_p)
    L_p = ufl.rhs(F_p)
    
    problem_p = LinearProblem(a_p, L_p, mpc_p, u=p_, bcs=bcs_p,
                              petsc_options={"ksp_type": "cg", "ksp_rtol": rtol, "pc_type": "hypre", "pc_hypre_type": "boomeramg",
                                             "pc_hypre_boomeramg_max_iter": 1, "pc_hypre_boomeramg_cycle_type": "v",
                                             "pc_hypre_boomeramg_print_statistics": 0})
    
    # Problem for p after perturbation (nabla u = nabla ( -(k/mu(T)) nabla P) = 0, u = ux0_2 at x = 0)
    if (constDeltaP == True):
        F_p2 = mob_ * ufl.dot(ufl.grad(T), ufl.grad(b)) * ufl.dx - dfx.fem.Constant(mesh, 0.0) * b * ufl.dx
    else:
        F_p2 = mob_ * ufl.dot(ufl.grad(T), ufl.grad(b)) * ufl.dx - ux0_2 * b * ds(1)
    a_p2 = ufl.lhs(F_p2)
    L_p2 = ufl.rhs(F_p2)
    
    problem_p2 = LinearProblem(a_p2, L_p2, mpc_p, u=p_, bcs=bcs_p2,
                              petsc_options={"ksp_type": "cg", "ksp_rtol": rtol, "pc_type": "hypre", "pc_hypre_type": "boomeramg",
                                             "pc_hypre_boomeramg_max_iter": 1, "pc_hypre_boomeramg_cycle_type": "v",
                                             "pc_hypre_boomeramg_print_statistics": 0})
    # Problem for T
    F_T = (T - T_n)*b/dt * ufl.dx # dT/dt
    F_T += ufl.dot(u_, ufl.grad(T)) * b * ufl.dx # u \cdot \nabla T
    F_T += (1./Pe) * ufl.inner(ufl.grad(T), ufl.grad(b)) * ufl.dx # -(1/Pe) \nabla^2 T
    F_T += (2.*Pe/105) * ufl.inner(u_, ufl.grad(b)) * ufl.inner(u_, ufl.grad(T)) * ufl.dx # -(2 Pe/105) \nabla (u \prod u) \nabla T
    F_T += 1.*Gamma * T * b * ufl.dx # Gamma * T
    
    a_T = ufl.lhs(F_T)
    L_T = ufl.rhs(F_T)
    
    problem_T = LinearProblem(a_T, L_T, mpc_T, bcs=bcs_T,
                              petsc_options={"ksp_type": "bcgs", "ksp_rtol": rtol, "pc_type": "jacobi"})
    
    # Project u for visualization (only used for steps multiple of dump_intv)
    #F_u = ufl.dot(u - u_, v) * ufl.dx
    
    #a_u = ufl.lhs(F_u)
    #L_u = ufl.rhs(F_u)
    
    #problem_u = LinearProblem(a_u, L_u, mpc_u, bcs=[])
    
    # Prepare files for saving
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
    
    if (constDeltaP == True):
        out_dir = "results/constDeltaP_"
    else:
        out_dir = "results/"
    out_dir += "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str, holdpert_str]) + "/" # directoty for output
    xdmff_T = dfx.io.XDMFFile(mesh.comm, out_dir + "T.xdmf", "w")
    xdmff_p = dfx.io.XDMFFile(mesh.comm, out_dir + "p.xdmf", "w")
    # xdmff_u = dfx.io.XDMFFile(mesh.comm, out_dir + "u.xdmf", "w")

    xdmff_T.write_mesh(mesh)
    xdmff_p.write_mesh(mesh)
    # xdmff_u.write_mesh(mesh)

    it = 0 # iterative step
    
    while t < t_end:
        if mpi_rank == 0:
            print(f"it={it} t={t}")
        
        # Solve problem for p
        if holdpert is True:
            p_h = problem_p.solve()
        else:
            if t < t_pert:
                p_h = problem_p.solve() # solve problem with perturbation
            else:
                # ux0.x.array[:] = u0
                p_h = problem_p2.solve() # solve problem without perturbation
        
        # Update p (u will update consequently)
        p_.x.array[:] = p_h.x.array[:]
        p_.x.scatter_forward()
        
        # Solve problem for T
        T_h = problem_T.solve()

        # Update T (mob will update consequently)
        T_n.x.array[:] = T_h.x.array
        T_n.x.scatter_forward()

        if it % dump_intv == 0: # print only for steps multiple of dump_intv
            xdmff_T.write_function(T_n, t)
            xdmff_p.write_function(p_, t)

            # Project u for visualization
            #u_h = problem_u.solve()
            #u_h.name = "u"
            #xdmff_u.write_function(u_h, t)

        t += dt
        it += 1
        
    xdmff_T.close()
    xdmff_p.close()
    #xdmff_u.close()

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_string = f"Elapsed Time: {elapsed_time} seconds"
print(elapsed_time_string)

create_script(elapsed_time, Nx, Ny, rtol, dt, dump_intv, t_end, t_pert, out_dir)
