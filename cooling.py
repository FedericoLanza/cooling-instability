import os, sys
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import dolfinx as dfx
import gmsh
import math
import numpy as np
import time
import ufl

from dolfinx.io import gmshio
from dolfinx.mesh import create_unit_square, CellType, locate_entities_boundary, meshtags
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from basix.ufl import element
from mpi4py import MPI
from pathlib import Path
from petsc4py import PETSc
from utils import get_next_subfolder, mpi_comm, mpi_rank, mpi_size

comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser(description='Process some parameters.')

# Define the command-line arguments
parser.add_argument('--Pe', default=1000, type=float, help='Value for Peclet number')
parser.add_argument('--Gamma', default=1e-7, type=float, help='Value for heat transfer ratio')
parser.add_argument('--beta', default=1e-3, type=float, help='Value for viscosity ratio')
parser.add_argument('--ueps', default=1e-3, type=float, help='Value for amplitude of the perturbation')
parser.add_argument('--Ly', default=13000000, type=float, help='Value for wavelength')
parser.add_argument('--Lx', default=50000000, type=float, help='Value for system size')
#parser.add_argument('--Ly', default=1, type=float, help='Value for wavelength')
#parser.add_argument('--Lx', default=1, type=float, help='Value for system size')
parser.add_argument('--dt', default=20000, type=float, help='Value for time interval')
parser.add_argument('--t_pert', default=20000, type=float, help='Value for perturbation time')
parser.add_argument('--t_end', default=4e07, type=float, help='Value for final time')
parser.add_argument('--ny', default=300, type=float, help='Value for tile density along y')
parser.add_argument('--nx', default=600, type=float, help='Value for tile density along x')
#parser.add_argument('rtol', type=float, help='Value for error function')
parser.add_argument('--rnd',action='store_true', help='Flag for random velocity at inlet')
parser.add_argument('--holdpert',action='store_true', help='Flag for holding perturbation')
parser.add_argument('--Tpert',action='store_true', help='Flag for perturbing T, instead of u, at the inlet')
parser.add_argument('--constDeltaP',action='store_true', help='Flag for imposing constant pressure, instead of constant flow rate, at the inlet boundary')
parser.add_argument('--twoperiods',action='store_true', help='Flag for performing simulations with 2 wavelenghts per spatial period')

# Parse the command-line arguments
args = parser.parse_args()

tol = 1e-7
Lx = args.Lx # x-lenght of domain (system size)
Ly = args.Ly # y-lenght of domain (wavelength)
    
def create_script(elapsed_time, Nx, Ny, rtol, dt, dump_intv, t_end, t_pert, out_dir):
    out_dir = Path(out_dir)              # works if you pass str or Path
    if MPI.COMM_WORLD.rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        notes_path = out_dir / "notes.txt"
        script = (
            f"Elapsed Time: {elapsed_time} seconds\n"
            f"Nx = {Nx}, Ny = {Ny}\n"
            f"rtol = {rtol}\n\n"
            f"dt = {dt}\n"
            f"dump_intv = {dump_intv}\n"
            f"t_pert = {t_pert}\n"
            f"t_end = {t_end}\n"
        )
        with notes_path.open("w", encoding="utf-8") as f:
            f.write(script)
    MPI.COMM_WORLD.barrier()
    
def global_count(n_local: int) -> int:
    return MPI.COMM_WORLD.allreduce(n_local, op=MPI.SUM)

def global_count_int(n_local: int) -> int:
    return MPI.COMM_WORLD.allreduce(int(n_local), op=MPI.SUM)

def global_minmax(vec):
    loc_min = float(vec.min()) if vec.size else float("inf")
    loc_max = float(vec.max()) if vec.size else float("-inf")
    comm = MPI.COMM_WORLD
    return comm.allreduce(loc_min, op=MPI.MIN), comm.allreduce(loc_max, op=MPI.MAX)

def global_minmax_array(arr):
    loc_min = float(arr.min()) if arr.size else float("inf")
    loc_max = float(arr.max()) if arr.size else float("-inf")
    comm = MPI.COMM_WORLD
    return comm.allreduce(loc_min, op=MPI.MIN), comm.allreduce(loc_max, op=MPI.MAX)

def global_minmax_on_dofs(f, dofs):
    if dofs.size > 0:
        loc_min = float(f.x.array[dofs].min())
        loc_max = float(f.x.array[dofs].max())
    else:
        loc_min, loc_max = float("inf"), float("-inf")
    gmin = MPI.COMM_WORLD.allreduce(loc_min, op=MPI.MIN)
    gmax = MPI.COMM_WORLD.allreduce(loc_max, op=MPI.MAX)
    return gmin, gmax

def G(expr, tag):
    val = dfx.fem.assemble_scalar(dfx.fem.form(expr * ds(tag)))
    return MPI.COMM_WORLD.allreduce(float(val), op=MPI.SUM)

def amp_of_t(t, amp, t_pert, hold_on=False, ramp_dt=None):
    if hold_on:
        return amp
    if ramp_dt is None:
        # hard switch
        return amp if t < t_pert else 0.0
    # 1-step triangular ramp (optional)
    if t < ramp_dt:
        return amp * (t / ramp_dt)
    if t < t_pert:
        return amp
    if t < t_pert + ramp_dt:
        return amp * (1.0 - (t - t_pert) / ramp_dt)
    return 0.0

def build_random_profile_on_y(Ly, M=6, seed=12345):
    # root creates random coeffs
    if comm.rank == 0:
        rng    = np.random.default_rng(seed)
        amps   = rng.normal(size=M)
        phases = rng.uniform(0.0, 2*np.pi, size=M)
    else:
        amps = phases = None
    amps   = comm.bcast(amps, root=0)
    phases = comm.bcast(phases, root=0)

    # return a numpy-evaluable function for fem.Function.interpolate
    denom = float(np.sqrt((amps**2).sum()) + 1e-15)
    def f(x):
        yy = x[1]                       # shape (N,)
        s = np.zeros_like(yy)
        for m, (a, phi) in enumerate(zip(amps, phases), start=1):
            s += a * np.cos(2*np.pi*m*y/Ly + phi)
        s /= denom
        return 1.0 + float(Teps_c.value) * s          # centered around 1
    return f

start_time = time.time()

if __name__ == "__main__":
    
    nx = args.nx # tile density along x
    ny = args.ny # tile density along y
    lc = 0.05  # target mesh size (smaller = finer mesh)
    
    Nx = int(nx) # number of tiles along x ( = Number of divisions along the x-axis)
    Ny = int(ny) # number of tiles along y ( = Number of divisions along the y-axis. Proportional to the length Ly to be coherent in resolution between simulations with different Ly)
    
    # Global parameters
    Pe = args.Pe # Peclet number
    Gamma = args.Gamma # Heat transfer ratio
    beta = args.beta # Viscosity ratio ( nu(T) = beta^(-T) )
    
    # Inlet parameters
    ueps = args.ueps # amplitude of the perturbation
    u0 = 1.0 # base inlet velocity
    
    # Base state parameters
    kappa_eff = 1./Pe + 2*Pe*u0*u0/105 # effective constant diffusion for the base state
    xi = (- u0 + math.sqrt(u0*u0 + 4*kappa_eff*Gamma)) / (2*kappa_eff) # decay constant for the base state
    
    # Flags
    rnd = args.rnd
    holdpert = args.holdpert
    Tpert = args.Tpert
    constDeltaP = args.constDeltaP
    twoperiods = args.twoperiods
    
    dt = args.dt # time interval (default value: 0.005)
    t = 0. # starting time
    t_pert = args.t_pert # perturbation time (default value: 0.1)
    t_end = args.t_end # final time (default value: 50.01)
    dump_intv = 10 # saving interval
    
    rtol = 1e-12 # tolerance for solving linear problem
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
        
    def mesh_warp_y(y): # function for non-constant length of grid along y
        w = 0.5 # fraction of interval with higher tile density
        slope_less = 3./5 # has to be in the range (1/2, 1]
        slope_more = (1. - w)/(1 - w/slope_less)
        limit = w/slope_less # percentage of tiles after which you change function for length
        ids_less = y < limit
        ids_more = np.logical_not(ids_less)
        y_out = np.zeros_like(y)
        y_out[ids_less] = slope_less * y[ids_less]
        y_out[ids_more] = slope_more * (y[ids_more] - limit) + w
        return y_out
        
    def mesh_warp_y_2(y): # function for non-constant length of grid along y
        w = 0.25 # fraction of interval with higher tile density
        f1 = 0.125
        slope_less = 0.3
        slope_more = (1. - w)/(1 - w/slope_less) # 4.5
        limit1 = f1/slope_more # percentage of tiles after which you change function for length
        limit2 = w/slope_less
        ids_more_1 = y < limit1
        ids_less = np.logical_and(y >= limit1, y < limit1 + limit2)
        ids_more_2 = y >= limit1 + limit2
        y_out = np.zeros_like(y)
        y_out[ids_more_1] = slope_more * y[ids_more_1]
        y_out[ids_less] = slope_less * (y[ids_less] - limit1) + f1
        y_out[ids_more_2] = slope_more * (y[ids_more_2] - limit2) + w
        return y_out
    
    tiles = 'rectangle'
    
    if tiles == 'rectangle':
        # Structured rectangular mesh (unit square stretched)
        mesh = create_unit_square(MPI.COMM_WORLD, Nx, Ny, cell_type=CellType.quadrilateral)
        
        mesh.geometry.x[:, 0] = mesh_warp_x(mesh.geometry.x[:, 0]) * Lx
        #mesh.geometry.x[:, 0] *= Lx
        if (twoperiods or rnd):
            mesh.geometry.x[:, 1] *= Ly
        else:
            mesh.geometry.x[:, 1] *= Ly
            #mesh.geometry.x[:, 1] = mesh_warp_y_2(mesh.geometry.x[:, 1]) * Ly
            
        # Boundary marker functions
        def inlet_boundary(x):   return np.isclose(x[0], 0, atol=tol) # inlet at x = 0
        def outlet_boundary(x):  return np.isclose(x[0], Lx, atol=tol) # outlet at x = Lx
        def bottom_boundary(x):  return np.isclose(x[1], 0, atol=tol) # bottom at y = 0
        def top_boundary(x):     return np.isclose(x[1], Ly, atol=tol) # top at y = Ly
        
        # Build connectivity before tagging & locating
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        mesh.topology.create_connectivity(tdim - 1, 0)
        mesh.topology.create_connectivity(0, tdim - 1)
        
        # Mark facets
        inlet_facets = locate_entities_boundary(mesh, 1, inlet_boundary)
        outlet_facets = locate_entities_boundary(mesh, 1, outlet_boundary)
        bottom_facets = locate_entities_boundary(mesh, 1, bottom_boundary)
        top_facets = locate_entities_boundary(mesh, 1, top_boundary)

        facets = np.hstack([inlet_facets, outlet_facets, bottom_facets, top_facets])
        values = np.hstack([np.full_like(inlet_facets, 1),
                        np.full_like(outlet_facets, 2),
                        np.full_like(bottom_facets, 3),
                        np.full_like(top_facets, 4)])
        sorted_facets = np.argsort(facets)
        facet_tags = meshtags(mesh, 1, facets[sorted_facets], values[sorted_facets])
        
    elif tiles == 'triangle':

        gmsh.initialize()
        gmsh.model.add("rect_domain") #"rectangle"

        # Geometry: rectangle [0, Lx] x [0, Ly]
        rect = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
        gmsh.model.occ.synchronize()
    
        # Mesh options: unstructured triangular mesh with good quality
        gmsh.option.setNumber("Mesh.Algorithm", 6)       # Delaunay (robust, isotropic)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)    # no quads, only triangles
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
        gmsh.model.mesh.generate(2) # Generate mesh
        
        # Boundaries
        inlet = gmsh.model.getEntitiesInBoundingBox(0, 0, 0, 0, Ly, 0, 1)
        outlet = gmsh.model.getEntitiesInBoundingBox(Lx, 0, 0, Lx, Ly, 0, 1)
        bottom = gmsh.model.getEntitiesInBoundingBox(0, 0, 0, Lx, 0, 0, 1)
        top = gmsh.model.getEntitiesInBoundingBox(0, Ly, 0, Lx, Ly, 0, 1)
        
        gmsh.model.addPhysicalGroup(1, [e[1] for e in inlet], 1)
        gmsh.model.addPhysicalGroup(1, [e[1] for e in outlet], 2)
        gmsh.model.addPhysicalGroup(1, [e[1] for e in bottom], 3)
        gmsh.model.addPhysicalGroup(1, [e[1] for e in top], 4)
        gmsh.model.addPhysicalGroup(2, [rect], 5)
        gmsh.model.occ.synchronize()

        # Import into FEniCSx
        mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        mesh.name = "rectangle"
        
        gmsh.finalize()
    
    # Define the finite element function spaces
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    v_cg0 = element("DG", mesh.topology.cell_name(), 0, shape=(mesh.geometry.dim, ))
    S = dfx.fem.functionspace(mesh, s_cg1)
    V = dfx.fem.functionspace(mesh, v_cg0)
    
    x = ufl.SpatialCoordinate(mesh)
    y = x[1]
    
    # Map them to dofs
    inlet_facets  = facet_tags.find(1)
    outlet_facets = facet_tags.find(2)
    #epsx = max(1e-12*Lx, 100*np.finfo(float).eps * Lx)  # robust band near x=0 and x=Lx
    inlet_dofs  = dfx.fem.locate_dofs_geometrical(S, lambda x: np.isclose(x[0], 0.0, atol=tol))
    outlet_dofs = dfx.fem.locate_dofs_geometrical(S, lambda x: np.isclose(x[0], Lx,  atol=tol))
    
    T_inlet_fun = dfx.fem.Function(S, name="Tinlet") # Function that will hold the (time-dependent) inlet temperature values if Tpert == True

    # Create the Dirichlet BCs
    
    Teps_c = dfx.fem.Constant(mesh, PETSc.ScalarType(ueps))
    # temperature at inlet (with or without perturbation depending on Tpert)
    if Tpert:
        if rnd:
            rand_T_profile = build_random_profile_on_y(Ly, M=6, seed=42)
            Tinlet_fun.interpolate(rand_T_profile) # 1 + Teps * random(y)
        else:
            Tinlet_fun.interpolate(lambda x: 1.0 + Teps_c * np.cos(nmode*np.pi*(x[1]-Ly/2)/(Ly/2)))

        bc_T_inlet = dfx.fem.dirichletbc(Tinlet_fun, inlet_dofs) # function-valued BC
    else:
        # no T-pert: plain T=1 at inlet
        bc_T_inlet = dfx.fem.dirichletbc(PETSc.ScalarType(1.0), inlet_dofs, S)

    bcs_T = [bc_T_inlet] # Dirichlet boundary condition for T problem
    bc_p_outlet = dfx.fem.dirichletbc(PETSc.ScalarType(0.0), outlet_dofs, S) # P = 0 at outlet
    
    if (constDeltaP == True): # to be fixed
        
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
        bc_values = np.array([1.0 + ueps*np.cos(2.0 * np.pi * (coord[1] - Ly/2) / Ly) for coord in dof_coords[inlet_dofs]])
        
        # Now, we will create a Function in the space `S` to store the boundary condition values
        bc_constant = dfx.fem.Constant(mesh, bc_values[0])
        
        # Define Dirichlet BC using NumPy array
        bc_p_inlet = dfx.fem.dirichletbc(bc_constant, inlet_dofs, S) # P = 1. + (2*pi/lambda)*cos(y) at x = 0.
        bcs_p = [bc_p_inlet, bc_p_outlet]
        
        bc_p_inlet_2 = dfx.fem.dirichletbc(1., inlet_dofs, S) # P = 1. at x = 0.
        bcs_p2 = [bc_p_inlet_2, bc_p_outlet]
    else:
        bcs_p = [bc_p_outlet] # Dirichlet boundary condition for Darcy problem
        bcs_p2 = [bc_p_outlet]

    # wall at y = Ly
    def periodic_boundary(x):
        on_top = np.isclose(x[1], Ly, atol=tol)
        not_corner = ~np.isclose(x[0], 0.0, atol=tol) & ~np.isclose(x[0], Lx, atol=tol)
        return on_top & not_corner

    # periodic relation ( f(x,y) = f(x,y - Ly) )
    def periodic_relation(x_in):
        out_x = np.zeros_like(x_in)
        out_x[0] = x_in[0]
        out_x[1] = x_in[1] - Ly
        return out_x
        
    # Build periodic constraints
    mpc_T = MultiPointConstraint(S)
    mpc_T.create_periodic_constraint_geometrical(S, periodic_boundary, periodic_relation, bcs_T)
    mpc_T.finalize()

    mpc_p = MultiPointConstraint(S)
    #mpc_p.create_periodic_constraint_geometrical(S, periodic_boundary, periodic_relation, bcs_p)
    mpc_p.finalize()
    
    mpc_u = MultiPointConstraint(V)
    mpc_u.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation, [])
    mpc_u.finalize()
    
    # Define the domain of the surface integral
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags, metadata={"quadrature_degree": 6})

    # Define trial and test functions
    T = ufl.TrialFunction(S) # scalar trial function for temperature
    b = ufl.TestFunction(S) # scalar test function for temperature
    
    p = ufl.TrialFunction(S) # scalar trial function for pressure
    q = ufl.TestFunction(S) # scalar test function for pressure

    u = ufl.TrialFunction(V) # vectorial trial function (for velocity)
    v = ufl.TestFunction(V) # vectorial test function

    # Define physical functions
    T_ = dfx.fem.Function(mpc_T.function_space, name="T") # temperature
    T_n = dfx.fem.Function(mpc_T.function_space, name="T") # temperature at previous step
    T_n.interpolate(lambda x: np.exp(-xi*x[0]))
    
    assert np.isfinite(T_n.x.array).all()
    Tg_min, Tg_max   = global_minmax_array(T_n.x.array)
    Tin_min, Tin_max = global_minmax_on_dofs(T_n, inlet_dofs)
    
    if MPI.COMM_WORLD.rank == 0:
        print("T_n range:", Tg_min, Tg_max)
        print("T_n on inlet (min/max):", Tin_min, Tin_max)
    
    ln_beta = dfx.fem.Constant(mesh, PETSc.ScalarType(np.log(beta)))
    arg = ufl.max_value(ufl.min_value(-ln_beta * T_n, 50.0), -50.0) # cap exponent to avoid overflow/underflow in exp
    mob_raw = ufl.exp(arg)

    # clamp mobility to a reasonable range
    mob_min = dfx.fem.Constant(mesh, PETSc.ScalarType(1e-8))
    mob_max = dfx.fem.Constant(mesh, PETSc.ScalarType(1e+8))
    mob_ = ufl.min_value(ufl.max_value(mob_raw, mob_min), mob_max)
    #mob_ = dfx.fem.Constant(mesh, PETSc.ScalarType(1.0))

    # --- DEBUG: true min/max at DG0 points (NOT integrals) ---
    Q0 = dfx.fem.functionspace(mesh, ("DG", 0))   # <-- factory, not fem.FunctionSpace

    uy0 = dfx.fem.Function(Q0)
    um0 = dfx.fem.Function(Q0)

    # --- Interpolate UFL expression to DG0 and get real min/max ---
    mob_dg0 = dfx.fem.Function(Q0)
    mob_expr = dfx.fem.Expression(mob_, Q0.element.interpolation_points())  # mob_ is your UFL expr
    mob_dg0.interpolate(mob_expr)

    local_min = float(mob_dg0.x.array.min())
    local_max = float(mob_dg0.x.array.max())
    global_min = MPI.COMM_WORLD.allreduce(local_min, op=MPI.MIN)
    global_max = MPI.COMM_WORLD.allreduce(local_max, op=MPI.MAX)

    if MPI.COMM_WORLD.rank == 0:
        print("mobility range (DG0):", global_min, global_max)

    p_ = dfx.fem.Function(mpc_p.function_space, name="p") # pressure (before: p = dfx.fem.Function(S, name="p"))
    u_ = - mob_ * ufl.grad(p_) # velocity
    
    # velocity at inlet (with or without perturbation depending on Tpert)
    u0_c   = dfx.fem.Constant(mesh, PETSc.ScalarType(u0))
    ueps_c = dfx.fem.Constant(mesh, PETSc.ScalarType(ueps))
    
    if Tpert:
        #profile_U = PETSc.ScalarType(0.0) # won’t be used
        ux0_expr = u0_c
    else:
        if rnd:
            # random SCALAR speed, MPI-safe and rampable via ueps_c
            if comm.rank == 0:
                rng = np.random.default_rng(4242)
                xi  = rng.standard_normal()
            else:
                xi = None
            xi = comm.bcast(xi, root=0)
            rand_val = PETSc.ScalarType(xi)
            ux0_expr  = u0_c + ueps_c * rand_val
        else:
            cosy = ufl.cos(2.0 * ufl.pi * (y - Ly/2) / Ly) if (not twoperiods) else ufl.cos(4.0 * ufl.pi * (y - Ly/2) / Ly)
            ux0_expr  = u0_c + ueps_c * cosy # velocity at inlet

            #val = u0 + ueps * np.random.randn() if MPI.COMM_WORLD.rank == 0 else None
            #val = MPI.COMM_WORLD.bcast(val, root=0)
            #ux0_expr = fem.Constant(mesh, PETSc.ScalarType(val))
        
    n = ufl.FacetNormal(mesh)
    g_in = -ux0_expr
    
    # Problem for p
    # constant inlet flow rate: nabla u = nabla ( - mob * nabla P) = 0, u = ux0 at x = 0
    # constant pressure drop: nabla u = nabla ( - mob * nabla P) = 0
    if (constDeltaP == True):
        F_p = mob_ * ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx - dfx.fem.Constant(mesh, 0.0) * q * ufl.dx
    else:
        F_p = mob_ * ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx + g_in * q * ds(1)
    a_p, L_p = ufl.lhs(F_p), ufl.rhs(F_p)
    
    petsc_p = {
    "ksp_type": "cg",
    "ksp_rtol": rtol,
    "ksp_atol": 1e-14,
    "ksp_max_it": 1000,
    "ksp_error_if_not_converged": True,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_strong_threshold": 0.6,
    }
    
    problem_p = LinearProblem(a_p, L_p, mpc_p, u=p_, bcs=bcs_p, petsc_options=petsc_p)
    
    # Problem for T (old version)
    #F_T = (T - T_n)*b/dt * ufl.dx # dT/dt
    #F_T += ufl.dot(u_, ufl.grad(T)) * b * ufl.dx # u \cdot \nabla T
    #F_T += (1./Pe) * ufl.inner(ufl.grad(T), ufl.grad(b)) * ufl.dx # -(1/Pe) \nabla^2 T
    #F_T += (2.*Pe/105) * ufl.inner(u_, ufl.grad(b)) * ufl.inner(u_, ufl.grad(T)) * ufl.dx # -(2 Pe/105) \nabla (u \prod u) \nabla T
    #F_T += 1.*Gamma * T * b * ufl.dx # Gamma * T
    
    alpha = dfx.fem.Constant(mesh, PETSc.ScalarType(0.1))  # ramp to 1.0
    dtC   = dfx.fem.Constant(mesh, PETSc.ScalarType(dt/5))   # small startup dt
    Gamma_c = dfx.fem.Constant(mesh, PETSc.ScalarType(Gamma))
    
    # --- diffusion & reaction (unchanged) ---
    F_T  = (T - T_n)*b/dtC * ufl.dx # dT/dt
    F_T += (1./Pe) * ufl.inner(ufl.grad(T), ufl.grad(b)) * ufl.dx # -(1/Pe) \nabla^2 T
    F_T += Gamma_c * T * b * ufl.dx # Gamma * T
    F_T += 0.5 * ( ufl.dot(u_, ufl.grad(T)) * b - ufl.dot(u_, ufl.grad(b)) * T ) * ufl.dx # u \cdot \nabla T (skew-symmetric advection)
    
    Kcross = alpha * (2.0*Pe/105.0) * ufl.outer(u_, u_)  # 2x2 tensor
    vKgradT = ufl.dot(Kcross, ufl.grad(T)) # matrix–vector product
    F_T += ufl.dot(vKgradT, ufl.grad(b)) * ufl.dx # Add cross diffusion in bilinear form: ∫ (K ∇T)·∇b dx
    
    # --- SUPG stabilization ---
    hK      = ufl.CellDiameter(mesh)
    kappa   = 1.0/Pe
    umag    = ufl.sqrt(1e-30 + ufl.dot(u_, u_))
    Ck = 9.
    Cg = 1. # Cg = 1 works fine for Gamma = 1e-7
    tau_SUPG = 1.0 / ufl.sqrt( (2*umag/hK)**2 + (Ck*kappa/hK**2)**2 + (Cg*Gamma_c)**2 )
    tau_SUPG = ufl.min_value(tau_SUPG, hK/(2*umag + 1e-30))

    R_T = (T - T_n)/dtC \
         + ufl.dot(u_, ufl.grad(T)) \
         - kappa * ufl.div(ufl.grad(T)) \
         - ufl.div(vKgradT) \
         + Gamma_c * T
         
    F_T += tau_SUPG * ufl.dot(u_, ufl.grad(b)) * R_T * ufl.dx
    
    # Shock-capturing viscosity (Codina/Kuzmin-style, simple isotropic version)
    Csc   = 0.2  # try 0.1–0.5
    epsG  = 1e-12
    gradTn = ufl.grad(T_n)
    Gmag_n  = ufl.sqrt(ufl.inner(gradTn, gradTn) + epsG)

    nu_sc = Csc * hK * umag * (Gmag_n / (Gmag_n + epsG))

    # Add to the diffusion operator
    F_T += nu_sc * ufl.inner(ufl.grad(T), ufl.grad(b)) * ufl.dx
    
    a_T = ufl.lhs(F_T)
    L_T = ufl.rhs(F_T)
    
    petsc_T = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-10,
    "ksp_atol": 1e-12,
    "ksp_max_it": 1000,
    "ksp_error_if_not_converged": True,

    # Make GMRES numerically safer
    "ksp_norm_type": "unpreconditioned",
    "ksp_gmres_restart": 50,                      # shorter cycles
    "ksp_gmres_modifiedgramschmidt": None,        # stable orthogonalization
    "ksp_gmres_cgs_refinement_type": "refine_ifneeded",
    "ksp_check_norm_iteration": 1,                # verify true residual each it
    #"ksp_monitor_true_residual": None,          # optional during debugging
    # "ksp_view" : None,

    # Robust PC
    "pc_type": "asm",
    "pc_asm_overlap": 2,
    "sub_ksp_type": "preonly",
    "sub_pc_type": "ilu",
    "sub_pc_factor_levels": 1,
    #"pc_type": "hypre",
    #"pc_hypre_type": "boomeramg",
    #"pc_hypre_boomeramg_coarsen_type": "HMIS",
    #"pc_hypre_boomeramg_interp_type":  "ext+i",
    #"pc_hypre_boomeramg_relax_type_all": "symmetric-SOR/Jacobi",
    #"pc_hypre_boomeramg_strong_threshold": 0.5, # 0.5 works fine for Gamma = 1e-7
    #"pc_hypre_boomeramg_agg_nl": 1, # Gamma = 1e-7 works even without this
    #"pc_hypre_boomeramg_agg_num_paths": 2, # Gamma = 1e-7 works even without this
    }
    
    problem_T = LinearProblem(a_T, L_T, mpc_T, bcs=bcs_T, petsc_options=petsc_T)
    
    # Project u for visualization (only used for steps multiple of dump_intv)
    #F_u = ufl.dot(u - u_, v) * ufl.dx
    
    #a_u = ufl.lhs(F_u)
    #L_u = ufl.rhs(F_u)
    
    #problem_u = LinearProblem(a_u, L_u, mpc_u, bcs=[])
    
    # Prepare files for saving
    Pe_str = f"Pe_{Pe:.10g}"
    Gamma_str = f"Gamma_{Gamma:.10g}"
    beta_str = f"beta_{beta:.10g}"
    Ly_str = f"Ly_{Ly:.10g}"
    Lx_str = f"Lx_{Lx:.10g}"
    dt_str = f"dt_{dt:.10g}"
    ny_str = f"ny_{ny:.10g}"
    nx_str = f"nx_{nx:.10g}"
    #rtol_str = f"rtol_{rtol:.10g}"
    rnd_str = f"rnd_{rnd}"
    holdpert_str = f"holdpert_{holdpert}"
    Tpert_str = f"Tpert_{Tpert}"
    
    if (constDeltaP == True):
        out_dir = "results/constDeltaP_"
    else:
        out_dir = "results/"
    out_dir += "_".join([Pe_str, Gamma_str, beta_str, Ly_str, Lx_str, dt_str, ny_str, nx_str, rnd_str, Tpert_str, holdpert_str]) # directory for output
    if twoperiods:
        out_dir += "_twoperiods"
    out_dir += "_triangularmesh/"
    
    # 1) Make sure the directory exists (rank 0), then sync
    out_dir = Path(out_dir)  # your string -> Path
    if MPI.COMM_WORLD.rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    MPI.COMM_WORLD.barrier()
    
    # 2) Build the filename safely
    fname_T = out_dir / "T.xdmf"
    fname_p = out_dir / "p.xdmf"
    
    # 3) Use a context manager so the file is definitely closed
    with dfx.io.XDMFFile(MPI.COMM_WORLD, str(fname_T), "w") as xdmf_T, \
        dfx.io.XDMFFile(MPI.COMM_WORLD, str(fname_p), "w") as xdmf_p:
        
        xdmf_T.write_mesh(mesh)
        xdmf_p.write_mesh(mesh)
    
    #xdmff_T = dfx.io.XDMFFile(mesh.comm, str(fname_T), "w")
    #xdmff_p = dfx.io.XDMFFile(mesh.comm, str(fname_p), "w")
    # xdmff_u = dfx.io.XDMFFile(mesh.comm, out_dir + "u.xdmf", "w")

    #xdmff_T.write_mesh(mesh)
    #xdmff_p.write_mesh(mesh)
    #xdmff_u.write_mesh(mesh)

    ninlet  = global_count_int(inlet_facets.size)
    noutlet = global_count_int(outlet_facets.size)
    ndofs_in  = global_count_int(inlet_dofs.size)
    ndofs_out = global_count_int(outlet_dofs.size)
    
    if MPI.COMM_WORLD.rank == 0:
        print("inlet facets:", ninlet, "outlet facets:", noutlet)
        print("inlet dofs:", ndofs_in, "outlet dofs:", ndofs_out)

    it = 0 # iterative step
    ramp_dt = None
    
    #alpha.value = PETSc.ScalarType(0.25)
    
    while t < t_end:
        if mpi_rank == 0:
            print(f"it={it} t={t}")
        
        # Control perturbation
        if (not holdpert) and (t > t_pert):
            if Tpert:
                # set back to exactly 1 on the inlet
                Tinlet_fun.x.array.fill(1.0)
                Tinlet_fun.x.scatter_forward()
                Teps_c.value = PETSc.ScalarType(0.0) # (If you want to ramp down: multiply Teps over time and re-interpolate)

            else:
                # revert to base: U(y) = u0
                ueps_c.value = PETSc.ScalarType(0.0)
        
        # Solve problem for p
        t0 = MPI.Wtime()
        p_h = problem_p.solve() # solve p-problem
        mpc_p.backsubstitution(p_h)
        p_h.x.scatter_forward()

        # Explicit NaN/Inf check for p
        if not np.isfinite(p_h.x.array).all():
            raise RuntimeError(f"NaN/Inf in p at step {it}, t={t}")
        
        # Update p (u will update consequently)
        p_.x.array[:] = p_h.x.array[:]
        p_.x.scatter_forward()
        
        # Solve problem for T
        t1 = MPI.Wtime()
        T_h = problem_T.solve()
        mpc_T.backsubstitution(T_h)
        T_h.x.scatter_forward()
        t2 = MPI.Wtime()
        
        if (MPI.COMM_WORLD.rank == 0 and it < 5):
            print(f"p-solve: {t1-t0:.3f}s  T-solve: {t2-t1:.3f}s", flush=True)
        
        # Explicit NaN/Inf check for T
        if not np.isfinite(T_h.x.array).all():
            raise RuntimeError(f"DG0NaN/Inf in T at step {it}, t={t}")
        
        # Update T (mob will update consequently)
        T_h.x.array[:] = np.clip(T_h.x.array, 0.0, 1.0)
        T_n.x.array[:] = T_h.x.array
        T_n.x.scatter_forward()

        if it % dump_intv == 0: # print only for steps multiple of dump_intv
                with dfx.io.XDMFFile(MPI.COMM_WORLD, str(fname_T), "a") as xdmf_T:
                    xdmf_T.write_function(T_n, t)
                with dfx.io.XDMFFile(MPI.COMM_WORLD, str(fname_p), "a") as xdmf_p:
                    xdmf_p.write_function(p_, t)

            # Project u for visualization
            #u_h = problem_u.solve()
            #u_h.name = "u"
            #xdmff_u.write_function(u_h, t)
        
        if (it % 100 == 0):
            #print("mob bounds:", dfx.fem.assemble_scalar(dfx.fem.form(ufl.min_value(mob_, mob_max)*ufl.dx)), dfx.fem.assemble_scalar(dfx.fem.form(ufl.max_value(mob_, mob_min)*ufl.dx)))
            t3 = MPI.Wtime()
            pmin, pmax = global_minmax(p_.x.array)
            Tmin, Tmax = global_minmax(T_n.x.array)
            Tin_min, Tin_max = global_minmax_on_dofs(T_h, inlet_dofs)
                    
            # Local extrema
            loc_min_uy = float(uy0.x.array.min())
            loc_max_uy = float(uy0.x.array.max())

            # Use a physical floor (e.g., 0.5–1% of u0 works well)
            denom = np.maximum(um0.x.array, 1e-2*abs(u0))
            loc_max_ratio = float((np.abs(uy0.x.array)/denom).max())

            # Global reductions
            mn_uy  = MPI.COMM_WORLD.allreduce(loc_min_uy,   op=MPI.MIN)
            mx_uy  = MPI.COMM_WORLD.allreduce(loc_max_uy,   op=MPI.MAX)
            mx_ratio = MPI.COMM_WORLD.allreduce(loc_max_ratio, op=MPI.MAX)

            # Boundary L2 norm (also collective)
            area = dfx.fem.assemble_scalar(dfx.fem.form(1.0 * ds(1)))
            area = MPI.COMM_WORLD.allreduce(float(area), op=MPI.SUM)
            inlet_l2 = float("nan")
            if area > 0:
                err2 = dfx.fem.assemble_scalar(dfx.fem.form((T_h - 1.0)**2 * ds(1)))
                err2 = MPI.COMM_WORLD.allreduce(float(err2), op=MPI.SUM)
                inlet_l2 = (err2/area)**0.5
            
            u_vec = -mob_ * ufl.grad(p_)

            # Lengths (sanity)
            Lin, Lout, Ltop, Lbot = (G(1.0, 1), G(1.0, 2), G(1.0, 4), G(1.0, 3))
            if MPI.COMM_WORLD.rank == 0:
                print(f"lengths: inlet={Lin:.6e} outlet={Lout:.6e} top={Ltop:.6e} bot={Lbot:.6e}")

            # Outward normal fluxes u·n on each side
            Qn_in   = G(ufl.dot(u_vec, n), 1)
            Qn_out  = G(ufl.dot(u_vec, n), 2)
            Qn_top  = G(ufl.dot(u_vec, n), 4)
            Qn_bot  = G(ufl.dot(u_vec, n), 3)
            MB_ext = Qn_in + Qn_out
            rel_MB = abs(MB_ext) / abs(Qn_in)

            # For intuition (not a flux): ∫ u_x ds
            Qx_in   = G(u_vec[0], 1)
            Qx_out  = G(u_vec[0], 2)
            
            # Inlet target integral  (this is what the PDE is asked to match weakly)
            Ubar_in = G(ux0_expr, 1)
            #target_in = dfx.fem.assemble_scalar(dfx.fem.form((u0 + ueps*ufl.sin(2*ufl.pi*y/Ly)) * ds(1)))
            #rel_target = abs(Qn_in + target_in)/abs(target_in)
            
            if MPI.COMM_WORLD.rank == 0:
            
                print("p range:", pmin, pmax)
                print("T_n range:", Tmin, Tmax)
                print("T on inlet (min/max):", Tin_min, Tin_max)
                print(f"u_y range (DG0): {mn_uy:.3e} … {mx_uy:.3e}")
                print(f"max |u_y|/|u|: {mx_ratio:.3e}")
                print("||T-1||_L2(inlet):", inlet_l2)
                
                print(f"outward fluxes: in={Qn_in:.6e}, out={Qn_out:.6e}, top={Qn_top:.6e}, bot={Qn_bot:.6e}")
                print(f"sum(u·n) all sides = {(Qn_in+Qn_out+Qn_top+Qn_bot):.6e} (should be ~ 0)")
                print(f"inlet:  ∫u·n ds ≈ -∫U ds ⇒ {Qn_in:.6e} vs {-Ubar_in:.6e}")
                print(f"∫u_x ds: inlet={Qx_in:.6e}, outlet={Qx_out:.6e}  (both should be ≈ +∫U ds)")
                print("rel ext mass err:", rel_MB)
                #print("rel inlet err vs discrete target:", rel_target)
                
            Q0 = dfx.fem.functionspace(mesh, ("DG", 0))
            dpdx = dfx.fem.Function(Q0)
            dpdx.interpolate(dfx.fem.Expression(ufl.grad(p_)[0], Q0.element.interpolation_points()))
            mn_dpdx = MPI.COMM_WORLD.allreduce(float(dpdx.x.array.min()), op=MPI.MIN)
            mx_dpdx = MPI.COMM_WORLD.allreduce(float(dpdx.x.array.max()), op=MPI.MAX)
            
            if MPI.COMM_WORLD.rank == 0:
                print(f"dp/dx range (DG0): {mn_dpdx:.3e} … {mx_dpdx:.3e} (should be ≤ 0 almost everywhere)")
            t4 = MPI.Wtime()
            if (MPI.COMM_WORLD.rank == 0 and it < 5):
                print(f"Checks: {t4-t3:.3f}s", flush=True)
            if MPI.COMM_WORLD.rank == 0:
                print(f"dp/dx range (DG0): {mn_dpdx:.3e} … {mx_dpdx:.3e} (should be ≤ 0 almost everywhere)")
                print(f" ")
            
        # ramp alpha
        if it in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            alpha.value = PETSc.ScalarType(min(1.0, float(alpha.value) + 0.1))

        # ramp dt
        if it == 10:
            dtC.value = PETSc.ScalarType(dt)

        # (optional) one-time Hypre tuning just before ramp starts
        if False:
            ksp = problem_T._solver
            pc  = ksp.getPC()
            opts = PETSc.Options()
            opts["pc_hypre_boomeramg_coarsen_type"] = "HMIS"
            opts["pc_hypre_boomeramg_interp_type"]  = "ext+i"
            opts["pc_hypre_boomeramg_strong_threshold"] = 0.5
            pc.setFromOptions(); ksp.setFromOptions(); ksp.setUp()

        t += float(dtC.value)
        it += 1
        
    #xdmff_T.close()
    #xdmff_p.close()
    #xdmff_u.close()

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_string = f"Elapsed Time: {elapsed_time} seconds"
print(elapsed_time_string)

create_script(elapsed_time, Nx, Ny, rtol, dt, dump_intv, t_end, t_pert, out_dir)
