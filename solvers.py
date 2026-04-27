import numpy as np
from contextlib import contextmanager
import time
from collections import namedtuple
from matplotlib import pyplot as plt
import jax

jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

@contextmanager
def measure_time(label):
    """Measure the time it takes to execute a block of code

    :arg label: label for the time measurement
    """
    t_start = time.perf_counter()
    try:
        yield
    finally:
        t_finish = time.perf_counter()
        t_elapsed = t_finish - t_start
        print(f"time [{label}] = {t_elapsed:8.2f} s")

def get_xizero(grid_spec, dtype=jnp.float64):
    """Construct the normalised frequency vectors

    Let k = (k_0,k_1,k_2) with k_d = 0,1,...,N_d-1 be a three-dimensional Fourier index.

    The normalised momentum vector is xi_d = 2 pi k_d / N_d, with 0 <= xi_0 < 2pi

    For a given k we then have that

    tilde(xi)_0 = 2/h_0 * sin(xi_0/2) * cos(xi_1/2) * cos(xi_2/2)
    tilde(xi)_1 = 2/h_1 * cos(xi_0/2) * sin(xi_1/2) * cos(xi_2/2)
    tilde(xi)_2 = 2/h_2 * cos(xi_0/2) * cos(xi_1/2) * sin(xi_2/2)

    This function returns a tensor of shape (3,N_0,N_1,N_2) which contains
    the normalised xi^0 = tilde(xi) / ||tilde(xi)|| for all Fourier modes.

     :arg grid_spec: namedtuple with grid specifications
     :arg dtype: data type

    """
    # Normalised momentum vectors in all three spatial directions
    K = [2 * np.pi * np.arange(n) / n for n in grid_spec.N]
    # Grid with normalised momentum vectors
    xi = np.meshgrid(*K, indexing="ij")
    h = grid_spec.h
    # Grid with tilde(xi)
    xi_tilde = np.stack(
        [
            2 / h[0] * np.sin(xi[0] / 2) * np.cos(xi[1] / 2) * np.cos(xi[2] / 2),
            2 / h[1] * np.cos(xi[0] / 2) * np.sin(xi[1] / 2) * np.cos(xi[2] / 2),
            2 / h[2] * np.cos(xi[0] / 2) * np.cos(xi[1] / 2) * np.sin(xi[2] / 2),
        ]
    )
    # Normalise tilde(xi) to obtain xi^0
    xi_nrm = np.linalg.norm(xi_tilde, axis=0)
    xi_nrm[xi_nrm < 1.0e-12] = 1  # avoid division by zero
    return (xi_tilde / xi_nrm).astype(dtype)


def initialise_material(grid_spec, shape="homogeneous", center=None, size=0.2, 
                        mu_mat=1, lmbda_mat=1, 
                        mu_inc=0.5, lmbda_inc=0.5, margin_width= 0.05,
                        add_crack=True, crack_length=0.5,
                        dtype=jnp.float64):
    """This function initialises the spatially varying Lame parameters for a given inclusion shape and material properties.

    
    :arg shape: shape of inclusion: "sphere", "cuboid", "cylinder"(Parallel to z-axis)or "homogeneous"
    :arg center: Tuple of (x,y,z) coordinates. Defaults to grid center.
    :arg size: Radius for sphere, or edge length(s) for cuboid.
    :arg mu_mat/lmbda_mat: Lamé parameters for the background matrix.
    :arg mu_inc/lmbda_inc: Lamé parameters for the inclusion (fibre).
    :arg margin_width: Width of the margin region near the left and right boundaries where the material is set to be very soft to avoid numerical instability.
    :arg add_crack: Whether to add a crack in the middle of the domain. The crack is represented as a region with zero stiffness.'  
    :arg crack_length: Length of the crack along the x direction. The crack is located in the middle of the domain along y and z directions.
    :arg dtype: data type for the output arrays
    :arg grid_spec: namedtuple with grid specifications
    
    """

    Lx = grid_spec.N[0] * grid_spec.h[0]
    Ly = grid_spec.N[1] * grid_spec.h[1]
    Lz = grid_spec.N[2] * grid_spec.h[2]

    # Find the center if one isnt inputted
    if center is None:
        cx, cy, cz = Lx / 2, Ly / 2, Lz / 2
    else:
        cx, cy, cz = center

    
    X, Y, Z = np.meshgrid(
        *[h * (1 / 2 + np.arange(n)) for (n, h) in zip(grid_spec.N, grid_spec.h)],
        indexing="ij",
    )
    
    
    if shape == "sphere":
        # The size acts as the radius
        idx = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2 < size**2
        
    elif shape == "cuboid":
        # Check if size is a single number (cube) or a list of 3 numbers (rectangle)
        if isinstance(size, (int, float)):
            sx, sy, sz = size, size, size
        else:
            sx, sy, sz = size
            
        idx = (np.abs(X - cx) < sx/2) & (np.abs(Y - cy) < sy/2) & (np.abs(Z - cz) < sz/2)

    elif shape == "cylinder":

        # Check if size is a single number (continuous) or a tuple (chopped short fiber)
        if isinstance(size, (int, float)):
            # Continuous fiber: We only restrict X and Y. It runs infinitely along Z.
            idx = (X - cx)**2 + (Y - cy)**2 < size**2
        else:
            # Chopped fiber: size = (radius, length)
            r, length = size
            idx = ((X - cx)**2 + (Y - cy)**2 < r**2) & (np.abs(Z - cz) < length/2)

    elif shape == "homogeneous":
        # An empty mask (no inclusion at all)
        idx = np.zeros_like(X, dtype=bool)
        
        
    else:
        raise ValueError(f"Unknown shape requested: {shape}")


  
    #True if inside use _inc else use _mat
    mu = np.where(idx, mu_inc, mu_mat)
    
    lmbda = np.where(idx, lmbda_inc, lmbda_mat)


    if margin_width is not None:
    
        margin_bound = (X < margin_width) | (X > (Lx - margin_width))
        
        # Override the Lame parameters with 0.0 in the margin regions
        mu = np.where(margin_bound, 0.0, mu)
        lmbda = np.where(margin_bound, 0.0, lmbda)
    
    if add_crack:
        
        crack_end_x = margin_width + crack_length
        
        h_y = grid_spec.h[1]
        crack_y_max = Ly / 2
        crack_y_min = (Ly / 2) - h_y 
        
        crack_bound = (X <= crack_end_x) & (Y > crack_y_min) & (Y < crack_y_max)
        
        # Override the Lame parameters with 0.0 in the crack region
        mu = np.where(crack_bound, 0.0, mu)
        lmbda = np.where(crack_bound, 0.0, lmbda)
        
    return jnp.array(mu, dtype=dtype), jnp.array(lmbda, dtype=dtype)
    
    


 

def backward_derivative(g, grid_spec, direction):
    """Compute the backward derivative in a specified direction

    :arg g: function to take the derivative of. Assumed to be of shape (*,N_0,N_1,N_2)
    :arg grid_spec: namedtuple with grid specification
    :arg direction: direction in which to take the derivative: 0, 1 or 2
    """
    if direction == 0:
        dg = (
            g
            + jnp.roll(g, 1, axis=-2)
            + jnp.roll(g, 1, axis=-1)
            + jnp.roll(g, (1, 1), axis=(-2, -1))
            - jnp.roll(g, 1, axis=-3)
            - jnp.roll(g, (1, 1), axis=(-3, -2))
            - jnp.roll(g, (1, 1), axis=(-3, -1))
            - jnp.roll(g, (1, 1, 1), axis=(-3, -2, -1))
        )
    elif direction == 1:
        dg = (
            g
            + jnp.roll(g, 1, axis=-3)
            + jnp.roll(g, 1, axis=-1)
            + jnp.roll(g, (1, 1), axis=(-3, -1))
            - jnp.roll(g, 1, axis=-2)
            - jnp.roll(g, (1, 1), axis=(-3, -2))
            - jnp.roll(g, (1, 1), axis=(-2, -1))
            - jnp.roll(g, (1, 1, 1), axis=(-3, -2, -1))
        )
    elif direction == 2:
        dg = (
            g
            + jnp.roll(g, 1, axis=-3)
            + jnp.roll(g, 1, axis=-2)
            + jnp.roll(g, (1, 1), axis=(-3, -2))
            - jnp.roll(g, 1, axis=-1)
            - jnp.roll(g, (1, 1), axis=(-3, -1))
            - jnp.roll(g, (1, 1), axis=(-2, -1))
            - jnp.roll(g, (1, 1, 1), axis=(-3, -2, -1))
        )

    return 1 / (4 * grid_spec.h[direction]) * dg


def backward_divergence(sigma, grid_spec):
    """Compute backward derivative of symmetric 3x3 tensor sigma_{ij}

    The componets of the tensor are assumed to be represented in vector form
    using Voigt notation:

    (sigma_{00}, sigma_{11}, sigma_{22}, sigma_{12}, sigma_{02}, sigma_{01})

    Returns a tensor of shape (3,N_0,N_1,N_2) with the three components of the
    divergence vector:

    [ dsigma_{00}/dx_0 + dsigma_{01}/dx_1 + dsigma_{02}/dx_2 ]
    [ dsigma_{10}/dx_0 + dsigma_{11}/dx_1 + dsigma_{12}/dx_2 ]
    [ dsigma_{20}/dx_0 + dsigma_{21}/dx_1 + dsigma_{22}/dx_2 ]

    :arg sigma: tensor representation using Voigt notation
    :arg grid_spec: namedtuple with grid specification
    """

    return jnp.stack(
        [
            backward_derivative(sigma[0, ...], grid_spec, 0)
            + backward_derivative(sigma[5, ...], grid_spec, 1)
            + backward_derivative(sigma[4, ...], grid_spec, 2),
            backward_derivative(sigma[5, ...], grid_spec, 0)
            + backward_derivative(sigma[1, ...], grid_spec, 1)
            + backward_derivative(sigma[3, ...], grid_spec, 2),
            backward_derivative(sigma[4, ...], grid_spec, 0)
            + backward_derivative(sigma[3, ...], grid_spec, 1)
            + backward_derivative(sigma[2, ...], grid_spec, 2),
        ]
    )


def fourier_solve(tau_hat, lmbda0, mu0, xizero):
    """Solve residual equation for reference material in Fourier space

    Computes hat(epsilon)_{kl} = -Gamma^0_{klij} hat(tau)_{ij}

    :arg tau_hat: The residual hat(tau) in Fourier space
    :arg lmbda0: coefficient lambda^0 of homogeneous reference material
    :arg mu0: coefficient mu^0 of homogeneous reference material
    :arg xizero: Normalised momentum vectors
    """
    epsilon_hat_A = jnp.stack(
        [
            xizero[0, ...] ** 2 * tau_hat[0, ...]
            + xizero[0, ...]
            * (xizero[2, ...] * tau_hat[4, ...] + xizero[1, ...] * tau_hat[5]),
            xizero[1, ...] ** 2 * tau_hat[1, ...]
            + xizero[1, ...]
            * (xizero[2, ...] * tau_hat[3, ...] + xizero[0, ...] * tau_hat[5]),
            xizero[2, ...] ** 2 * tau_hat[2, ...]
            + xizero[2, ...]
            * (xizero[1, ...] * tau_hat[3, ...] + xizero[0, ...] * tau_hat[4]),
            1
            / 2
            * (
                xizero[1, ...] * xizero[2, ...] * (tau_hat[1, ...] + tau_hat[2, ...])
                + (xizero[1, ...] ** 2 + xizero[2, ...] ** 2) * tau_hat[3, ...]
                + xizero[0, ...]
                * (xizero[1, ...] * tau_hat[4, ...] + xizero[2, ...] * tau_hat[5, ...])
            ),
            1
            / 2
            * (
                xizero[0, ...] * xizero[2, ...] * (tau_hat[0, ...] + tau_hat[2, ...])
                + (xizero[0, ...] ** 2 + xizero[2, ...] ** 2) * tau_hat[4, ...]
                + xizero[1, ...]
                * (xizero[0, ...] * tau_hat[3, ...] + xizero[2, ...] * tau_hat[5, ...])
            ),
            1
            / 2
            * (
                xizero[0, ...] * xizero[1, ...] * (tau_hat[0, ...] + tau_hat[1, ...])
                + (xizero[0, ...] ** 2 + xizero[1, ...] ** 2) * tau_hat[5, ...]
                + xizero[2, ...]
                * (xizero[0, ...] * tau_hat[3, ...] + xizero[1, ...] * tau_hat[4, ...])
            ),
        ]
    )
    Xi = jnp.stack(
        [
            xizero[0, ...] ** 2,
            xizero[1, ...] ** 2,
            xizero[2, ...] ** 2,
            xizero[1, ...] * xizero[2, ...],
            xizero[0, ...] * xizero[2, ...],
            xizero[0, ...] * xizero[1, ...],
        ]
    )
    Xi_dot_tau = (
        xizero[0, ...] ** 2 * tau_hat[0, ...]
        + xizero[1, ...] ** 2 * tau_hat[1, ...]
        + xizero[2, ...] ** 2 * tau_hat[2, ...]
        + 2
        * (
            xizero[1, ...] * xizero[2, ...] * tau_hat[3, ...]
            + xizero[0, ...] * xizero[2, ...] * tau_hat[4, ...]
            + xizero[0, ...] * xizero[1, ...] * tau_hat[5, ...]
        )
    )
    epsilon_hat_B = Xi * Xi_dot_tau
    return (
        1 / mu0 * (-epsilon_hat_A + (lmbda0 + mu0) / (lmbda0 + 2 * mu0) * epsilon_hat_B)
    )




def relative_divergence(sigma, grid_spec):
    """Compute ratio divergence div(sigma)_i = d sigma_{ij} / dx_j  and <||div(sigma)||^2>^{1/2}

    :arg sigma: stress
    :arg grid_spec: grid specification
    """
    dV = grid_spec.h[0] * grid_spec.h[1] * grid_spec.h[2]
    dsigma = backward_divergence(sigma, grid_spec)
    dsigma_nrm = jnp.sqrt(jnp.sum(dsigma**2) * dV)
    # Compute average <sigma> and norm ||<sigma>||
    sigma_avg = jnp.sum(sigma * dV, axis=[1, 2, 3])
    sigma_avg_nrm = jnp.sqrt(
        (jnp.sum(sigma_avg[:3] ** 2) + 2 * jnp.sum(sigma_avg[3:] ** 2))
    )
    return dsigma_nrm / sigma_avg_nrm


@jax.jit(static_argnames=["grid_spec", "tolerance", "depth"])
def lippmann_schwinger(lmbda, mu, E_mean, grid_spec, d=None,k=1e-12, tolerance=1e-6, depth=3, maxiter=1000):
    """Lippmann Schwinger iteration taking phase-field damage into account
   
    :arg lmbda: spatially varying Lame parameter lambda
    :arg mu: spatially varying Lame parameter lambda
    :arg E_mean: mean value of epsilon
    :arg grid_spec: grid specification as a namedtuple
    :arg tolerance: tolerance on (relative) stress divergence to check convergence
    :arg depth: depth of Anderson acceleration
    :arg maxiter: maximal number of iterations
    """
    # reference values of Lame paraeter
    mu0 = 1 / 2 * (jnp.min(mu) + jnp.max(mu))
    lmbda0 = 1 / 2 * (jnp.min(lmbda) + jnp.max(lmbda))
    # Fourier vectors
    xizero = get_xizero(grid_spec, dtype=E_mean.dtype)
     # storage for solution and residual, arrays of shape (d+1,6,Nx,Ny,Nz)

    epsilon = jnp.zeros((depth + 1, 6) + grid_spec.N, dtype=E_mean.dtype)
    epsilon = epsilon.at[0, ...].set(jnp.expand_dims(E_mean, [1, 2, 3]))
    residual = jnp.zeros((depth + 1, 6) + grid_spec.N, dtype=epsilon.dtype)
     # Anderson matrix and vectors
    A_anderson = jnp.eye(depth + 1, dtype=epsilon.dtype)
    u_rhs = jnp.zeros(depth + 1, dtype=epsilon.dtype)
    
    # stress calculated using damage variable
    sigma = compute_sigma(lmbda, mu, epsilon[0, ...], d=d)

    print(f"Shape of sigma : {sigma.shape}")

    def exit_condition(state):
        epsilon, residual, sigma, A_anderson, u_rhs, iter = state
        rel_error = relative_divergence(sigma, grid_spec)
        return (rel_error > tolerance) & (iter < maxiter)

    def loop_body(state):
        epsilon, residual, sigma, A_anderson, u_rhs, iter = state
        sigma_hat = jnp.fft.fftn(sigma, axes=[-3, -2, -1])
        r_hat = fourier_solve(sigma_hat, lmbda0, mu0, xizero)
        r = jnp.real(jnp.fft.ifftn(r_hat, axes=[-3, -2, -1]))
        
        residual = jnp.roll(residual, 1, axis=0)
        residual = residual.at[0, ...].set(r)
        A_anderson = jnp.roll(A_anderson, (1, 1), axis=(0, 1))
        dotproduct_scaling = jnp.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=epsilon.dtype)
        A_anderson = A_anderson.at[0, :].set(jnp.einsum("aijk,saijk,a->s", r, residual, dotproduct_scaling))
        A_anderson = A_anderson.at[:, 0].set(A_anderson[0, :])
        u_rhs = jnp.roll(u_rhs, 1)
        u_rhs = u_rhs.at[0].set(1)
        
        v = jnp.linalg.solve(A_anderson, u_rhs)
        alpha = v / jnp.dot(v, u_rhs)
        epsilon_tilde = jnp.einsum("s,saijk", alpha, epsilon + residual)
        epsilon = jnp.roll(epsilon, 1, axis=0)
        epsilon = epsilon.at[0, ...].set(epsilon_tilde)
        
        # Incorped damage variable
        sigma = compute_sigma(lmbda, mu, epsilon[0, ...], d=d)
        iter += 1
        return (epsilon, residual, sigma, A_anderson, u_rhs, iter)

    epsilon, residual, sigma, A_anderson, u_rhs, iter = jax.lax.while_loop(
        exit_condition, loop_body, init_val=(epsilon, residual, sigma, A_anderson, u_rhs, 0)
    )

    return epsilon[0, ...], sigma, iter


def get_xi_sq(grid_spec, dtype=jnp.float64):
    """Construct the squared frequency vectors for the Laplacian operator
    
    This function computes xi * xi in Fourier space. 

    :arg grid_spec: grid specifications
    :arg dtype: data type
    """
    # Normalised momentum vectors in all three spatial directions
    K = [2 * np.pi * np.arange(n) / n for n in grid_spec.N]
    # Grid with momentum vectors
    xi = np.meshgrid(*K, indexing="ij")
    h = grid_spec.h
    
    # Grid with tilde(xi) using the finite-difference based definition
    xi_tilde = np.stack(
        [
            2 / h[0] * np.sin(xi[0] / 2) * np.cos(xi[1] / 2) * np.cos(xi[2] / 2),
            2 / h[1] * np.cos(xi[0] / 2) * np.sin(xi[1] / 2) * np.cos(xi[2] / 2),
            2 / h[2] * np.cos(xi[0] / 2) * np.cos(xi[1] / 2) * np.sin(xi[2] / 2),
        ]
    )

    
    # print(f"Shape of xi_tilde : {xi_tilde.shape}")
    
    # Compute the dot product xi . xi
    xi_sq = jnp.sum(xi_tilde**2, axis=0)
    
    #print(f"Shape of xi_sq (After Sum): {xi_sq.shape}")
  
    
    # Compute the dot product xi . xi (Note: axis 0 contains the vector components 3x64x64x64)
    xi_sq = jnp.sum(xi_tilde**2, axis=0)
    
    return jnp.array(xi_sq, dtype=dtype)


def get_laplacian(grid_spec, dtype=jnp.float64):
    """Construct the laplacian operator in Fourier space

    This function returns a tensor of shape (1,N_0,N_1,N_2) which contains
    the laplacian operator (xixi) in fourier space.

     :arg grid_spec: namedtuple with grid specifications
     :arg dtype: data type

    """
    h = grid_spec.h
    # Normalised momentum vectors in all three spatial directions
    K = [2 * np.pi * np.arange(n) / n for n in grid_spec.N]
    # Grid with normalised momentum vectors
    xi = np.meshgrid(*K, indexing="ij")
    # Grid with xi*xi (laplacian in Fourier space)
    xixi = 2.* (jnp.cos(xi[0]) - 1.) / h[0]**2 \
         + 2.* (jnp.cos(xi[1]) - 1.) / h[1]**2 \
         + 2.* (jnp.cos(xi[2]) - 1.) / h[2]**2
    return jnp.array(xixi, dtype=dtype)



@jax.jit(static_argnames=["grid_spec", "tolerance", "maxiter"])
def phase_field_fixed_point(H, d_old, gc, lc, grid_spec, tolerance=1e-6, maxiter=10000):
    """Solve the phase-field equation using a FFT-based fixed-point algorithm
    
    Solves A_0 d(x) - \Delta d(x) = \chi(x) in Fourier space.

    :arg H: History field of maximum positive elastic energy
    :arg d_old: Fracture phase field from the previous time step
    :arg gc: Critical fracture energy 
    :arg lc: Characteristic length scale 
    :arg grid_spec: grid specification
    :arg tolerance: Tolerance on the polarization field residual to check convergence
    :arg maxiter: Max number of iterations
    
    """

    # Could add visous parameter here but it slows down the computation speed

    # Compute the coefficients A^t_n and B^t_n
    A_n = 1.0 / (lc**2) + 2.0 * H / (gc * lc)
    B_n = 2.0 * H / (gc * lc)
    
    # Homogeneous reference parameter A_0
    A0_n = 0.5 * (jnp.min(A_n) + jnp.max(A_n))
    
    
    # Pre-compute the Laplacian operator in Fourier space
    laplacian = get_laplacian(grid_spec, dtype=d_old.dtype)

    def exit_condition(state):
        """Check exit condition
        
        Check whether the residual ||chi^{k+1} - chi^k||_2 / ||chi^{k+1}||_2 > tolerance or iter > maxiter
        """
        d_k, chi_k, iter, residual = state
        return (residual > tolerance) & (iter < maxiter)

    def loop_body(state):
        """Update phase-field, polarization field, and compute residual"""
        d_k, chi_k, iter, residual = state
        
        d_new = jnp.real(jnp.fft.ifftn(jnp.fft.fftn(chi_k) / (A0_n - laplacian)))

        chi_new = B_n - (A_n - A0_n) * d_new
        
        norm_diff = jnp.linalg.norm(chi_new - chi_k)

        norm_chi_new = jnp.linalg.norm(chi_new)
        #Check for residual divergence and avoid division by zero
        residual = jnp.where(norm_chi_new > 1e-12, norm_diff / norm_chi_new, 0.0)
        
        return (d_new, chi_new, iter + 1, residual)
    
    # Initialising variables for the first step (at step k=0)
    d_initial = d_old
    chi_initial = B_n - (A_n - A0_n) * d_initial # Eq 17
    # Set initial residual to 1
    initial_residual = jnp.array(1.0, dtype=d_old.dtype)

    # Execute the fixed-point iteration loop
    d_final, chi_final, iter_count, res_final = jax.lax.while_loop(
        exit_condition,
        loop_body,
        init_val=(d_initial, chi_initial, 0, initial_residual)
    )

    return d_final, iter_count





def voigt_to_tensor(v):
    """Convert 6-component Voigt notation (6, Nx, Ny, Nz) to 3x3 symmetric tensor (Nx, Ny, Nz, 3, 3).
    """
    v0, v1, v2, v3, v4, v5 = v[0], v[1], v[2], v[3], v[4], v[5]
    
    row0 = jnp.stack([v0, v5, v4], axis=-1) #slots into the end
    row1 = jnp.stack([v5, v1, v3], axis=-1) 
    row2 = jnp.stack([v4, v3, v2], axis=-1)
    
    return jnp.stack([row0, row1, row2], axis=-2) #


def tensor_to_voigt(t):
    """Convert 3x3 symmetric tensor (Nx, Ny, Nz, 3, 3) back to 6-component Voigt (6, Nx, Ny, Nz).
    """
    v0 = t[..., 0, 0]
    v1 = t[..., 1, 1]
    v2 = t[..., 2, 2]
    v3 = t[..., 1, 2]
    v4 = t[..., 0, 2]
    v5 = t[..., 0, 1]
    
    return jnp.stack([v0, v1, v2, v3, v4, v5], axis=0)


def compute_strain_energy(lmbda, mu, epsilon):
    """Compute ONLY the positive/tensile elastic strain energy to drive the fracture."""
    # 1. Convert to 3x3 tensor
    eps_tensor = voigt_to_tensor(epsilon)
    
    # 2. Get trace and split into positive volumetric part
    tr_eps = jnp.trace(eps_tensor, axis1=-2, axis2=-1)
    tr_eps_plus = jnp.maximum(tr_eps, 0.0)
    
    # 3. Calculate deviatoric strain
    I_tensor = jnp.eye(3, dtype=epsilon.dtype)
    eps_dev = eps_tensor - (1.0 / 3.0) * tr_eps[..., None, None] * I_tensor
    
    # 4. Deviatoric energy (eps_dev : eps_dev)
    eps_dev_sq = jnp.sum(eps_dev * eps_dev, axis=(-2, -1))
    
    # 5. 3D Bulk Modulus
    K = lmbda + (2.0 / 3.0) * mu
    
    # 6. Compute driving energy: Volumetric tension + full Deviatoric shear
    psi_plus = 0.5 * K * (tr_eps_plus**2) + mu * eps_dev_sq
    
    return psi_plus


def compute_sigma(lmbda, mu, epsilon, d=None, k=1e-12):
    """Compute stress with asymmetric volumetric-deviatoric degradation."""
    eps_tensor = voigt_to_tensor(epsilon)
    
    # 1. Volumetric strains
    tr_eps = jnp.trace(eps_tensor, axis1=-2, axis2=-1)
    tr_eps_plus = jnp.maximum(tr_eps, 0.0)
    tr_eps_minus = jnp.minimum(tr_eps, 0.0)
    
    # 2. Deviatoric strain
    I_tensor = jnp.eye(3, dtype=epsilon.dtype)
    eps_dev = eps_tensor - (1.0 / 3.0) * tr_eps[..., None, None] * I_tensor
    
    # 3. 3D Bulk modulus
    K = lmbda + (2.0 / 3.0) * mu
    
    # Identity matrix in Voigt notation for the Trace operation
    I_voigt = jnp.stack([
        jnp.ones_like(tr_eps), jnp.ones_like(tr_eps), jnp.ones_like(tr_eps), 
        jnp.zeros_like(tr_eps), jnp.zeros_like(tr_eps), jnp.zeros_like(tr_eps)
    ], axis=0)
    
    # 4. Compute separated stress components
    eps_dev_v = tensor_to_voigt(eps_dev)
    
    sigma_dev = 2.0 * mu * eps_dev_v
    sigma_vol_plus = K * tr_eps_plus * I_voigt
    sigma_vol_minus = K * tr_eps_minus * I_voigt
    
    # 5. Group into degradable (tension + shear) and non-degradable (compression)
    sigma_plus = sigma_vol_plus + sigma_dev
    sigma_minus = sigma_vol_minus
    
    # 6. Apply damage degradation ONLY to the tension/shear stress
    if d is not None:
        g_d = (1.0 - d)**2 + k
        return g_d * sigma_plus + sigma_minus
    
    return sigma_plus + sigma_minus




def solve_fracture_staggered(grid_spec, lmbda, mu, gc, lc, E_mean_steps, d_0=None, k=1e-12, save_steps=None):
    """Use the staggered scheme to solve the coupled phase-field and mechanical problem
    
    :arg E_mean_steps: Applied strain steps
    :arg save_steps: List of integer step indices at which to save the full 3D fields. 
    If None, it only saves the final step. To save all steps, pass list(range(len(E_mean_steps))
    :arg d_0: Initial damage field. If None, it is initialised to zero everywhere. Otherwise, it should be an array of shape (Nx, Ny, Nz) with values in [0,1].
    :arg k: Stabilisation parameter for the damage degradation function to avoid singularity when d=1. 
    :arg other args: see previous functions

    
    """
    if d_0 is None:
        d = jnp.zeros(grid_spec.N, dtype=lmbda.dtype)
        
    else:
        d = d_0
    
    H = jnp.zeros(grid_spec.N, dtype=lmbda.dtype)
    

    if save_steps is None:
        save_steps = [len(E_mean_steps) - 1]

    d_history = {}
    eps_history = {}
    sigma_history = {}
    
    sig_lst = []
    eps_lst = []
    
    for step, E_mean in enumerate(E_mean_steps):
        print(f"Time Step {step} ")
        
        with measure_time("Phase-field solver"):
            d, iter_pf = phase_field_fixed_point(H, d, gc, lc, grid_spec)
            jax.block_until_ready(d)
            print(f"  Converged in {iter_pf} iterations.")

            #Damage print
            #print(f"  Damage Field:  Max: {jnp.max(d)} | Min: {jnp.min(d):.4f} | Mean: {jnp.mean(d):.4f}")
        
        with measure_time("Mechanical solver"):
            eps, sigma, iter_mech = lippmann_schwinger(lmbda, mu, E_mean, grid_spec, d=d, k=k)
            jax.block_until_ready(eps)
            print(f"  Converged in {iter_mech} iterations.")
            
        

       
        #  Save the macroscopic averages 
        sigAV  = np.array([np.mean(sigma[i]) for i in range(6)])
        sig_lst.append(sigAV)

        epsAV = np.array([np.mean(eps[i]) for i in range(6)])
        eps_lst.append(epsAV)

        with measure_time("History update"):
            psi = compute_strain_energy(lmbda, mu, eps)
            H = jnp.maximum(H, psi)
            jax.block_until_ready(H)
            
        
        if step in save_steps:
            print(f" >>Saving full 3D fields for step {step}...")
            
            d_history[step] = d
            eps_history[step] = eps
            sigma_history[step] = sigma
            
    return d_history, eps_history, sigma_history, eps_lst, sig_lst