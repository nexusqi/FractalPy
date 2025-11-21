"""
Fixed-Point Solver for Fractal Neural Field.

Implements multiple strategies for finding fixed points:
1. Naive iterative solver (jax.lax.scan)
2. Anderson Acceleration (Jacobian-free)
3. Multigrid V-Cycle (simplified)
"""

from typing import Callable, Tuple, Optional
import jax
import jax.numpy as jnp


def naive_solver(
    kernel_fn: Callable,
    z_init: jnp.ndarray,
    input_injection: jnp.ndarray,
    num_steps: int = 30,
    capture_steps: Optional[list] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[dict]]:
    """
    Naive fixed-point solver using iterative updates.
    
    Args:
        kernel_fn: Function that takes (z, input_injection) -> z_new
        z_init: Initial hidden state [B, C, H, W] (complex)
        input_injection: Input injection [B, C, H, W] (complex)
        num_steps: Number of iteration steps
        capture_steps: Optional list of step indices to capture states
    
    Returns:
        Tuple of (final_state, convergence_history, captured_states)
        - final_state: [B, C, H, W] (complex)
        - convergence_history: [num_steps] (real, delta values)
        - captured_states: dict mapping step -> state, or None
    """
    capture_set = set(capture_steps) if capture_steps else set()
    
    def step(carry, step_idx):
        z, _ = carry
        z_new = kernel_fn(z, input_injection)
        # Compute convergence delta (change magnitude)
        delta = jnp.mean(jnp.abs(z_new - z))
        return (z_new, delta), (z_new, delta)
    
    step_indices = jnp.arange(num_steps)
    (final_state, _), (all_states, deltas) = jax.lax.scan(step, (z_init, 0.0), step_indices)
    
    # Extract captured states after scan
    captured_states = None
    if capture_steps:
        captured_states = {}
        for i in range(num_steps):
            if i in capture_set:
                captured_states[i] = all_states[i]
        # Also capture initial state if requested
        if 0 in capture_set:
            captured_states[0] = z_init
    
    return final_state, deltas, captured_states


def anderson_acceleration(
    kernel_fn: Callable,
    z_init: jnp.ndarray,
    input_injection: jnp.ndarray,
    num_steps: int = 30,
    m: int = 5,
    beta: float = 1.0,
    tol: float = 1e-6
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Anderson Acceleration for fixed-point finding.
    
    Stores history of last m residuals and computes optimal linear combination
    to minimize error. Jacobian-free method.
    
    Args:
        kernel_fn: Function that takes (z, input_injection) -> z_new
        z_init: Initial hidden state [B, C, H, W] (complex)
        input_injection: Input injection [B, C, H, W] (complex)
        num_steps: Maximum number of iteration steps
        m: History size for Anderson mixing
        beta: Mixing parameter (typically 1.0)
        tol: Convergence tolerance
    
    Returns:
        Tuple of (final_state, convergence_history)
    """
    def anderson_step(carry, _):
        z, residuals, f_vals = carry
        
        # Compute new function value
        f_z = kernel_fn(z, input_injection)
        residual = f_z - z
        
        # Compute convergence delta
        delta = jnp.mean(jnp.abs(residual))
        
        # Check convergence
        converged = delta < tol
        
        # Update history
        residuals_new = jnp.concatenate([residuals[1:], residual[None, ...]], axis=0)
        f_vals_new = jnp.concatenate([f_vals[1:], f_z[None, ...]], axis=0)
        
        # Anderson mixing
        if residuals.shape[0] > 1:
            # Compute differences
            df = f_vals_new[1:] - f_vals_new[:-1]  # [m-1, B, C, H, W]
            dr = residuals_new[1:] - residuals_new[:-1]  # [m-1, B, C, H, W]
            
            # Flatten for linear algebra
            df_flat = df.reshape(df.shape[0], -1)  # [m-1, B*C*H*W]
            dr_flat = dr.reshape(dr.shape[0], -1)  # [m-1, B*C*H*W]
            
            # Solve: min ||df @ gamma - residual||^2
            # Using least squares: gamma = (df^T @ df)^(-1) @ df^T @ residual
            residual_flat = residual.reshape(-1)  # [B*C*H*W]
            
            # Compute (df^T @ df) and (df^T @ residual)
            dfT_df = df_flat @ df_flat.T  # [m-1, m-1]
            dfT_res = df_flat @ residual_flat  # [m-1]
            
            # Solve linear system
            try:
                gamma = jnp.linalg.solve(dfT_df + 1e-8 * jnp.eye(dfT_df.shape[0]), dfT_res)
            except:
                # Fallback if solve fails
                gamma = jnp.zeros(m - 1)
            
            # Compute Anderson update
            # z_new = f_z - sum(gamma[i] * (f_vals[i+1] - f_vals[i]))
            f_update = jnp.sum(gamma[:, None, None, None, None] * df, axis=0)
            z_new = f_z - f_update
        else:
            # Not enough history yet, use standard update
            z_new = f_z
        
        # Apply mixing parameter
        z_new = beta * z_new + (1 - beta) * z
        
        return (z_new, residuals_new, f_vals_new), delta
    
    # Initialize history
    z0 = z_init
    f0 = kernel_fn(z0, input_injection)
    r0 = f0 - z0
    
    # Initialize history arrays
    residuals_init = jnp.zeros((m, *z_init.shape), dtype=z_init.dtype)
    f_vals_init = jnp.zeros((m, *z_init.shape), dtype=z_init.dtype)
    residuals_init = residuals_init.at[0].set(r0)
    f_vals_init = f_vals_init.at[0].set(f0)
    
    carry_init = (f0, residuals_init, f_vals_init)
    
    # Run Anderson acceleration
    (final_state, _, _), deltas = jax.lax.scan(
        anderson_step, carry_init, None, length=num_steps
    )
    
    return final_state, deltas


def downsample(z: jnp.ndarray, factor: int = 2) -> jnp.ndarray:
    """
    Downsample using average pooling.
    
    Args:
        z: Complex tensor [B, C, H, W]
        factor: Downsampling factor
    
    Returns:
        Downsampled tensor [B, C, H//factor, W//factor]
    """
    B, C, H, W = z.shape
    # Use average pooling
    # For complex numbers, pool real and imag separately
    z_real = jnp.real(z)
    z_imag = jnp.imag(z)
    
    # Reshape for pooling: [B, C, H, W] -> [B*C, 1, H, W]
    z_real_reshaped = z_real.reshape(B * C, 1, H, W)
    z_imag_reshaped = z_imag.reshape(B * C, 1, H, W)
    
    # Average pooling
    kernel_size = (factor, factor)
    strides = (factor, factor)
    
    # Manual pooling using conv with ones kernel
    pool_kernel = jnp.ones((1, 1, factor, factor)) / (factor * factor)
    
    z_real_pooled = jax.lax.conv_general_dilated(
        z_real_reshaped,
        pool_kernel,
        window_strides=strides,
        padding='VALID',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )
    
    z_imag_pooled = jax.lax.conv_general_dilated(
        z_imag_reshaped,
        pool_kernel,
        window_strides=strides,
        padding='VALID',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )
    
    # Reshape back
    H_new, W_new = z_real_pooled.shape[2], z_real_pooled.shape[3]
    z_real_pooled = z_real_pooled.reshape(B, C, H_new, W_new)
    z_imag_pooled = z_imag_pooled.reshape(B, C, H_new, W_new)
    
    return z_real_pooled + 1j * z_imag_pooled


def upsample(z: jnp.ndarray, target_shape: Tuple[int, int]) -> jnp.ndarray:
    """
    Upsample using bilinear interpolation.
    
    Args:
        z: Complex tensor [B, C, H, W]
        target_shape: Target (H, W) dimensions
    
    Returns:
        Upsampled tensor [B, C, target_H, target_W]
    """
    B, C, H, W = z.shape
    target_H, target_W = target_shape
    
    # For complex numbers, interpolate real and imag separately
    z_real = jnp.real(z)
    z_imag = jnp.imag(z)
    
    # Use JAX's resize (bilinear interpolation)
    z_real_upsampled = jax.image.resize(
        z_real, (B, C, target_H, target_W), method='bilinear'
    )
    z_imag_upsampled = jax.image.resize(
        z_imag, (B, C, target_H, target_W), method='bilinear'
    )
    
    return z_real_upsampled + 1j * z_imag_upsampled


def multigrid_vcycle(
    kernel_fn: Callable,
    z_init: jnp.ndarray,
    input_injection: jnp.ndarray,
    num_steps_coarse: int = 10,
    num_steps_fine: int = 10,
    solver_fn: Callable = naive_solver
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simplified Multigrid V-Cycle for fixed-point solving.
    
    Strategy:
    1. Downsample to coarse grid (e.g., 16x16)
    2. Solve on coarse grid
    3. Upsample solution
    4. Refine on fine grid (e.g., 28x28)
    
    This simulates "global to local" attention.
    
    Args:
        kernel_fn: Function that takes (z, input_injection) -> z_new
        z_init: Initial hidden state [B, C, H, W] (complex)
        input_injection: Input injection [B, C, H, W] (complex)
        num_steps_coarse: Steps on coarse grid
        num_steps_fine: Steps on fine grid
        solver_fn: Solver function to use (naive_solver or anderson_acceleration)
    
    Returns:
        Tuple of (final_state, convergence_history)
    """
    B, C, H, W = z_init.shape
    
    # Determine coarse grid size (approximately half)
    coarse_H = H // 2
    coarse_W = W // 2
    
    # Step 1: Downsample initial state and input
    z_coarse_init = downsample(z_init, factor=2)
    input_coarse = downsample(input_injection, factor=2)
    
    # Step 2: Create coarse kernel function
    def coarse_kernel_fn(z_coarse, input_coarse_inj):
        # Apply kernel on coarse grid
        z_new_coarse = kernel_fn(z_coarse, input_coarse_inj)
        return z_new_coarse
    
    # Step 3: Solve on coarse grid
    z_coarse_final, deltas_coarse = solver_fn(
        coarse_kernel_fn, z_coarse_init, input_coarse, num_steps_coarse
    )
    
    # Step 4: Upsample coarse solution
    z_fine_init = upsample(z_coarse_final, (H, W))
    
    # Step 5: Refine on fine grid
    z_fine_final, deltas_fine = solver_fn(
        kernel_fn, z_fine_init, input_injection, num_steps_fine
    )
    
    # Combine convergence history
    convergence_history = jnp.concatenate([deltas_coarse, deltas_fine])
    
    return z_fine_final, convergence_history


def solve_fixed_point(
    kernel_fn: Callable,
    z_init: jnp.ndarray,
    input_injection: jnp.ndarray,
    method: str = "naive",
    num_steps: int = 30,
    **kwargs
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unified interface for fixed-point solvers.
    
    Args:
        kernel_fn: Function that takes (z, input_injection) -> z_new
        z_init: Initial hidden state [B, C, H, W] (complex)
        input_injection: Input injection [B, C, H, W] (complex)
        method: Solver method ("naive", "anderson", "multigrid")
        num_steps: Number of iteration steps
        **kwargs: Additional arguments for specific solvers
    
    Returns:
        Tuple of (final_state, convergence_history)
    """
    if method == "naive":
        result = naive_solver(kernel_fn, z_init, input_injection, num_steps, **kwargs)
        # Return only first two elements for backward compatibility
        return result[0], result[1]
    elif method == "anderson":
        return anderson_acceleration(
            kernel_fn, z_init, input_injection, num_steps, **kwargs
        )
    elif method == "multigrid":
        return multigrid_vcycle(
            kernel_fn, z_init, input_injection, **kwargs
        )
    else:
        raise ValueError(f"Unknown solver method: {method}")

