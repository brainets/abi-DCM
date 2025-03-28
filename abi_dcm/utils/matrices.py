import jax, jax.numpy as jnp


def damped_dynamics(D):
    '''Returns a damped dynamics version of a matrix D'''
    nvar = D.shape[0]
    D_conns = jnp.reshape(D[~jnp.isnan(D)], (-1,nvar-1))
    diag = -max(jnp.ceil(jnp.abs(jnp.sum(D_conns, axis=1))))
    # diag = -max(jnp.abs(jnp.sum(D_conns, axis=1)))
    
    '''
    # Considering only the input coeffs.
    D_st = D[:,[2,5]]
    D_vals = D_st[~jnp.isnan(D_st)]
    diag = -jnp.ceil(jnp.abs(jnp.sum(D_vals)))
    '''
    diag_idx = jnp.diag_indices_from(D)
    return D.at[diag_idx].set(diag) # damped dynamics at each node


def nZero_coeff_idx(A):
    ''' Extracts the indices of non-null 
    off-diagonal coefficients in matrix A'''
    
    # Upper triangle
    A_triu_idx = jnp.where(jnp.triu(A, k=1))
    # Lower triangle
    A_tril_idx = jnp.where(jnp.tril(A, k=-1))
    
    return A_triu_idx, A_tril_idx


def reconst_A(n_cols, A_triu_idx, A_tril_idx, A_vec):
    '''Reconstructs a squared matrix A from the upper and 
    lower triangles extracted from a flat array'''
    
    A_shape = (n_cols,n_cols,)
    A = jnp.zeros(A_shape)

    size_triu, size_tril = len(A_triu_idx[0]), len(A_tril_idx[0])
    # Upper triangle of the matrix
    A_triu_vec = jax.lax.dynamic_slice(A_vec, (0,), (size_triu,))
    A = A.at[A_triu_idx].set(A_triu_vec)  

    # Lower triangle of the matrix
    A_tril_vec = jax.lax.dynamic_slice(A_vec, (size_triu,), (size_tril,))
    A = A.at[A_tril_idx].set(A_tril_vec)

    # Damped dynamics at each node
    diag_idx = jnp.diag_indices_from(A)
    A_gd_diag = jax.lax.dynamic_slice(A_vec, (size_triu+size_tril,), (1,))
    A = A.at[diag_idx].set(A_gd_diag)
    
    return A
