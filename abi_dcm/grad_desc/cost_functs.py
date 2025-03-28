import jax, jax.numpy as jnp
import vbjax as vb
from ..utils.matrices import reconst_A
from ..utils.stims import stim_signal
from ..utils.models import dcm_bilinear_predict


def loss_fun_all(p_hat_vec, A_triu_idx, A_tril_idx, B, C_cond_nZero_idx, tau, \
                 pos_max, TRLs, dt, x0, eps, xs_exp, \
                 C_shape=None, stim_sh=0, ntime=None, onset_ind=0):
    '''A cost function to optimize a DCM by adjusting 
    matrices A and C, along with inputs parameters'''
    
    # Reconstruct matrix A from a flat vector of estimated coefficients
    A_ncols = C_shape[0]
    size_triu, size_tril = A_triu_idx[0].size, A_tril_idx[0].size
    A_hat_vec = jax.lax.dynamic_slice(p_hat_vec, (0,), (size_triu+size_tril+1,))
    A_hat = reconst_A(A_ncols, A_triu_idx, A_tril_idx, A_hat_vec)
    
    # Reconstruct matrix C from a flat vector of estimated coefficients
    C_hat = jnp.zeros(C_shape)
    size_C_stim = C_cond_nZero_idx[0].size
    C_hat_vec = jax.lax.dynamic_slice(p_hat_vec, (size_triu+size_tril+1,),(size_C_stim,))
    C_hat = C_hat.at[C_cond_nZero_idx].set(C_hat_vec)

    # Reconstruct input's parameters from a flat vector of estimated values
    ninputs = C_shape[1]
    par_shape = (ninputs,1)
    match stim_sh:
        case 1:  # Gamma input
            
            stim_pars_hat_vec = jax.lax.dynamic_slice(p_hat_vec, (size_triu+size_tril+1+size_C_stim,),(2*ninputs,))
            alpha_hat = jnp.reshape(stim_pars_hat_vec[:ninputs], par_shape)
            beta_hat = (alpha_hat - 1)/pos_max
            stim = stim_signal(shape='Gamma', ninputs=ninputs, ntime=ntime, stim_onset=onset_ind, alpha=alpha_hat, beta=beta_hat)
        case 0:  # Alpha input
            stim_pars_hat_vec = jax.lax.dynamic_slice(p_hat_vec, (size_triu+size_tril+1+size_C_stim,),(ninputs,))
            tau_hat = jnp.reshape(stim_pars_hat_vec, par_shape)
            stim = stim_signal(shape='Alpha', ninputs=ninputs, ntime=ntime, stim_onset=onset_ind, stim_tau=tau_hat)
        
    ts = jnp.r_[onset_ind:ntime]
    us_hat = jnp.matrix_transpose(stim[...,ts])
    p_hat = vb.DCMTheta(A=A_hat/tau, B=B/tau, C=C_hat/tau)
    xs_hat_c = dcm_bilinear_predict(TRLs, dt, x0, ts, us_hat, p_hat, eps)
    
    sse = lambda x,y: jnp.sum(jnp.square(x-y))
    return sse(xs_hat_c, xs_exp)

    
def loss_fun_A_C(p_hat_vec, A_triu_idx, A_tril_idx, B, C_cond_nZero_idx, tau, \
                 stim_pars_cond, TRLs, dt, x0, eps, xs_exp, \
                 C_shape=None, stim_sh=0, ntime=None, onset_ind=0):
    '''A cost function to optimize a DCM by adjusting 
    matrices A and C. Input parameters are excluded'''
    
    # Reconstruct matrix A from a flat vector of estimated coefficients
    A_ncols = C_shape[0]
    size_triu, size_tril = A_triu_idx[0].size, A_tril_idx[0].size
    A_hat_vec = jax.lax.dynamic_slice(p_hat_vec, (0,), (size_triu+size_tril+1,))
    A_hat = reconst_A(A_ncols, A_triu_idx, A_tril_idx, A_hat_vec)
    
    # Reconstruct matrix C from a flat vector of estimated coefficients
    C_hat = jnp.zeros(C_shape)
    size_C_stim = C_cond_nZero_idx[0].size
    C_hat_vec = jax.lax.dynamic_slice(p_hat_vec, (size_triu+size_tril+1,),(size_C_stim,))
    C_hat = C_hat.at[C_cond_nZero_idx].set(C_hat_vec)

    # Extracting Input's parameters from tensor of estimated values
    ninputs = C_shape[1]
    match stim_sh:
        case 1:  # Gamma input
            alpha_cond, beta_cond = stim_pars_cond[:ninputs], stim_pars_cond[ninputs:]
            stim = stim_signal(shape='Gamma', ninputs=ninputs, ntime=ntime, stim_onset=onset_ind, alpha=alpha_cond, beta=beta_cond)
        case 0:  # Alpha input
            stim_tau_cond = stim_pars_cond
            stim = stim_signal(shape='Alpha', ninputs=ninputs, ntime=ntime, stim_onset=onset_ind, stim_tau=stim_tau_cond)
    
    ts = jnp.r_[onset_ind:ntime]
    us_hat = jnp.matrix_transpose(stim[...,ts])
    p_hat = vb.DCMTheta(A=A_hat/tau, B=B/tau, C=C_hat/tau)
    xs_hat_c = dcm_bilinear_predict(TRLs, dt, x0, ts, us_hat, p_hat, eps)
    
    sse = lambda x,y: jnp.sum(jnp.square(x-y))
    return sse(xs_hat_c, xs_exp)