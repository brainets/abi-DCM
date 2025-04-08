import jax, jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_mean, init_to_median, Predictive
numpyro.set_host_device_count(jax.local_device_count())

from .prob_models import DCM_bilinear 
from ...utils.matrices import nZero_coeff_idx


def fit_model_conf(i_cond, xs_exp, A_gd, A, B, C_gd, C, conditions, stim_pars_cond, tau, \
                   TRLs, dt, x0, eps, Lambda_max, cov_matrix, stim_sh=1, ninputs=1, \
                   ntime=1, onset_ind=0, GD=False, GC_std=True, num_chains=None, rng_key=None):
    '''Runs Markov Chain Monte Carlo inversion for a probabilistic DCM model'''
   
    print(f'Fitting model configuration {i_cond+1}...') 
    if GD:
        Ahat0, A_fract = A_gd.at[i_cond].get(), 0.03
        Chat0 = C_gd.at[i_cond].get()
    else:
        Ahat0, Chat0 = A, C*conditions[i_cond]
        if GC_std: # To use GC in std of the prior probability for matrix A, resembling the approach in Chen et al. (2008)
            A_fract = 1.
        else:      # To use GC as mean of the prior probability for matrix A
            A_fract = 0.10
    Ahat0_triu_idx, Ahat0_tril_idx = nZero_coeff_idx(Ahat0)
    Chat0_nZero_idx = jnp.where(Chat0)

    nuts_kernel = NUTS(DCM_bilinear, target_accept_prob=0.95, init_strategy=init_to_mean)
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=1000, num_chains=num_chains)
    mcmc.run(rng_key, A, Ahat0, Ahat0_triu_idx, Ahat0_tril_idx, A_fract, B, Chat0, Chat0_nZero_idx, \
                     f'_{i_cond}', stim_pars_cond[i_cond], tau, TRLs, dt, x0, eps, Lambda_max, cov_matrix, \
                     GD, GC_std, stim_sh, ninputs, ntime, onset_ind, xs_exp)
    
    postSamples = mcmc.get_samples()
    DCM_PredictiveObject = Predictive(model=DCM_bilinear, posterior_samples=postSamples)
    postPredData = DCM_PredictiveObject(rng_key, A, Ahat0, Ahat0_triu_idx, Ahat0_tril_idx, A_fract, B, Chat0, Chat0_nZero_idx, \
                     f'_{i_cond}', stim_pars_cond[i_cond], tau, TRLs, dt, x0, eps, Lambda_max, cov_matrix, \
                     GD, GC_std, stim_sh, ninputs, ntime, onset_ind)

    return postSamples, postPredData
