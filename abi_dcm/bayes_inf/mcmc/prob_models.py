import jax, jax.numpy as jnp
import vbjax as vb
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_mean, init_to_median, Predictive
import numpyro.distributions as dist
numpyro.set_host_device_count(jax.local_device_count())

from ...utils.stims import stim_signal
from ...utils.models import dcm_bilinear_predict
from ...utils.matrices import nZero_coeff_idx

def DCM_bilinear(A, C, Ahat0, Ahat0_triu_idx, Ahat0_tril_idx, A_fract, Chat0, Chat0_nZero_idx, \
                 i_cond, stim_pars, tau, B, TRLs, dt, x0, eps, Lambda_max, cov_matrix, xs_exp, \
                 GD=False, GC_std=True, stim_sh=1, ninputs=1, ntime=None, onset_ind=0):
    ''' A generative model for a bilinear DCM, with a Gaussian likelihood model
        and Normal/LogNormal prior distributions. Only matrices A and C are considered.'''
    
    #### Priors for matrix A ####
    A_hat = jnp.zeros(Ahat0.shape)
    if GD:
        Ahat0_diag = Ahat0[0,0]
    else:
        Ahat0_diag = A[0,0]
    ## Damped dynamics for matrix A_hat
    diag_idx = jnp.diag_indices_from(A)
    A_hat = A_hat.at[diag_idx].set(Ahat0_diag)
    # A_diag_std = A_fract*jnp.abs(Ahat0_diag)
    # A_diag_hat = numpyro.sample('A_diag_hat', dist.Normal(Ahat0_diag, A_diag_std))
    # A_hat = A_hat.at[diag_idx].set(A_diag_hat)
  
    ### Complete A_hat from a flat array on non-null coefficients in Ahat0
    ## Upper triangle for A_hat
    if Ahat0_triu_idx:
        Ahat0_triu_vec = Ahat0[Ahat0_triu_idx]
        Ahat0_triu_std = A_fract*jnp.abs(Ahat0_triu_vec)
        if GC_std:
            Ahat0_triu_hat = numpyro.sample('A_triu_hat', dist.Normal(0., Ahat0_triu_std))
        else:
            Ahat0_triu_hat = numpyro.sample('A_triu_hat', dist.Normal(Ahat0_triu_vec, Ahat0_triu_std))
        A_hat = A_hat.at[Ahat0_triu_idx].set(Ahat0_triu_hat)
    ## Lower triangle for A_hat
    if Ahat0_tril_idx:
        Ahat0_tril_vec = Ahat0[Ahat0_tril_idx]
        Ahat0_tril_std = A_fract*jnp.abs(Ahat0_tril_vec)
        if GC_std:
            Ahat0_tril_hat = numpyro.sample('A_tril_hat', dist.Normal(0., Ahat0_tril_std))
        else:
            Ahat0_tril_hat = numpyro.sample('A_tril_hat', dist.Normal(Ahat0_tril_vec, Ahat0_tril_std))
        A_hat = A_hat.at[Ahat0_tril_idx].set(Ahat0_tril_hat)
    ## A_hat
    A_hat = numpyro.deterministic(f'A_hat_{i_cond}', A_hat)
    
    #### Priors for matrix C ####
    C_hat = jnp.zeros(Chat0.shape)
    ### Complete C_hat from a flat array on non-null coefficients in Chat0
    Chat0_vec = jnp.expand_dims(Chat0[Chat0_nZero_idx], axis=1)
    Chat0_hat = numpyro.sample('C_hat', dist.LogNormal(jnp.log(Chat0_vec),1/16))
    C_hat = C_hat.at[Chat0_nZero_idx].set(Chat0_hat.squeeze())
    C_hat = numpyro.deterministic(f'C_hat_{i_cond}',C_hat)
    
    #### Priors for the stimuli ####
    if stim_sh: # Gamma shape
        # Parameters for Gamma-shape input functions, approx. as in Chen et al.[2008]
        alpha_cond, beta_cond = stim_pars[:ninputs], stim_pars[ninputs:]
        theta_sigma = numpyro.sample(f'theta_sigma_{i_cond}', dist.LogNormal(jnp.log(alpha_cond/beta_cond**2),1/16))
        while True: # alpha must be larger than 1
            theta_mu = numpyro.sample(f'theta_mu_{i_cond}', dist.LogNormal(jnp.log(alpha_cond/beta_cond),1/16))
            if (theta_mu > jnp.sqrt(theta_sigma)).all:
                break
                
        alpha = numpyro.deterministic(f'alpha_{i_cond}', theta_mu**2/theta_sigma)
        beta = numpyro.deterministic(f'beta_{i_cond}', theta_mu/theta_sigma)
        stim = stim_signal(shape='Gamma', ntime=ntime, ninputs=ninputs, stim_onset=onset_ind, alpha=alpha, beta=beta)
    else: # Alpha input
        stim_tau_cond = stim_pars
        stim_tau_hat = numpyro.sample(f'stim_tau_{i_cond}', dist.LogNormal(jnp.log(stim_pars),1/16))
        stim = stim_signal(shape='Alpha', ntime=ntime, ninputs=ninputs, stim_onset=onset_ind, stim_tau=stim_tau_hat)
    
    ts = jnp.r_[onset_ind:ntime]
    stim_hat = numpyro.deterministic(f'stim_{i_cond}', stim[...,ts])
    us_hat = jnp.matrix_transpose(stim_hat)
    
    ### Prior for tau ####
    tau_hat = numpyro.sample(f'tau_{i_cond}', dist.LogNormal(jnp.log(tau),1/16)) # in seconds
    
    ### Bilinear model's output ####
    p_hat = vb.DCMTheta(A=A_hat/tau_hat, B=B/tau_hat, C=C_hat/tau_hat)
    xs_hat_c = dcm_bilinear_predict(TRLs, dt, x0, ts, us_hat, p_hat, eps)

    #### Likelihood model ####
    Lambda = numpyro.sample(f'Lambda_{i_cond}', dist.LogNormal(jnp.log(Lambda_max),1))
    numpyro.sample(f'xs_hat_c_{i_cond}', dist.MultivariateNormal(loc=xs_hat_c, covariance_matrix=1/Lambda*cov_matrix), obs=xs_exp)


def fit_model_conf(i_cond, A_gd=None, C_gd=None, A_ref=None, C=None, \
                   conditions=None, A_fract=None, stim_pars=None, 
                   num_chains=None, rng_key=None, GD=False, GC_std=True):
    '''Runs Markov Chain Monte Carlo inversion for a probabilistic DCM model'''
    
    print(f'Fitting model configuration {i_cond+1}...')
            
    if GD:
        Ahat0, A_fract = A_gd.at[i_cond].get(), 0.03
        Chat0 = C_gd.at[i_cond].get()
    else:
        Ahat0, Chat0 = A_ref, C*conditions[i_cond]
        if GC_std: # To use GC in std of the prior probability for matrix A, resembling the approach in Chen et al. (2008)
            A_fract = 1.
        else:      # To use GC as mean of the prior probability for matrix A
            A_fract = 0.10
    Ahat0_triu_idx, Ahat0_tril_idx = nZero_coeff_idx(Ahat0)
    Chat0_nZero_idx = jnp.where(Chat0)

    nuts_kernel = NUTS(DCM_bilinear, target_accept_prob=0.95, init_strategy=init_to_mean)
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=1000, num_chains=num_chains)
    mcmc.run(rng_key, Ahat0, Ahat0_triu_idx, Ahat0_tril_idx, A_fract, \
             Chat0, Chat0_nZero_idx, stim_pars, i_cond, xs_exp)
    
    postSamples = mcmc.get_samples()
    DCM_PredictiveObject = Predictive(model=DCM_bilinear, posterior_samples=postSamples)
    postPredData = DCM_PredictiveObject(rng_key, Ahat0, Ahat0_triu_idx, Ahat0_tril_idx, A_fract, \
                                        Chat0, Chat0_nZero_idx, stim_pars, i_cond, xs_exp=None)

    return postSamples, postPredData
