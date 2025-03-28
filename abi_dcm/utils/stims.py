import jax, jax.numpy as jnp


def stim_time_shape(shape=None, ts_stim=None, stim_tau=None, alpha=None, beta=None):
    '''Computes a signal stimulus with a specific shape'''
    
    match shape:
        case 'Exp':   #Exponential shape        
            stim = jnp.exp(-ts_stim/stim_tau)
    
        case 'Alpha': #Alpha shape
            # Normalized as a probability distribution
            # mean = 2*stim_tau, var = 2*stim_tau**2, mode = stim_tau
            stim = ts_stim/stim_tau**2*jnp.exp(-ts_stim/stim_tau)
            # stim = ts_stim/stim_tau*jnp.exp(1-ts_stim/stim_tau)
    
        case 'Gamma': # Gamma shape (alpha>1)
            # mean = theta_mu = alpha/beta, var = theta_sigma = mean/beta
            # mode = (alpha-1)/beta = (theta_mu**2-theta_sigma)/theta_mu
            stim = jnp.power(beta,alpha)*jnp.power(ts_stim,alpha-1) \
                  *jnp.exp(-beta*ts_stim)/jax.scipy.special.gamma(alpha)
            
        case 'Pulse': #Pulse shape
            stim = jnp.ones((dur_dt,))
        
        case _:
            print('Signal shape not recognized')

    return stim


def stim_time_full(stim_shape, stim_onset=None, ntime=None, ninputs=1):
    '''Computes a signal stimulus in the full time domain'''
    
    stim_time = jnp.zeros((ntime,))    
    return jax.lax.dynamic_update_slice(stim_time, stim_shape, (stim_onset,))


def stim_signal(shape='Alpha', ntrl=1, ninputs=1, stim_onset=None, ntime=None, \
                stim_tau=None, alpha=None, beta=None):
    '''Constructs a set of different stimulus signals, but identical across trials'''
    
    INPUTs = jnp.r_[:ninputs]
    ts_stim = jnp.tile(jnp.r_[:ntime-stim_onset],(ninputs,1))
    stim_shape = stim_time_shape(shape, ts_stim, stim_tau, alpha, beta)
    
    stim_trl = jax.vmap(lambda input: stim_time_full(stim_shape[input], stim_onset, ntime))(INPUTs)
    stim_trl = jnp.expand_dims(stim_trl,axis=0)
    
    return jnp.tile(stim_trl, (ntrl,1,1))