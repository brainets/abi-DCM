import jax, jax.numpy as jnp
import vbjax as vb

def dcm(x, up):
    '''Extension of function vbjax.dcm_dfun to unpack 
    input & parameters from a joint tuple (u,p)'''
    u, p = up
    return vb.dcm_dfun(x, u, p)


def make_ode(dt, dfun, adhoc=None):
    '''Extension of function vbjax.make_ode to simulate
    time-dependent input profiles'''
    
    def step(x, t, up):
        return vb.heun_step(x, dfun, dt, up, adhoc=adhoc)

    @jax.jit
    def loop(x0, tus, p):
        def op(x, tu):
            t,u = tu
            x = step(x, t, (u,p))
            return x, x
        return jax.lax.scan(op, x0, tus)[1]

    return step, loop


@jax.jit
def dcm_bilinear_predict(TRLs, dt, x0, ts, us, p, eps):
    '''Compute predictions for all time steps in the DCM process'''
    _, loop = make_ode(dt, dcm)
    xs = jax.vmap(lambda trl: loop(x0, (ts,us[trl]), p))(TRLs)
    return jnp.add(xs,eps)
