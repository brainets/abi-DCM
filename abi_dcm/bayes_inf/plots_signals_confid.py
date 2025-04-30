import jax.numpy as jnp


def plot_ROIs_probDCM(i_cond=0, axes=None, titles=None, num_ROIs=5, name_ROIs_set=None, colors=None, \
                       data=None, xs_model=None, percentiles_mod=None, time_pts=None, onset_ind=0):
    '''Plots the trial-average for some ROIs, together with DCM probabilsitc model predictions, 
       and confidence intervals. Returns the Sum of Squared Errors (SSE) respect to exp. data'''
    
    # SSE to compare model predictions with exp. data
    sse = lambda x,y: jnp.sum(jnp.square(x-y))
    
    # Exprimental data: exclude baseline activity
    xs_data = data[:,onset_ind:]
    
    ROIs = range(num_ROIs)
    for roi in ROIs:
        ax = axes[roi,i_cond]
        ax.plot(time_pts, data[...,roi].T, color=colors[roi])
        ax.plot(time_pts[onset_ind:], xs_model[...,roi].T, linestyle='-.', color=colors[roi])
        # plot 90% confidence level of predictions
        # ax.fill_between(time_pts[onset_ind:], percentiles_mod[0,...,roi], percentiles_mod[1,...,roi], color="lightblue")
        
        if roi==0:
            ax.set_title(f'{titles[i_cond]}: sse = {sse(xs_model, xs_data):0.2e}', fontsize=18)
        if roi==num_ROIs-1:
            ax.set_xlabel('time (dt)')
        if i_cond==0:
            ax.text(.1,.7, name_ROIs_set.iat[roi], fontsize=15)
            
    return sse(xs_model, xs_data)
