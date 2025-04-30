import matplotlib.pyplot as plt
import vbjax as vb
import jax.numpy as jnp
from numpy import NaN

from ..utils.models import dcm_bilinear_predict
from ..utils.stims import stim_signal


def plot_signals(time_pts=None, data_plot=None, num_plots=5, label_= 'Signal'):

    fig, axes = plt.subplots(num_plots, 1, sharex=True,sharey=True)

    for i in range(num_plots): # Trials

        if isinstance(label_,str):
            label = f'{label_} {i}-th'
        else:
            label = label_[i]

        if time_pts is None:
            axes[i].plot(data_plot[i], 'k', label = label)
        else:
            axes[i].plot(time_pts, data_plot[i], 'k', label = label)

        if i<num_plots-1:
            axes[i].tick_params(labelbottom=False)
        axes[i].legend(fontsize=6, loc='upper left')


def plot_trials(time_pts=None, data=None, ROI=0, num_trials=5):
    '''Plots some trial signals for one ROI'''

    num_plots=num_trials
    data_plot = data[:,ROI]
    plot_signals(time_pts, data_plot, num_plots, label_ = 'Trial')


def plot_ROIs(time_pts=None, data=None, num_ROIs=5, name_ROIs=None):
    '''Plots the trial-average for some ROIs'''

    num_plots=num_ROIs
    data_plot = data
    if name_ROIs is not None:
        plot_signals(time_pts, data_plot, num_plots, label_ = name_ROIs)
    else:
        plot_signals(time_pts, data_plot, num_plots, label_ = 'ROI')


def plot_ROIs_DCM(i_cond=0, axes=None, titles=None, num_ROIs=5, name_ROIs_set=None, colors=None, \
                  data=None, stim_sh=0, stim=None, stim_pars=None, p=None, time_pts=None, onset_ind=0, \
                  ninputs=None, ntime=None, TRLs=None, dt=None, x0=None, eps=None):
    '''Plots the trial-average for some ROIs, together with DCM model predictions.
       Returns model predictions and the Sum of Squared Errors (SSE) respect to exp. data'''
    
    # Exprimental data, excluding baseline activity
    xs_data = data[:,onset_ind:]
    
    # Model prediction
    if stim is None:
        # Create the stimulus input from a vector of given parameter values
        if stim_sh: # Gamma input
            alpha, beta = stim_pars[:ninputs], stim_pars[ninputs:]
            stim = stim_signal(shape='Gamma', ninputs=ninputs, ntime=ntime, stim_onset=onset_ind, alpha=alpha, beta=beta)
        else: # Alpha input
            stim_tau = stim_pars
            stim = stim_signal(shape='Alpha', ninputs=ninputs, ntime=ntime, stim_onset=onset_ind, stim_tau=stim_tau)
    ts = time_pts[onset_ind:]        
    us = jnp.matrix_transpose(stim[...,ts])
    xs_model = dcm_bilinear_predict(TRLs, dt, x0, ts, us, p, eps).squeeze()
    
    # SSE to compare model predictions with exp. data
    sse = lambda x,y: jnp.sum(jnp.square(x-y))
    
    ROIs = range(num_ROIs)
    for roi in ROIs:
        ax = axes[roi, i_cond]
        ax.plot(ts,   xs_model[...,roi].T, color=colors[roi], linestyle='-.')
        ax.plot(time_pts, data[...,roi].T, color=colors[roi])
        if roi==0:
            ax.set_title(f'{titles[i_cond]}: sse = {sse(xs_model, xs_data):0.2e}', fontsize=18)
        if roi==num_ROIs-1:
            ax.set_xlabel('time (dt)')
        if i_cond==0:
            ax.text(.1,.5, name_ROIs_set.iat[roi], fontsize=15)
    
    return sse(xs_model, xs_data), xs_model
    