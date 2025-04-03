import matplotlib.pyplot as plt
import vbjax as vb
import jax.numpy as jnp
from numpy import NaN
import seaborn as sns

from ..utils.models import dcm_bilinear_predict
from ..utils.stims import stim_signal

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


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
    
    # SSE to compare model predictions with exp. data
    sse = lambda x,y: jnp.sum(jnp.square(x-y))
    
    # Exprimental data: exclude baseline activity
    xs_data = jnp.expand_dims(data[onset_ind:], axis=0)
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
    
    ROIs = range(num_ROIs)
    for roi in ROIs:
        ax = axes[roi, i_cond]   
        ax.plot(time_pts[onset_ind:], xs_model[:,roi], linestyle='-.', color=colors[roi])
        ax.plot(time_pts, data[:,roi], color=colors[roi])
        if roi==0:
            ax.set_title(f'{titles[i_cond]}: sse = {sse(xs_model, xs_data):0.2e}', fontsize=18)
        if roi==num_ROIs-1:
            ax.set_xlabel('time (dt)')
        if i_cond==0:
            ax.text(.1,.5, name_ROIs_set.iat[roi], fontsize=15)
            
    return sse(xs_model, xs_data), xs_model

    
def plot_matrix(A=None, title_str='A', title_size=20, cmap='viridis', no_diag=True, no_zeros= True, \
                annot=True, annot_fs=8, annot_fmt=".3f", \
                shrink=1.0, vmin=None, vmax=None, cbar=False, ytickls=False, xtickls=False):
    '''Plots matrix A as a seaborn heatmap'''
    
    if no_diag:
        diag_idx = jnp.diag_indices_from(A)
        diag_A = A[0,0]
        A = A.at[diag_idx].set(NaN)
    if no_zeros:
        A = A.at[jnp.where(A==0.00)].set(NaN)
    
    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, square=True, cbar_kws={"shrink": shrink}, annot_kws={"fontsize":annot_fs})
    ax = sns.heatmap(A, **kw, cbar=cbar, yticklabels=ytickls, xticklabels=xtickls, annot=annot, fmt=annot_fmt)
    if no_diag and ~jnp.isnan(diag_A): 
        for i in range(len(A)):
            ax.text(diag_idx[0][i]+.5, diag_idx[1][i]+.5, f'{diag_A: .3f}', fontsize=annot_fs, \
                    horizontalalignment='center', verticalalignment='center', )
    ax.set_yticklabels(ax.get_yticklabels(),rotation=30)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=70)

    # Drawing the frame 
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1) 
    
    plt.title(title_str, fontsize=title_size)
    
    
def plot_correlations(trial=0, ROI=0, data=None):
    '''Plots autocorrelation and partial-autocorrelations for one trial of a ROI'''

    # Augmented Dickey Fuller test
    df_stationarityTest = adfuller(data[trial,ROI], autolag='AIC')
    # Check the p-value
    pval = df_stationarityTest[1]

    fig, axes = plt.subplots(1, 2, sharey=False)

    lag_max=15
    plot_pacf(data[trial,ROI], lags=lag_max, auto_ylims=True, ax=axes[0], \
             method='ywm')
    # plt.gca().set_box_aspect(0.9)
    # plt.legend(fontsize=7, loc='upper right')
    axes[0].set_xlabel('lag')
    axes[0].set_xlim(-0.5, lag_max)

    lag_max=30
    plt.suptitle(f'Trial {trial}-th, adf_pval: {pval:0.4f}')
    plot_acf(data[trial,ROI], lags=lag_max, auto_ylims=True, ax=axes[1])
    # plt.gca().set_box_aspect(0.7)
    axes[1].set_xlabel('lag')
    axes[1].set_xlim(-0.5, lag_max)
    
    