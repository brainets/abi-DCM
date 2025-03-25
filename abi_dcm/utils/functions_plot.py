import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import seaborn as sns
from numpy import NaN
import jax.numpy as jnp

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
    '''Plots some trials for one ROI'''

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
    