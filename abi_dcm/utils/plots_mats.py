import matplotlib.pyplot as plt
import jax.numpy as jnp
from numpy import NaN
import seaborn as sns
from scipy.stats import pearsonr


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
    

def plot_matrices_grid(A_ref=None, A_ref_label='A_ref', A_conds=None, A_label='A', cond_idx=None, \
                   vmin=-1, vmax=1, sse_conds=None, shrink=0.75, title_size=10, titles=None, name_ROIs_set=None):
    ''' Plots Matrix A_ref, together with estimated matrices A_conds from different model configurations'''

    fig = plt.figure(figsize=(12,7))
    plt.subplot(1,4,1)
    plot_matrix(A=A_ref, shrink = 0.75, title_str=A_ref_label, title_size=12, ytickls=name_ROIs_set, xtickls=name_ROIs_set)

    A_conds_max = [ jnp.abs(A_conds[i_cond]).max() for i_cond in cond_idx ]
    for i_cond in cond_idx:
        plt.subplot(2,4,i_cond + 2)
        plot_matrix(A=A_conds[i_cond]/A_conds_max[i_cond], title_size=title_size, vmin=vmin, vmax=vmax, shrink=shrink,\
                    title_str=f'{titles[i_cond]} \n sse={sse_conds[i_cond]:.2e} \n\n {A_label}/|max|, |max|={A_conds_max[i_cond]:.2f}')
        plt.subplot(2,4,i_cond + 6)
        plot_matrix(A=jnp.abs(A_conds[i_cond])/A_conds_max[i_cond], title_str=f'|{A_label}|/|max|', \
                    title_size=title_size, vmin=vmin, vmax=vmax, shrink=shrink)
        
    plt.tight_layout()


def plot_matrices_coeff_grid(A_ref=None, A_ref_label='A_ref', A_conds=None, A_label='A', cond_idx=None, \
                             sse_conds=None, A_triu_idx=None, A_tril_idx=None, titles=None):
    ''' Plots Matrix A_ref coefficients, versus matrices A_conds coefficients estimated from different model configurations''' 
    
    plt.figure(figsize=(17,8))
    
    A_conds_max = [ jnp.abs(A_conds[i_cond]).max() for i_cond in cond_idx ]
    for i_cond in cond_idx: # range(nconds):
        # Extracting Matrices's coefficients from tensor of A_conds values
        A_cond = A_conds[i_cond] 
    
        # Plotting A_ref vs A_cond (non-null) coefficients
        plt.subplot(2,3,i_cond+1)
        y = jnp.concatenate([ A_ref[A_triu_idx],  A_ref[A_tril_idx]])
        x = jnp.concatenate([A_cond[A_triu_idx], A_cond[A_tril_idx]])/A_conds_max[i_cond]
        plt.plot(x,y,'o')
        plt.title(f'{titles[i_cond]} \n sse={sse_conds[i_cond]:.2e}', fontsize=14)
        plt.xlabel(f'{A_label}/|max|, |max|={A_conds_max[i_cond]:.2f}', fontsize=10)
        plt.xlim(-1,1)
        if i_cond==0:
            plt.ylabel(A_ref_label)
            
        # Plotting |A_ref| vs |A_cond| (non-null) coefficients
        plt.subplot(2,3,i_cond+4)
        y = jnp.concatenate([jnp.abs( A_ref[A_triu_idx]),  jnp.abs( A_ref[A_tril_idx])])
        x = jnp.concatenate([jnp.abs(A_cond[A_triu_idx]), jnp.abs(A_cond[A_tril_idx])])/A_conds_max[i_cond]
        plt.plot(x,y,'o')
        corr, p = pearsonr(x,y)
        plt.text(f'r = {corr: .3f}, p = {p: .3f}')
        plt.xlabel(f'|{A_label}|/|max|', fontsize=10)
        if i_cond==0:
            plt.ylabel(f'|{A_ref_label}|')
 

    plt.tight_layout()
    