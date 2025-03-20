import numpy
import statsmodels
from statsmodels.tsa.stattools import adfuller

# Check for stationarity of the time-series data
# We will look for p-value. In case, p-value is less than 0.05, 
# the time series data can said to have stationarity
# Source: https://vitalflux.com/autoregressive-ar-models-with-python-examples/       
def stats_trials(data=None):

    '''Plots percent of stationary trials in each ROI'''

    num_trials = data.shape[0]
    num_ROIs = data.shape[1]

    n_stat = numpy.empty(num_ROIs, dtype=object)
    adf_stats = numpy.empty((num_trials, num_ROIs), dtype=object)
    adf_pvals = numpy.empty((num_trials, num_ROIs), dtype=object)
    for ROI in range(num_ROIs):

        for i in range(num_trials): # Trials

            # Augmented Dickey Fuller test    
            df_stationarityTest = adfuller(data[i,ROI], autolag='AIC')
            # Check the p-value
            adf_stats[i,ROI], adf_pvals[i,ROI] = df_stationarityTest[0], df_stationarityTest[1] 

        n_stat[ROI] = [p<0.05 for p in adf_pvals[:,ROI]].count(True)/num_trials*100

    return (adf_stats, adf_pvals, n_stat)
    
