import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.simplefilter('ignore')

def get_moving_average(arr, window_size, trim_edges=False, return_centers=False):

    n_pts_original = len(arr)
    window_half1 = int(np.floor(window_size/2))
    window_half2 = int(np.ceil(window_size/2))
    
    if trim_edges:
        window_centers = np.arange(window_half1, n_pts_original-window_half2+1)
    else:
        window_centers = np.arange(0,n_pts_original)

    window_starts = [np.maximum(0,ii-window_half1) for ii in window_centers]
    window_stops = [np.minimum(n_pts_original,ii+window_half2) for ii in window_centers]
    inds_list = [np.arange(ww1, ww2) for ww1,ww2 in zip(window_starts, window_stops)]

    smoothed_data = np.array([np.mean(arr[inds]) for inds in inds_list])
    
    if return_centers:       
        return smoothed_data, window_centers
    else:
        return smoothed_data
    
def mixed_effect_model(dat, conds_compare, epoch_range):
   
    """
    Fit a linear mixed effects model comparing two conditions
    (implement using statsmodels.formula.api.mixedlm)
    Use data from just one time window at a time, assuming linearity within each window.
    """
    
    epoch_window_size = len(epoch_range)
    n_conds_use=len(conds_compare)
    n_trials = dat.shape[1]
    
    # first, need to put my array data into a list format
    # (all acc values in a long list, with accompanying labels
    # for the trial number, condition, epoch number)
    dat_list = np.zeros((epoch_window_size*n_conds_use*n_trials,))
    cond_list = np.zeros((epoch_window_size*n_conds_use*n_trials,),dtype=int)
    trial_list = np.zeros((epoch_window_size*n_conds_use*n_trials,),dtype=int)
    epoch_list = np.zeros((epoch_window_size*n_conds_use*n_trials,),dtype=int)

    st=0;
    tc=0; 
    # treating every trial as independent here, not repeated measures
    # bc every trial was random in diff conditions
    for ci, cc in enumerate(conds_compare):
        for tt in range(n_trials):
            inds = np.arange(st, st+epoch_window_size)
            dat_list[inds] = dat[cc,tt,:][epoch_range]
            cond_list[inds] = ci;
            trial_list[inds] = tc;
            tc+=1
            epoch_list[inds] = np.arange(epoch_window_size)
            st+=epoch_window_size
        
    dat_df = pd.DataFrame({'acc': dat_list, 'cond': cond_list, \
                       'trial_num': trial_list, 'epoch_num': epoch_list})
    
    # model the accuracy as a fn of condition and epoch number
    # using "trial" as a grouping factor (random effect)
    # by default it uses a random intercept for each trial grouping
    md = smf.mixedlm(formula='acc ~ C(cond) + epoch_num', data=dat_df, \
                 groups=dat_df['trial_num'])
    # md = mixed_lm("acc ~ cond + epoch_num", dat_df, \
                 # groups=dat_df['trial_num'])
    try:
        mdf = md.fit()
    except:
        print('fit failed to converge')
        return np.nan, np.nan
    # sometimes it prints a warning which is generally ok 
    # but if it doesnt converge, then we can't use result
    if not mdf.converged:
        print('fit failed to converge')
        return np.nan, np.nan
    
    # pull out the coeffs for just the condition effect, which we're interested in
    coeffs = np.array(mdf.summary().tables[1])[:,0]
    coeffs = [float(cc) for cc in coeffs]
    coeff_condition = coeffs[1]
    
    # and p value for condition effect
    pvals = np.array(mdf.summary().tables[1])[:,3]
    pvals = [float(pp) if pp!='' else np.nan for pp in pvals ]
    pval_condition = pvals[1]
    
    return coeff_condition, pval_condition