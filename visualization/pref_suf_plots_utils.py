import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import torch

def create_dataframes(prefix_dicts, suffix_dicts, string_list_models):
    """
    Create dataframes for average Damerau-Levenshtein similarity (DLS) 
    and Mean Absolute Error (MAE) based on prefix and suffix lengths from 
    N models (N being the number of models for which the results should 
    be analyzed, and hence also the number of dictionaries in the 
    `prefix_dicts` and `suffix_dicts` lists, as well as the number of 
    strings in the `string_list_models` list. 

    The order of the prefix and suffix dictionaries contained within the 
    `prefix_dicts` and `suffix_dicts` lists respectively, as well as the 
    order of the strings in the `string_list_models`, should match. 

    Parameters
    ----------
    prefix_dicts : list of dict
        List of dictionaries containing results aggregated over prefix 
        lengths for different models. Each dictionary should have keys as 
        integer prefix lengths and values as lists of three elements: 
        [average DLS, average MAE in minutes, total instance count].
    suffix_dicts : list of dict
        List of dictionaries containing results aggregated over suffix 
        lengths for different models. Each dictionary should have keys as 
        integer suffix lengths and values as lists of three elements: 
        [average DLS, average MAE in minutes, total instance count].
    string_list_models : list of str 
        List of N model names, with the order of the model names 
        corresponding to the order in which the prefix and suffix 
        dictionaries (`prefix_dicts` and `suffix_dicts`) are sorted. 

    Returns
    -------
    df_prefix_dls : pd.DataFrame
        DataFrame containing prefix lengths, instance counts, and average 
        DLS for each model.
    df_prefix_mae : pd.DataFrame
        DataFrame containing prefix lengths, instance counts, and average 
        MAE for each model.
    df_suffix_dls : pd.DataFrame
        DataFrame containing suffix lengths, instance counts, and average 
        DLS for each model.
    df_suffix_mae : pd.DataFrame
        DataFrame containing suffix lengths, instance counts, and average 
        MAE for each model.
    """
    prefix_lengths = sorted(prefix_dicts[0].keys())
    suffix_lengths = sorted(suffix_dicts[0].keys())
    
    prefix_instance_counts = [prefix_dicts[0][k][2] for k in prefix_lengths]
    suffix_instance_counts = [suffix_dicts[0][k][2] for k in suffix_lengths]
    
    prefix_dls = {f'{string_list_models[i]}_dls': [d[k][0] for k in prefix_lengths] for i, d in enumerate(prefix_dicts)}
    prefix_mae = {f'{string_list_models[i]}_mae': [d[k][1] for k in prefix_lengths] for i, d in enumerate(prefix_dicts)}
    
    suffix_dls = {f'{string_list_models[i]}_dls': [d[k][0] for k in suffix_lengths] for i, d in enumerate(suffix_dicts)}
    suffix_mae = {f'{string_list_models[i]}_mae': [d[k][1] for k in suffix_lengths] for i, d in enumerate(suffix_dicts)}
    
    df_prefix_dls = pd.DataFrame({
        'prefix_length': prefix_lengths,
        'instance_count': prefix_instance_counts,
        **prefix_dls
    })
    
    df_prefix_mae = pd.DataFrame({
        'prefix_length': prefix_lengths,
        'instance_count': prefix_instance_counts,
        **prefix_mae
    })
    
    df_suffix_dls = pd.DataFrame({
        'suffix_length': suffix_lengths,
        'instance_count': suffix_instance_counts,
        **suffix_dls
    })
    
    df_suffix_mae = pd.DataFrame({
        'suffix_length': suffix_lengths,
        'instance_count': suffix_instance_counts,
        **suffix_mae
    })
    
    return df_prefix_dls, df_prefix_mae, df_suffix_dls, df_suffix_mae



def create_plots_log(pref_suf_dfs, 
                     configs, 
                     log_name, 
                     include_legend,
                     time_unit='minutes'):
    """Create four plots:

    #. Average Damerau-Levenstein similarity over the prefix lengths for 
       each of the models (configurations). 

    #. Average MAE RRT over the prefix lengths for 
       each of the models (configurations). 

    #. Average Damerau-Levenstein similarity over the suffix lengths for 
       each of the models (configurations). 

    #. Average MAE RRT over the suffix lengths for 
       each of the models (configurations). 

    Parameters
    ----------
    pref_suf_dfs : list of pd.DataFrame
        Four dataframes:

        #. DataFrame containing prefix lengths, instance counts, and 
           average DLS over each prefix length, for each model.

        #. DataFrame containing prefix lengths, instance counts, and 
           average MAE RRT over each prefix length, for each model.

        #. DataFrame containing suffix lengths, instance counts, and 
           average DLS over each suffix length, for each model.

        #. DataFrame containing suffix lengths, instance counts, and 
           average MAE RRT over each suffix length, for each model.

    configs : list of str
        List of N model names, with the order of the model names 
        corresponding to the order in which the prefix and suffix 
        dictionaries (`prefix_dicts` and `suffix_dicts`) are sorted. 
        Make sure to name them according to one of the keys in the 
        `config_string` dictionary defined below. This should also 
        be accounted for in the `string_list_models` list of the 
        `create_dataframes()` function. 
    log_name : str
        Name of the model for which the plots are created. 
    include_legend : bool 
        If `True`, legend for the different configs will be included. 
    time_unit : str 
        The time unit in which the MAE is displayed. 
    """
    config_string = {'SEP_LSTM' : 'SEP-LSTM', 
                    'CRTP_LSTM' : 'CRTP-LSTM (NDA)', 
                    'CRTP_LSTM_DA' : 'CRTP-LSTM', 
                    'ED_LSTM' : 'ED-LSTM', 
                    'SuTraN' : 'SuTraN (NDA)', 
                    'SuTraN_DA' : 'SuTraN'}
    config_styles = {
        'CRTP_LSTM': ('#9467bd', '--'),
        'CRTP_LSTM_DA': ('#9467bd', '-'),
        # 'SuTraN': ('#1f77b4', '--'),
        # 'SuTraN_DA': ('#1f77b4', '-'),
        'SuTraN': ('#2ca02c', '--'),
        'SuTraN_DA': ('#2ca02c', '-'),
        'SEP_LSTM': ('#d62728', '--'),
        'ED_LSTM': ('#ff7f0e', '--'),
    }

    fontsize = 22
    labelsize = 16
    fig, ax = plt.subplots(2, 2, figsize=(20, 14))
    fig.subplots_adjust(hspace=0.6, wspace=0.4, top=0.85)

    print(log_name)
    
    # Looping over the models / configurations 
    for config in configs:
        # Retrieving column names for DLS and RRT for that config 
        dls_col = config + '_dls'
        mae_col = config + '_mae'

        df_1 = pref_suf_dfs[0]
        df_2 = pref_suf_dfs[1]
        df_3 = pref_suf_dfs[2]
        df_4 = pref_suf_dfs[3]
        color, linestyle = config_styles[config]
        label = config_string[config]
        # marker = marker_mappings[config]
        ax[0, 0].plot(df_1['prefix_length'], df_1[dls_col], label=label, color=color, linestyle=linestyle)
        ax[0, 0].set_title('Average DLS over the prefix lengths', fontsize=fontsize)

        ax[0, 1].plot(df_2['prefix_length'], df_2[mae_col], label=label, color=color, linestyle=linestyle)
        ax[0, 1].set_title('Average MAE ({}) over the prefix lengths'.format(time_unit), fontsize=fontsize)

        ax[1, 0].plot(df_3['suffix_length'], df_3[dls_col], label=label, color=color, linestyle=linestyle)
        ax[1, 0].set_title('Average DLS over the suffix lengths', fontsize=fontsize)

        ax[1, 1].plot(df_4['suffix_length'], df_4[mae_col], label=label, color=color, linestyle=linestyle)
        ax[1, 1].set_title('Average MAE ({}) over the suffix lengths'.format(time_unit), fontsize=fontsize)
    ax_1 = ax[0, 0].twinx()
    # ax[0, 0].set_title(title)
    ax[0, 0].set_ylabel('DLS', fontsize=fontsize)
    ax[0, 0].set_xlabel('Prefix Length', fontsize=fontsize)
    ax_1.plot(df_1['prefix_length'], df_1['instance_count'], label='Number of Instances', color='grey', linestyle='--')
    ax_1.fill_between(df_1['prefix_length'], 0, df_1['instance_count'], color='grey', alpha=0.3, zorder=0)
    ax_1.set_ylabel("Instances", color='grey', fontsize=fontsize)
    ax_1.tick_params('y', colors='grey', labelsize=labelsize)

    ax_2 = ax[0, 1].twinx()
    # ax[0, 1].set_title(title)
    ax[0, 1].set_ylabel('MAE Remaining Time ({})'.format(time_unit), fontsize=fontsize)
    ax[0, 1].set_xlabel('Prefix Length', fontsize=fontsize)
    ax_2.plot(df_2['prefix_length'], df_2['instance_count'], label='Number of Instances', color='grey', linestyle='--')
    ax_2.fill_between(df_2['prefix_length'], 0, df_2['instance_count'], color='grey', alpha=0.3, zorder=0)
    ax_2.set_ylabel("Instances", color='grey', fontsize=fontsize)
    ax_2.tick_params('y', colors='grey', labelsize=labelsize)

    ax_3 = ax[1, 0].twinx()
    # ax[1, 0].set_title(title)
    ax[1, 0].set_ylabel('DLS', fontsize=fontsize)
    ax[1, 0].set_xlabel('Suffix Length', fontsize=fontsize)
    ax_3.plot(df_3['suffix_length'], df_3['instance_count'], label='Number of Instances', color='grey', linestyle='--')
    ax_3.fill_between(df_3['suffix_length'], 0, df_3['instance_count'], color='grey', alpha=0.3, zorder=0)
    ax_3.set_ylabel("Instances", color='grey', fontsize=fontsize)
    ax_3.tick_params('y', colors='grey', labelsize=labelsize)

    ax_4 = ax[1, 1].twinx()
    # ax[1, 1].set_title(title)
    ax[1, 1].set_ylabel('MAE Remaining Time ({})'.format(time_unit), fontsize=fontsize)
    ax[1, 1].set_xlabel('Suffix Length', fontsize=fontsize)
    ax_4.plot(df_4['suffix_length'], df_4['instance_count'], label='Number of Instances', color='grey', linestyle='--')
    ax_4.fill_between(df_4['suffix_length'], 0, df_4['instance_count'], color='grey', alpha=0.3, zorder=0)
    ax_4.set_ylabel("Instances", color='grey', fontsize=fontsize)
    ax_4.tick_params('y', colors='grey', labelsize=labelsize)

    for ax_row in ax:
        for axis in ax_row:
            axis.tick_params(axis='both', which='major', labelsize=labelsize) 

    if include_legend:
        # Collect handles and labels for the figure's legend
        handles, labels = ax[0, 0].get_legend_handles_labels()

        # Create a single, common legend at the bottom of the figure
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.0002), ncol=6)

    fig.tight_layout()
    plt.show()

################################################################################
# Functionality for Case Length De-noising
################################################################################

def create_pref_suf_dicts(pref_len_tensor, 
                          suf_len_tensor, 
                          window_size, 
                          dam_lev_similarity, 
                          MAE_rrt_minutes):
    """Create the prefix and suffix dictionary. This function can be used 
    for generating the two dictionaries for a certain model after 
    having it evaluated on a test set, or generating the new prefix and 
    suffix dictionaries after having performed Case Length De-noising. 
    In the latter case, the four input tensors should have been subsetted 
    by the de-noising procedure contained within the 
    `discard_noisy_cases()` function. 

    Parameters
    ----------
    pref_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the prefix length 
        (in terms of number of events contained within the sequence of 
        prefix events). 
    suf_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the suffix length 
        of the ground truth suffix (in terms of number of events 
        contained within the sequence of suffix events). 
    window_size : int
        The maximum sequence length (both for the prefixes and suffixes) 
        corresponding to the event log at hand. Can be found by querying 
        the maximum integer value contained within the `pref_len_tensor` 
        if needed. 
    dam_lev_similarity : torch.Tensor
        torch.float32 tensor of shape (T,) containing the normalized 
        Damerau-Levenshtein Similarity score for each of the T test set 
        predictions. 
    MAE_rrt_minutes : torch.Tensor
        torch.float32 tensor of shape (T,) containing the Mean Absolute 
        Error in minutes for each of the T test set predictions. 

    Returns
    -------
    results_dict_pref : dict of list
        Dictionary containing the results aggregated over prefix 
        lengths. Should have keys as integer prefix lengths and values as 
        lists of three elements: 
        [average DLS, average MAE in minutes, total instance count].
    results_dict_suf : dict of list
        Dictionary containing the results aggregated over suffix 
        lengths. Should have keys as integer suffix lengths and values as 
        lists of three elements: 
        [average DLS, average MAE in minutes, total instance count].
    """
    # Making dictionaries of the results for over both prefix and suff length. 
    results_dict_pref = {}
    for i in range(1, window_size+1):
        bool_idx = pref_len_tensor==i
        dam_levs = dam_lev_similarity[bool_idx].clone()
        MAE_rrt_i = MAE_rrt_minutes[bool_idx].clone()
        num_inst = dam_levs.shape[0]
        if num_inst > 0:
            avg_dl = (torch.sum(dam_levs) / num_inst).item()
            avg_mae = (torch.sum(MAE_rrt_i) / num_inst).item()
            results_i = [avg_dl, avg_mae, num_inst]
            results_dict_pref[i] = results_i
    results_dict_suf = {}
    for i in range(1, window_size+1):
        bool_idx = suf_len_tensor==i
        dam_levs = dam_lev_similarity[bool_idx].clone()
        MAE_rrt_i = MAE_rrt_minutes[bool_idx].clone()
        num_inst = dam_levs.shape[0]
        if num_inst > 0:
            avg_dl = (torch.sum(dam_levs) / num_inst).item()
            avg_mae = (torch.sum(MAE_rrt_i) / num_inst).item()
            results_i = [avg_dl, avg_mae, num_inst]
            results_dict_suf[i] = results_i
    
    return results_dict_pref, results_dict_suf



def get_corrected_distribution_tensor(pref_len_tensor, suf_len_tensor):
    """Derive a tensor representing the original distribution of case 
    lengths within the test set. I.e. each complete case assigned to the 
    test set, delivers multiple prefix-suffix pairs aka test instances. 
    This method generates a tensor containing the case length (in number 
    of events) distribution from the original set of cases that were 
    used to derive the test set instances, instead of the case length 
    distribution over the ultimately generated test set instances, 
    since the latter would inherently be biased towards the longest 
    case lengths (because a case of length X, will be split up into 
    X instances / prefix-suffix pairs). 

    Parameters
    ----------
    pref_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the prefix length 
        (in terms of number of events contained within the sequence of 
        prefix events). 
    suf_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the suffix length 
        of the ground truth suffix (in terms of number of events 
        contained within the sequence of suffix events). 
    """
    # Deriving a tensor containing the case length for each of the 
    # instances 
    case_len = pref_len_tensor + suf_len_tensor - 1 

    # Get list of unique case lengths present in the test set 
    counts_clen = torch.bincount(case_len)
    unique_caselengths = []
    for integer_value, count in enumerate(counts_clen):
        if count > 0: 
            unique_caselengths.append(integer_value)

    # Initializing dictionary that will store for each case length, the 
    # original amount of cases that were used to create prefix-suffix 
    # pairs pertaining to that total case length. 
    unique_cases_dict = {}
    for uni_len in unique_caselengths:
        # Boolean index. True if prefix-suffix pair corresponding to that 
        # index has case lenght equal to `uni_len` 
        bool_idx = case_len == uni_len

        # Selecting only those prefix lengths of which the indices correspond 
        # to prefix-suffix pairs pertaining to a case of length `uni_len`
        pref_len_subset = pref_len_tensor[bool_idx]

        # Out of that subset, select only the prefix lengths equal 
        # to `uni_len` 
        bool_idx_unilen = pref_len_subset == uni_len 
        # Compute number of unique cases of length `uni_len` used 
        # to generate prefix-suffix pairs
        num_cases_unilen = torch.sum(bool_idx_unilen).item() # integer 

        # Store amount to dictionary 
        unique_cases_dict[uni_len] = num_cases_unilen

    # Deriving list representing original case length distribution 
    # `clen_corr`

    unique_caselens = list(unique_cases_dict.keys())
    # list of number of original cases used to derive 
    # prefix-suffix pairs pertaining to a certain case length
    number_ogcases = list(unique_cases_dict.values())
    clen_corr = []
    for i in range(len(unique_caselens)):
        # retrieve unique case length 
        u_len = unique_caselens[i]

        # retrieve number of original cases of that case length 
        num_og = number_ogcases[i]

        # Add `u_len` `num_og` times 
        clen_corr += [u_len for _ in range(num_og)]
    
    # Making tensor out of it: 
    clen_corr = torch.tensor(data=clen_corr, dtype=torch.float32)

    return clen_corr



def get_subset_bool(pref_len_tensor, 
                    suf_len_tensor, 
                    discard_fraction_lb, 
                    discard_fraction_ub, 
                    return_bounds=False):
    """Create boolean torch.Tensor of shape (T,), with T being the 
    number of test set instances (i.e., prefix-suffix pairs), evaluating 
    to True for those instances of which the case length (in number of 
    events), falls within the range of the percentiles determined by 
    the `discard_fraction_lb` and `discard_fraction_ub` integers. 

    Parameters
    ----------
    pref_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the prefix length 
        (in terms of number of events contained within the sequence of 
        prefix events). 
    suf_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the suffix length 
        of the ground truth suffix (in terms of number of events 
        contained within the sequence of suffix events). 
    discard_fraction_lb : int
        Integer representing the minimum percentile of the case length 
        for which instances are still retained. be an integer between 0 
        and 100, inclusive. 
    discard_fraction_ub : int
        Integer representing the maximum percentile of the case length 
        for which instances are still retained. be an integer between 0 
        and 100, inclusive. 
    return_bounds : bool 
        Whether or not the function should also return the minimum and 
        maximum percentile determined by `discard_fraction_lb` and 
        `discard_fraction_ub`.
    """
    case_len = pref_len_tensor + suf_len_tensor - 1 
    
    # Get tensor mimicing original distribution case length test cases 
    clen_corr = get_corrected_distribution_tensor(pref_len_tensor, suf_len_tensor)

    # Create percentiles tensor corrected distribution case length 
    percentiles = torch.arange(0, 101) / 100.0

    # Compute percentiles 
    corr_distr_percentiles = torch.quantile(clen_corr, percentiles) # shape (101, )

    # Deriving lower bound and upper bound case length for which instances are still 
    # retained 
    lb_case_length = corr_distr_percentiles[discard_fraction_lb].item()
    lb_case_length = int(lb_case_length)
    ub_case_length = corr_distr_percentiles[discard_fraction_ub].item()
    ub_case_length = int(ub_case_length)
    
    retain_bool = (case_len >= lb_case_length) & (case_len <= ub_case_length)
    if return_bounds:
        return retain_bool, lb_case_length, ub_case_length
    else:
        return retain_bool