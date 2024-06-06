"""Module containing all the functionality to create plots displaying the 
average Damerau-Levenshtein Similarity (normalized), as well as the 
average Mean Absolute Error (MAE) (in minutes) for the remaining runtime 
prediction target, in function of the prefix and suffix length of the 
test set instances / prefix-suffix pairs. 

This entails:
- Case-Length De-noising: functionality to de-noise the prefix suffix 
plots by subsetting the results for only those cases of which the case 
length falls within a specified range, thereby removing noise in the 
plots induced by irregular test cases that can be considered outliers.
- functionality to create the plots displaying the evolution of 
prediction accuracy metrics (Damerau-Levenshtein similarity and Mean 
Absolute Error (MAE) in minutes) over varying prefix and suffix lengths 
for activity suffix and remaining runtime predictions respectively, with 
each plot showing all models' performance trends for a specific metric. 
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pref_suf_plots_utils import create_dataframes, create_plots_log, create_pref_suf_dicts, get_subset_bool
import torch


def create_plots(prefix_dicts, 
                 suffix_dicts, 
                 string_list_models, 
                 log_name, 
                 include_legend,
                 time_unit='minutes'):
    """Generates the plots displaying the evolution of 
    prediction accuracy metrics (Damerau-Levenshtein similarity (activity 
    suffix prediction) and Mean Absolute Error (MAE) in minutes 
    (remaining runtime prediction)) over varying prefix and suffix 
    lengths for activity suffix and remaining runtime predictions 
    respectively, with each plot showing all models' performance trends 
    for a specific metric.

    Parameters
    ----------
    prefix_dicts : list of dict
        List of dictionaries containing results aggregated over prefix 
        lengths for different models. Each dictionary should have keys as 
        integer prefix lengths and values as lists of three elements: 
        [average DLS, average MAE in minutes, total instance count]. Each 
        of the N dictionaries pertains to the results of one of the N 
        models evaluated on the test set corresponding to the event log 
        specified by the `log_name` string. The order in which these 
        dictionaries are specified, should correspond to the order of 
        the dictionaries contained within the `suffix_dicts` list, as 
        well as the order in which the model strings are defined within 
        the `string_list_models` list. 
    suffix_dicts : list of dict
        List of dictionaries containing results aggregated over suffix 
        lengths for different models. Each dictionary should have keys as 
        integer suffix lengths and values as lists of three elements: 
        [average DLS, average MAE in minutes, total instance count]. Each 
        of the N dictionaries pertains to the results of one of the N 
        models evaluated on the test set corresponding to the event log 
        specified by the `log_name` string. The order in which these 
        dictionaries are specified, should correspond to the order of 
        the dictionaries contained within the `prefix_dicts` list, as 
        well as the order in which the model strings are defined within 
        the `string_list_models` list. 
    string_list_models : list of str 
        List of N model names, with the order of the model names 
        corresponding to the order in which the prefix and suffix 
        dictionaries (`prefix_dicts` and `suffix_dicts`) are sorted. 
    log_name : str 
        Name of the event log for which the test set results are plotted. 
    include_legend : bool 
        If `True`, legend for the different configs will be included. 
    time_unit : str 
        Time unit in which the MAE values are expressed. 'minutes' by 
        default. 
    """
    # Create dataframes 
    df_prefix_dls, df_prefix_mae, df_suffix_dls, df_suffix_mae = create_dataframes(prefix_dicts, 
                                                                                   suffix_dicts, 
                                                                                   string_list_models=string_list_models)
    
    # Generate the four plots 
    create_plots_log(pref_suf_dfs=[df_prefix_dls, df_prefix_mae, df_suffix_dls, df_suffix_mae], 
                     configs=string_list_models, 
                     log_name=log_name,
                     include_legend=include_legend, 
                     time_unit=time_unit)


################################################################################
# Functionality for Case Length De-noising
################################################################################




def discard_noisy_cases(pref_len_tensor, 
                        suf_len_tensor, 
                        dam_lev_similarity_list, 
                        MAE_rrt_minutes_list, 
                        discard_fraction_lb, 
                        discard_fraction_ub, 
                        window_size, 
                        return_bounds=False):
    """Case-Length De-noising procedure: functionality to de-noise the 
    prefix suffix plots by subsetting the results for only those cases of 
    which the case length falls within a specified range, thereby 
    removing noise in the plots induced by irregular test cases that can 
    be considered outliers. This is done for the results of all models 
    for predictions on one particular event log. 

    Parameters
    ----------
    pref_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the prefix length 
        (in terms of number of events contained within the sequence of 
        prefix events). Given a certain event log, these tensors are 
        identical for all models. 
    suf_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the suffix length 
        of the ground truth suffix (in terms of number of events 
        contained within the sequence of suffix events). Given a certain 
        event log, these tensors are identical for all models. 
    dam_lev_similarity_list : list of torch.Tensor
        List of torch.float32 tensors of shape (T,). Each tensor pertains 
        to one of the models ran on the test set of the event log under 
        consideration, and contains the normalized 
        Damerau-Levenshtein Similarity score for each of the T test set 
        predictions. The order of the tensors, in terms of the 
        model for which it contains the DLSs, should be the same order 
        as the one in the `MAE_rrt_minutes_list` list. 
    MAE_rrt_minutes_list : list of torch.Tensor
        List of torch.float32 tensors of shape (T,). Each tensor pertains 
        to one of the models ran on the test set of the event log under 
        consideration, and contains the Mean Absolute Error in minutes 
        for the Remaining Runtime (rrt) predictions made for each of the 
        T test set instances. The order of the tensors, in terms of the 
        model for which it contains the MAEs, should be the same order 
        as the one in the `dam_lev_similarity_list` list. 
    discard_fraction_lb : int
        Integer representing the minimum percentile of the case length 
        for which instances are still retained. be an integer between 0 
        and 100, inclusive. 
    discard_fraction_ub : int
        Integer representing the maximum percentile of the case length 
        for which instances are still retained. be an integer between 0 
        and 100, inclusive. 
    window_size : int
        The maximum sequence length (both for the prefixes and suffixes) 
        corresponding to the event log at hand. Can be found by querying 
        the maximum integer value contained within the `pref_len_tensor` 
        if needed. 
    return_bounds : bool 
        Whether or not the function should also return the minimum and 
        maximum percentile determined by `discard_fraction_lb` and 
        `discard_fraction_ub`.
    """
    # Retrieve subset bool tensor, evaluating to True for instances 
    # that fall within the bounds and hence should be retained. 
    # Let T* be the nubmer of instances, out of the T original 
    # test set instances (aka prefix-suffix pairs), for which 
    # the `subset_bool` evaluates to True. 
    if return_bounds:
        subset_bool, lb_case_length, ub_case_length = get_subset_bool(pref_len_tensor, 
                                                                      suf_len_tensor, 
                                                                      discard_fraction_lb, 
                                                                      discard_fraction_ub, 
                                                                      return_bounds) # shape (T, )
    else: 
        subset_bool = get_subset_bool(pref_len_tensor, 
                                     suf_len_tensor, 
                                     discard_fraction_lb, 
                                     discard_fraction_ub, 
                                     return_bounds) # shape (T, )
    
    # Initialize lists for the de-noised metrics 
    dam_lev_sim_deno_list = []
    MAE_rrt_min_deno_list = []
    
    # Initializing lists to be filled with de-noised prefix and suffix 
    # dictionaries. 
    prefix_dicts_deno = []
    suffix_dicts_deno = []

    # Creating de-noised pref and suf len tensors 
    pref_len_deno = pref_len_tensor[subset_bool].clone() # shape (T*, )
    suf_len_deno = suf_len_tensor[subset_bool].clone() # shape (T*, )

    # Subsetting the metric tensors for all models, as well as creating 
    # de-noiesd prefix and suffix dictionaries. 
    for i in range(len(dam_lev_similarity_list)): 
        dam_lev_sim_deno = dam_lev_similarity_list[i][subset_bool].clone() # shape (T*, )
        dam_lev_sim_deno_list.append(dam_lev_sim_deno)
        MAE_rrt_min_deno = MAE_rrt_minutes_list[i][subset_bool].clone() # shape (T*, )
        MAE_rrt_min_deno_list.append(MAE_rrt_min_deno)

        results_dict_pref, results_dict_suf = create_pref_suf_dicts(pref_len_deno, 
                                                                    suf_len_deno, 
                                                                    window_size, 
                                                                    dam_lev_sim_deno, 
                                                                    MAE_rrt_min_deno)
        
        prefix_dicts_deno.append(results_dict_pref)
        suffix_dicts_deno.append(results_dict_suf)
    
    # Gathering the returned valeus 

    #   De-noised prefix and suffix length tensors. 
    return_list = [pref_len_deno, suf_len_deno]

    #   Lists of denoised dam lev and MAE rrt metrics 
    return_list += [dam_lev_sim_deno_list, MAE_rrt_min_deno_list]

    #   Lists of de-noised prefix dicts and suffix dicts 
    return_list += [prefix_dicts_deno, suffix_dicts_deno]

    if return_bounds:
        return_list += [lb_case_length, ub_case_length]

    return return_list


def create_denoised_plots(pref_len_tensor, 
                          suf_len_tensor, 
                          dam_lev_similarity_list, 
                          MAE_rrt_list, 
                          discard_fraction_lb, 
                          discard_fraction_ub, 
                          window_size, 
                          string_list_models, 
                          log_name, 
                          return_bounds=False, 
                          include_legend=True,
                          time_unit='minutes'): 
    """
    Combines the case-length de-noising procedure with plot generation 
    to create denoised plots for predictive performance metrics.

    This function first applies the `discard_noisy_cases()` method to 
    subset the results for instances derived from cases whose lengths 
    fall within a specified 
    range, thereby removing noise caused by irregular and outlier test 
    cases. It then calls the `create_plots()` method to generate plots 
    that display the evolution of prediction accuracy metrics over 
    varying prefix and suffix lengths. The generated plots illustrate 
    the performance trends of different models for specific metrics, 
    such as Damerau-Levenshtein similarity for activity suffix 
    prediction and Mean Absolute Error (MAE) in minutes for remaining 
    runtime prediction.

    The combined functionality ensures that the plots provide a clearer 
    and more accurate representation of the models' performance by 
    removing the noise induced by outlier cases.

    Additionally, a tuple containing the subsetted results tensors, 
    including the subsetted versions of the input parameters, 
    and dictionaries (after 'de-noising'), is returned as well. 

    Parameters
    ----------
    pref_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the prefix length 
        (in terms of number of events contained within the sequence of 
        prefix events). Given a certain event log, these tensors are 
        identical for all models. 
    suf_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the suffix length 
        of the ground truth suffix (in terms of number of events 
        contained within the sequence of suffix events). Given a certain 
        event log, these tensors are identical for all models. 
    dam_lev_similarity_list : list of torch.Tensor
        List of torch.float32 tensors of shape (T,). Each tensor pertains 
        to one of the models ran on the test set of the event log under 
        consideration, and contains the normalized 
        Damerau-Levenshtein Similarity score for each of the T test set 
        predictions. The order of the tensors, in terms of the 
        model for which it contains the DLSs, should be the same order 
        as the one in the `MAE_rrt_list` list. 
    MAE_rrt_list : list of torch.Tensor
        List of torch.float32 tensors of shape (T,). Each tensor pertains 
        to one of the models ran on the test set of the event log under 
        consideration, and contains the Mean Absolute Error 
        for the Remaining Runtime (rrt) predictions made for each of the 
        T test set instances. The order of the tensors, in terms of the 
        model for which it contains the MAEs, should be the same order 
        as the one in the `dam_lev_similarity_list` list. The time unit 
        in which these MAEs are expressed, should coincide with the time 
        unit specified for the `time_unit` parameter, which is `minutes` 
        by default. 
    discard_fraction_lb : int
        Integer representing the minimum percentile of the case length 
        for which instances are still retained. be an integer between 0 
        and 100, inclusive. 
    discard_fraction_ub : int
        Integer representing the maximum percentile of the case length 
        for which instances are still retained. be an integer between 0 
        and 100, inclusive. 
    window_size : int
        The maximum sequence length (both for the prefixes and suffixes) 
        corresponding to the event log at hand. Can be found by querying 
        the maximum integer value contained within the `pref_len_tensor` 
        if needed. 
    string_list_models : list of str 
        List of N model names, with the order of the model names 
        corresponding to the order in which the tensors in the 
        `dam_lev_similarity_list` and `MAE_rrt_list` lists are 
        stored. 
    log_name : str 
        Name of the event log for which the test set results are plotted. 
        Just prints the log name before outputting the plots. Used for 
        obtained the plots for all event logs at the same time, in an 
        external loop. 
    return_bounds : bool, optional
        Whether or not the function should also return the minimum and 
        maximum percentile determined by `discard_fraction_lb` and 
        `discard_fraction_ub`.
    include_legend : bool, optional
        If `True`, legend for the different configs will be included. 
        True by default. 
    time_unit : str, optional 
        Time unit in which the MAE values are expressed. 'minutes' by 
        default. This time unit will be displayed on the generated 
        plots. 

    Returns
    -------
    de_noised_results : tuple 
        Tuple containing subsetted results tensors and dictionaries after 
        de-noising. This includes: 

        * `pref_len_deno` : torch.Tensor located at 
          `de_noised_results[0]`. Contains subsetted version of the 
          `pref_len_tensor` parameter fed to the function. Shape 
          (N_sub,), with `N_sub` being the number of instances 
          retained after the de-noising procedure. 
           
        
        * `suf_len_deno` : torch.Tensor located at 
          `de_noised_results[1]`. Contains subsetted version of the 
          `suf_len_tensor` parameter fed to the function. Shape 
          (N_sub,), with `N_sub` being the number of instances 
          retained after the de-noising procedure. 

        * `dam_lev_sim_deno_list` : list of torch.Tensor located at 
          `de_noised_results[2]`. Contains subsetted version of the 
          tensors contained within the `dam_lev_similarity_list` list 
          fed to the function. Each tensor, pertaining to a certain 
          model, has shape (N_sub,), with `N_sub` being the number of 
          instances retained after the de-noising procedure. 
        
        * `MAE_rrt_deno_list` : list of torch.Tensor located at 
          `de_noised_results[3]`. Contains subsetted version of the 
          tensors contained within the `MAE_rrt_list` list 
          fed to the function. Each tensor, pertaining to a certain 
          model, has shape (N_sub,), with `N_sub` being the number of 
          instances retained after the de-noising procedure. 

        * `prefix_dicts_deno` : list of dict located at 
          `de_noised_results[4]`. Contains the `prefix_dicts`, as 
          specified in the documentation of the `create_plots()` 
          function, but after de-noising. 
        
        * `suffix_dicts_deno` : list of dict located at 
          `de_noised_results[5]`. Contains the `suffix_dicts`, as 
          specified in the documentation of the `create_plots()` 
          function, but after de-noising. 

        * if `return_bounds=True`:

          * `lb_case_length`: integer located at  `de_noised_results[6]`. 
            Represents the lower bound of the case length 
            distribution, pertaining to the `discard_fraction_lb`'th 
            percentile. After de-noising, only the metrics pertaining 
            to instances derived from cases with 
            case length `∈ [lb_case_length, ub_case_length]` are retained.

          * `ub_case_length`: integer located at  `de_noised_results[7]`. 
            Represents the upper bound of the case length 
            distribution, pertaining to the `discard_fraction_ub`'th 
            percentile. After de-noising, only the metrics pertaining 
            to instances derived from cases with 
            case length `∈ [lb_case_length, ub_case_length]` are retained. 
    """

    # De-noise all metrics tensors for all models, by discarding 
    # scores pertaining to 'outlying' instances, as defined by 
    # the values specified for `discard_fraction_lb` and 
    # `discard_fraction_ub`. 
    de_noised_results = discard_noisy_cases(pref_len_tensor, 
                                            suf_len_tensor, 
                                            dam_lev_similarity_list, 
                                            MAE_rrt_list, 
                                            discard_fraction_lb, 
                                            discard_fraction_ub, 
                                            window_size, 
                                            return_bounds)
    
    # # Retrieving the elements contained within the `de_noised_results`
    # # list. 
    # pref_len_deno, suf_len_deno = de_noised_results[0:2]

    # # Subsetted / de-noised dam lev and MAE rrt tensors for 
    # # all models. 
    # dam_lev_sim_deno_list, MAE_rrt_deno_list = de_noised_results[2:4]

    # Lists of de-noised prefix and suffix dicts. 
    prefix_dicts_deno, suffix_dicts_deno = de_noised_results[4:6]

    # # the minimum and maximum case lengths determined by the specified 
    # # percentiles. 
    # if return_bounds:
    #     lb_case_length, ub_case_length = de_noised_results[-2:]


    # CREATING THE DE-NOISED PREFIX AND SUFFIX PLOTS 
    create_plots(prefix_dicts_deno, 
                 suffix_dicts_deno, 
                 string_list_models, 
                 log_name, 
                 include_legend=include_legend,
                 time_unit=time_unit)

    return de_noised_results