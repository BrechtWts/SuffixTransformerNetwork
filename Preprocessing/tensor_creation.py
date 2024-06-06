import pandas as pd 
import numpy as np 
# import tensorflow as tf 
import torch
import torch.nn.functional as F
import pickle 
import os 

def generate_prefix_tensors(case_id, 
                            prefix_df, 
                            cardinalities, 
                            cat_cols, 
                            num_cols, 
                            window_size, 
                            str_to_int):
    """_summary_

    Parameters
    ----------
    case_id : _type_
        _description_
    act_label : _type_
        _description_
    prefix_df : _type_
        _description_
    cardinalities : _type_
        _description_
    cat_cols : _type_
        _description_
    num_cols : _type_
        _description_
    window_size : _type_
        _description_
    str_to_int : dict 
        Integer mapping for string instance IDs. 
    """
    # Computing the event idx for each unique prefix. 
    prefix_df['evt_idx'] = prefix_df.groupby([case_id], sort = False).cumcount()

    # Giving each unique prefix a unique integer id. 

    prefix_df['prefix_id_int'] = prefix_df[case_id].map(str_to_int)
    
    # Computing the indices for the tensor scatter updates
    idx = torch.from_numpy(prefix_df[['prefix_id_int', 'evt_idx']].to_numpy()) # (num_rows, 2)
    num_prefs = max(prefix_df['prefix_id_int'])+1 # number of distinct prefixes 

    # Numericals:
    num_nums = len(num_cols) # number of numerical features 
    nums_updates = torch.from_numpy(prefix_df[num_cols].to_numpy()) # shape (num_rows, num_nums)
    nums_updates = nums_updates.type(torch.float32)
    # Initialize 0 tensor 
    num_tens = torch.zeros(size = [num_prefs, window_size, num_nums], dtype = torch.float32) # shape (num_prefs, window_size, num_nums)
    num_tens[idx[:, 0], idx[:,1]] = nums_updates

    # Based on num_tens, the (num_prefs, window_size) - shaped padding mask is derived. 
    # There will always be a valid num_tens tensor, since it will always contain 
    # at least the computed 'ts_start' and 'ts_prev' features. 


    
    # Initialize padding_mask as a boolean tensor, filled with True, of shape (num_prefs, window_size)
    padding_mask = torch.full(size = (num_prefs, window_size), fill_value = True) # (num_prefs, window_size)
    # Replace the True values with False on the indices pertaining to actual (and not padded) events. 
    padding_mask[idx[:, 0], idx[:, 1]] = False # (num_prefs, window_size)

    prefix_tensors = []
    for cat_col in cat_cols:
        cat_updates = torch.from_numpy(prefix_df[cat_col].to_numpy()) # shape (num_rows, )
        cat_updates = cat_updates.to(torch.int64)
        # Initialize 0 tensor 
        cat_tens = torch.zeros(size = (num_prefs, window_size), dtype = torch.int64) # (num_prefs, window_size)
        # Update zero tensor 
        cat_tens[idx[:, 0], idx[:, 1]] = cat_updates + 1 # (num_prefs, window_size)
        prefix_tensors.append(cat_tens)
    
    prefix_tensors.append(num_tens)

    # We return the tuple of feature tensors, with the first num_pref_cat 
    # tensors corresponding to the categoricals, and the last tensor 
    # corresponding to the numericals (in one tensor). Next to that, 
    # Also the padding mask is returned. 
    return tuple(prefix_tensors) + (padding_mask,) 

def generate_suffix_tensors(case_id, 
                            suffix_df, 
                            cardinalities, 
                            cat_cols, 
                            num_cols, 
                            window_size, 
                            str_to_int):
    """Generates the suffix tensors, i.e. the input tokens for the decoder. 

    Parameters
    ----------
    case_id : _type_
        _description_
    suffix_df : _type_
        _description_
    cardinalities : _type_
        _description_
    cat_cols : _type_
        _description_
    num_cols : _type_
        _description_
    window_size : _type_
        _description_
    str_to_int : dict 
        Integer mapping for string instance IDs. 
    """
    # Computing the event idx for each unique prefix. 
    suffix_df['evt_idx'] = suffix_df.groupby([case_id], sort = False).cumcount()
    
    suffix_df['prefix_id_int'] = suffix_df[case_id].map(str_to_int)
    
    # Computing the indices for the tensor scatter updates
    idx = torch.from_numpy(suffix_df[['prefix_id_int', 'evt_idx']].to_numpy()) # (num_rows, 2)
    num_prefs = max(suffix_df['prefix_id_int'])+1 # number of distinct prefixes 

    # Numericals:
    num_nums = len(num_cols) # number of numerical features 
    nums_updates = torch.from_numpy(suffix_df[num_cols].to_numpy()) # shape (num_rows, num_nums)
    nums_updates = nums_updates.type(torch.float32) # shape (num_rows, num_nums)
    # Initialize -1 tensor 
    num_tens = torch.full(size = (num_prefs, window_size, num_nums), fill_value = -1, dtype = torch.float32) # (num_prefs, window_size, num_nums)
    num_tens[idx[:, 0], idx[:,1]] = nums_updates # (num_prefs, window_size, num_nums)




    # Categoricals 
    suffix_tensors = []
    for cat_col in cat_cols:
        cat_updates = torch.from_numpy(suffix_df[cat_col].to_numpy()) # shape (num_rows, )
        cat_updates = cat_updates.to(torch.int64)
        cat_tens = torch.zeros(size = (num_prefs, window_size), dtype = torch.int64) # (num_prefs, window_size)
        cat_tens[idx[:, 0], idx[:, 1]] = cat_updates + 1 # (num_prefs, window_size)
        suffix_tensors.append(cat_tens)

    suffix_tensors.append(num_tens)

    return tuple(suffix_tensors)

def generate_timeLabel_tensor(case_id, 
                              timeLabel_df,  
                              num_cols, 
                              window_size, 
                              str_to_int):
    """Generates the timeLabel tensors for both prediction targets. 
    (Time till next event and complete remaining time)

    Parameters
    ----------
    case_id : _type_
        _description_
    timeLabel_df : _type_
        _description_
    num_cols : _type_
        _description_
    window_size : _type_
        _description_
    str_to_int : dict 
        Integer mapping for string instance IDs. 
    
    Returns
    -------
    timeLabel_tens: tuple of torch.Tensor
        Tuple containing 2 timelabel tensors, each of shape (num_prefs, window_size, 1). 
        First one corresponding to the labels for the time till next event prediction, 
        second one pertaining to the labels for the complete remaining time (until case 
        completion) predictions. 
    """
    # Computing the event idx for each unique prefix. 
    timeLabel_df['evt_idx'] = timeLabel_df.groupby([case_id], sort = False).cumcount()
    
    # Giving each unique prefix a unique integer id. 

    timeLabel_df['prefix_id_int'] = timeLabel_df[case_id].map(str_to_int)
    
    # Computing the indices for the tensor scatter updates
    idx = torch.from_numpy(timeLabel_df[['prefix_id_int', 'evt_idx']].to_numpy()) # (num_rows, 2)
    num_prefs = max(timeLabel_df['prefix_id_int'])+1 # number of distinct prefixes 

    # Numericals:
    num_nums = len(num_cols) # number of numerical features 
    nums_updates = torch.from_numpy(timeLabel_df[num_cols].to_numpy()) # shape (num_rows, num_nums)
    nums_updates = nums_updates.type(torch.float32) # shape (num_rows, num_nums)
    # Initialize -100 tensor 
    num_tens = torch.full(size = (num_prefs, window_size, num_nums), fill_value = -100, dtype = torch.float32) # (num_prefs, window_size, num_nums)
    num_tens[idx[:, 0], idx[:,1]] = nums_updates # (num_prefs, window_size, num_nums) = (", ", 2)

    ttnext_tens = num_tens[:, :, 0, None] # (num_prefs, window_size, 1)
    rtime_tens = num_tens[:, :, 1, None] # (num_prefs, window_size, 1)

    timeLabel_tens = (ttnext_tens, rtime_tens)

    return timeLabel_tens

def generate_actLabel_tensors(case_id, 
                              act_label, 
                              actLabel_df, 
                              cardinalities, 
                              window_size, 
                              str_to_int):
    """Generates the suffix tensors, i.e. the input tokens for the decoder. 

    Parameters
    ----------
    case_id : _type_
        _description_
    actLabel_df : _type_
        _description_
    cardinalities : _type_
        _description_
    cat_cols : _type_
        _description_
    num_cols : _type_
        _description_
    window_size : _type_
        _description_
    str_to_int : dict 
        Integer mapping for string instance IDs. 
    
    Returns
    -------
    act_lab_tens = tf.Tensor
        Activity labels. Tensor of shape (num_prefs, window_size) and of 
        dtype tf.int32. The integer encodings that originally started from 
        0 are now shifted by 1. 0 becomes the label indicating the padding 
        token. 
        Note that the cardinality of these activities, compared to 
        cardinality 'car' of activities in prefixes and decoder suffixes, 
        is 'car'+2. This is due to the fact that here we have to account for 
        2 additional activity tokens that we did not have to acccount for 
        previously: END_TOKEN (idx 'car'+1) and padding token (idx 0). 
    """


    # Cardinality + 1 here bcs for the activity labels, we have an 
    # additional END TOKEN that needs to be predicted! 
    car = cardinalities[act_label] + 1 
    # The current act_label column is still encoded as a categorical
    actLabel_df[act_label] = actLabel_df[act_label].astype(str)
    cat_act_labels = [str(i) for i in range(car-1)] + ['END_TOKEN']
    int_act_labels = [i for i in range(1, car+1)]
    str_labels_to_int = dict(zip(cat_act_labels, int_act_labels))
    actLabel_df[act_label] = actLabel_df[act_label].map(str_labels_to_int)
    # Computing the event idx for each unique prefix. 
    actLabel_df['evt_idx'] = actLabel_df.groupby([case_id], sort = False).cumcount()
    

    actLabel_df['prefix_id_int'] = actLabel_df[case_id].map(str_to_int)
    
    # Computing the indices for the tensor scatter updates
    idx = torch.from_numpy(actLabel_df[['prefix_id_int', 'evt_idx']].to_numpy()) # (num_rows, 2)
    num_prefs = max(actLabel_df['prefix_id_int'])+1 # number of distinct prefixes 


    act_updates = torch.from_numpy(actLabel_df[act_label].to_numpy()) # shape (num_rows, )
    act_updates = act_updates.type(torch.int64)
    # Initializing activity label tensor by filling it with 0s. 
    act_lab_tens = torch.zeros(size =[num_prefs, window_size], dtype = torch.long) # (num_prefs, window_size)
    # Filling for each of the prefix-suffix pairs the activity label indices that correspond actual labels. 
    # Therefore, 0th output label corresponds to the padding label. 
    act_lab_tens[idx[:,0], idx[:,1]] = act_updates # (num_prefs, window_size)

    return act_lab_tens

def generate_outLabel_tensor(case_id, 
                             act_label, 
                             outcomeLabel_df, 
                             outcome, 
                             str_to_int):
    """_summary_

    Parameters
    ----------
    case_id : _type_
        _description_
    act_label : _type_
        _description_
    outcomeLabel_df : _type_
        _description_
    outcome : str 
    str_to_int : dict 
        Integer mapping for string instance IDs. 

    """
    outcomeLabel_df['prefix_id_int'] = outcomeLabel_df[case_id].map(str_to_int)
    
    outLabel_tens = torch.from_numpy(outcomeLabel_df[outcome].to_numpy()) # (num_prefs, )

    return outLabel_tens.unsqueeze(1).to(torch.float32) # (num_prefs, 1)

    

def generate_tensordata(pref_suff, case_id, act_label, cardinality_dict, 
                        num_cols_dict, cat_cols_dict, window_size, outcome):
    """Generates the tensor data for either the train or test set. 

    Parameters
    ----------
    pref_suff : _type_
        _description_
    case_id : _type_
        _description_
    act_label : _type_
        _description_
    cardinality_dict : _type_
        _description_
    num_cols_dict : _type_
        _description_
    cat_cols_dict : _type_
        _description_
    window_size : _type_
        _description_
    outcome : {None, str}
        If a binary outcome column is contained within the event log and 
        outcome prediction labels are needed, outcome should be a string 
        representing the column name that contains the binary outcome. If 
        no binary outcome is present in the event log, or outcome 
        prediction is not needed, `outcome` is None. 

    Saves on disk
    -------
    a dataset of the all_tensors tuple : all tensors in the all_tensors tuple share the same 
        first dimension (num_prefs), and are sliced along that first dimension, thereby using 
        it as the dataset dimension. More information on the `all_tensors` tuple below:

    all_tensors : tuple of tf tensors
        Contains all prefix tensors, suffix tensors and label tensors. 
        Assume: 
            - nc_pref = number of categoricals in prefix 
            - nn_pref = number of numericals in prefix
            - nc_suff = number of categoricals in suffix 
            - nn_suff = number of numericals in suffix 
            - num_prefs = total number of prefixes (and hence suffixes)

        The contents of all_tensors are: 

            - all_tensors[0], ..., all_tensors[nc_pref-1] : (num_prefs, window_size, card_i) 
            for i in 0, ..., nc_pref-1. Contains the one hot encoded categorical tensors for the 
            prefixes. 

            - all_tensors[nc_pref] : contains all the numerical prefix features in one tensor. 
            Shape (num_prefs, window_size, nn_pref)

            - all_tensors[nc_pref + 1] : contains the padding mask. Shape (num_prefs, window_size)

            - all_tensors[i], for i in (nc_pref + 1 + 1), ..., (nc_pref + 1 + nc_suff): categoricals in suffix, 
            each with shape (num_prefs, window_size, card_i)

            - all_tensors[nc_pref + 1 + nc_suff + 1]: all suffix numericals in one tensor. 
            Shape (num_prefs, window_size, nn_suff)

            - all_tensors[(nc_pref + 1 + nc_suff + 1) + 1] & all_tensors[(nc_pref + 1 + nc_suff +1) + 2]: 
            Both shape (num_prefs, window_size, 1). First one containing the time labels for the time till 
            next event prediction, second one the time labels for the complete remaining time prediction. 

            - all_tensors[(nc_pref + 1 + nc_suff + 1 + 2)+1] : Containing the activity prediction labels, shape 
            (num_prefs, window_size). NOTE: these activity labels have a cardinality of 2 more than the 
            (one-hot encoded) activity labels in the prefix and decoder suffix. 
    """
    # Unpacking the dataframes 
    prefix_df = pref_suff[0]
    suffix_df = pref_suff[1]
    timeLabel_df = pref_suff[2]
    actLabel_df = pref_suff[3]
    if outcome:
        outcomeLabel_df = pref_suff[4]
    
    # Constructing prefix-suffix (/instance) id 
    # to integer mapping for consistent index assignment across the 
    # different tensors 
    prefix_list = list(prefix_df.drop_duplicates(subset = case_id)[case_id])
    int_map = [i for i in range(len(prefix_list))]
    str_to_int = dict(zip(prefix_list, int_map))
    # prefix_df['prefix_id_int'] = prefix_df[case_id].map(str_to_int)

    # First nc_pref tensors in prefix_tensors tuple correspond to the categoricals. Next tensor in that 
    # tuple contains all numericals. Last tensor is 'padding_mask' (num_prefs, window_size). 
    # (nc_pref = number of cats in prefix)
    print("Generating prefix tensors ...")
    prefix_tensors = generate_prefix_tensors(case_id, prefix_df, cardinalities = cardinality_dict, 
                                                           cat_cols = cat_cols_dict['prefix_df'], 
                                                           num_cols = num_cols_dict['prefix_df'], 
                                                           window_size = window_size, 
                                                           str_to_int=str_to_int)
    # 'suffix_tensors' is a tuple of tensors, with first nc_suff tensors corresponding to the 
    # categoricals in the suffix, and last tensor corresponding to the numericals. 
    print("Generating (decoder) suffix tensors ...")
    suffix_tensors = generate_suffix_tensors(case_id, suffix_df, cardinalities = cardinality_dict, 
                                             cat_cols = cat_cols_dict['suffix_df'], 
                                             num_cols = num_cols_dict['suffix_df'], 
                                             window_size = window_size, 
                                             str_to_int=str_to_int) # NEED cardinality for this too and num cats!
    
    # timeLabel_tens is a tuple of 2 tensors, both of shape (num_prefs, window_size, 1). First tensor 
    # contains the labels for the time until next event prediction, second one contains the labels for 
    # the remaining time prediction. 
    print("Generating time label tensors ...")
    timeLabel_tens = generate_timeLabel_tensor(case_id, timeLabel_df, num_cols = num_cols_dict['timeLabel_df'], 
                                               window_size = window_size, 
                                               str_to_int=str_to_int)
    
    # actLabel_tens is a single tensor (not a list) of shape (num_prefs, window_size) and contains the labels 
    # for the remaining trace / activity predictions. NOTE that it's cardinality is +2 compared to the activity 
    # token information in the other suffixes, since we account for a padding token (0) and an END TOKEN here. 
    # Consequently, the dimension of the activity prediction output layer should be 
    # cardinality_dict[act_label] + 2. 
    print("Generating activity label tensors ...")
    actLabel_tens = generate_actLabel_tensors(case_id, act_label, actLabel_df, cardinalities = cardinality_dict, 
                                              window_size = window_size, 
                                              str_to_int=str_to_int)
    
    # Make one big tuple containing all the tensors 
    all_tensors = prefix_tensors + suffix_tensors + timeLabel_tens + (actLabel_tens,)

    if outcome: 
        outLabel_tens = generate_outLabel_tensor(case_id, 
                                                 act_label, 
                                                 outcomeLabel_df, 
                                                 outcome, 
                                                 str_to_int=str_to_int)
        all_tensors += (outLabel_tens, )
    return all_tensors
    

def generate_tensordata_train_test(train_pref_suff, 
                                   val_pref_suff,
                                   test_pref_suff, 
                                   case_id, 
                                   act_label, 
                                   cardinality_dict, 
                                   num_cols_dict, 
                                   cat_cols_dict, 
                                   window_size, 
                                   outcome, 
                                   log_name):
    """Takes as input the prefix and suffix dataframes (both for train and 
    test set), and generates the appropriate tensors for the inputs of the 
    crtp transformer encoder, decoder, and for the suffix labels. 
    
    Parameters
    ----------
    train_pref_suff : tuple of pd.DataFrame
        Contains 4 or 5 dataframes relating to the training set, 4 if 
        `outcome=None`, 5 otherwise.
    val_pref_suff : tuple of pd.DataFrame
        Contains 4 or 5 dataframes relating to the validation set, 4 if 
        `outcome=None`, 5 otherwise.
    test_pref_suff : tuple of pd.DataFrame
        Contains 4 or 5 dataframes relating to the test set, 4 if 
        `outcome=None`, 5 otherwise.
    cardinality_dict : dict
        Contains the categorical features as the keys, and the 
        corresponding cardinality as the values. 
    num_cols_dict : dict 
        Contains, for each of the 4 dataframes (same for train and 
        test set), the strings referring to the numeric columns 
        that need to be contained within the associated tensors. 
    cat_cols_dict : dict 
        Contains, for each of the 4 dataframes (same for train and 
        test set), the strings referring to the categorical columns 
        that need to be contained within the associated tensors. 
    window_size : int 
        ...
    outcome : {None, str}
        If a binary outcome column is contained within the event log and 
        outcome prediction labels are needed, outcome should be a string 
        representing the column name that contains the binary outcome. If 
        no binary outcome is present in the event log, or outcome 
        prediction is not needed, `outcome` is None. 
    log_name : str 
        Name of the log for which the prefix and suffix tensors are saved on disk.
    """
    # Dimensions and cardinalities prefix 

    #   the prefix categoricals 
    prefCatCols = cat_cols_dict['prefix_df']
    #   number of categ features in prefixes 
    num_pref_cat = len(prefCatCols)
    #   cardinalities in right order
    pref_cat_cars = [cardinality_dict[catcol] for catcol in prefCatCols]
    prefNumCols = num_cols_dict['prefix_df']

    # Dimensions and cardinalities (decoder) suffix

    #   the suffix categoricals 
    suffCatCols = cat_cols_dict['suffix_df']
    #   number of categ features in suffixes 
    num_suff_cat = len(suffCatCols)
    #   cardinalities suff in right order 
    suff_cat_cars = [cardinality_dict[catcol] for catcol in suffCatCols]

    # Train set tensors: 
    print("Computing train set tensors")
    train_data = generate_tensordata(pref_suff = train_pref_suff, case_id = case_id, 
                                     act_label = act_label, cardinality_dict = cardinality_dict, 
                                     num_cols_dict = num_cols_dict, 
                                     cat_cols_dict = cat_cols_dict, 
                                     window_size = window_size, 
                                     outcome=outcome)
    print("____________________________")
    print("Computing validation set tensors")
    val_data = generate_tensordata(pref_suff = val_pref_suff, case_id = case_id, 
                                     act_label = act_label, cardinality_dict = cardinality_dict, 
                                     num_cols_dict = num_cols_dict, 
                                     cat_cols_dict = cat_cols_dict, 
                                     window_size = window_size, 
                                     outcome=outcome)
    print("____________________________")
    print("Computing test set tensors")
    test_data = generate_tensordata(pref_suff=test_pref_suff, case_id = case_id, 
                                    act_label = act_label, cardinality_dict = cardinality_dict, 
                                    num_cols_dict = num_cols_dict, 
                                    cat_cols_dict = cat_cols_dict, 
                                    window_size = window_size, 
                                    outcome=outcome)
    # Num activities in the labels (including padding and end token)
    num_activities = cardinality_dict[act_label] + 2

    # Saving everything to disk
    output_directory = log_name
    os.makedirs(output_directory, exist_ok=True)

    #   Prefix cardinalities 
    prefix_cardinality_path = os.path.join(output_directory, log_name + '_cardin_list_prefix.pkl')
    with open(prefix_cardinality_path, 'wb') as file:
        pickle.dump(pref_cat_cars, file)

    #   Suffix cardinalities 
    suffix_cardinality_path = os.path.join(output_directory, log_name + '_cardin_list_suffix.pkl')
    with open(suffix_cardinality_path, 'wb') as file:
        pickle.dump(suff_cat_cars, file)

    

    return train_data, val_data, test_data, num_pref_cat, num_suff_cat, pref_cat_cars, suff_cat_cars, num_activities


    


def load_cardinality_lists(path_name):
    """Helper function to read cardinality list from disk."""
    with open(path_name, 'rb') as file:
        loaded_list = pickle.load(file)
    
    return loaded_list
