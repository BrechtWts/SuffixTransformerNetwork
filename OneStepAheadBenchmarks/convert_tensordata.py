"""This module contains the functionality needed for converting the 
original data formatted according to the requirements of our 
newly introduced model, into the format required for the 
Single-Event-Prediction (SEP) benchmarks. 
"""

import torch

def left_pad(cat_inputs, 
             numeric_inputs, 
             pad_mask, 
             num_nums):
    """Change the right padded prefix tensors to left padded prefix 
    tensors. 

    Parameters
    ----------
    cat_inputs : list of torch.Tensor
        Contains the prefix categorical tensors, with each prefix 
        categorical being a integer-encoded torch.Tensor of dtype 
        torch.int64 and shape (num_prefs, window_size). For the NDA SEP-
        LSTM benchmark, only the activity labels of the prefix events 
        are retained. 
    numeric_inputs : torch.Tensor 
        Dtype torch.float32. Contains all selected numeric prefix 
        features in one tensor of shape 
        (num_prefs, window_size, num_nums). 
        For the NDA SEP-LSTM benchmark, this amount is always equal to 2, 
        pertaining to the two numerical features serving as the timestamp 
        proxy (time since start aka tss, and time since previous event 
        aka tsp).
    pad_mask : torch.Tensor
        Tensor of dtype torch.bool and shape 
        (num_prefs, window_size). Indicates for each of the 
        'num_prefs' instances which of the indices pertain to padded 
        prefix events (True). Actual event indices are indicated with 
        False. 
    num_nums : int 
        The amount of numerical prefix features selected. For the NDA 
        SEP-LSTM benchmark, this amount is always equal to 2, pertaining 
        to the two numerical features serving as the timestamp proxy 
        (time since start aka tss, and time since previous event aka tsp).
    """
    num_prefs = numeric_inputs.shape[0]
    window_size = numeric_inputs.shape[1]

    # For each of the num_prefs instances, contains True for the 
    # actual event indices, False for the padded prefix events 
    select_og_bool = pad_mask==False # (num_prefs, window_size)

    # Reversed, True for the indices on the right where to insert in 
    # new right padded tensors 
    insert_new_bool = select_og_bool.flip(dims=[1]) # (num_prefs, window_size)
    
    # tensor 0 - nc_pref-1 are the prefix categoricals 
    cat_tensors = []
    for i in range(len(cat_inputs)):
        pref_cat = cat_inputs[i] # (num_prefs, window_size)

        # Converting right-padding categorical tensor into left-padding 
        new_pref_cat = torch.zeros(size=(num_prefs, window_size), dtype=torch.int64) # (num_prefs, window_size)

        # # Inserting the actual events at the end of the new left 
        # # padded tensor 
        new_pref_cat[insert_new_bool] = pref_cat[select_og_bool] # (num_prefs, window_size)

        cat_tensors.append(new_pref_cat)

    # Also converting right padded numerics tensor to left padded 
    new_prefix_numerics = torch.full(size=(num_prefs, window_size, num_nums), fill_value=0., dtype=torch.float32) # (num_prefs, window_size, num_nums)

    #   Inserting the actual events at the end of the new left 
    #   padded tensor 
    new_prefix_numerics[insert_new_bool] = numeric_inputs[select_og_bool] # (num_prefs, window_size, num_nums)

    return tuple(cat_tensors), new_prefix_numerics

def convert_to_SEP_data(data, 
                         outcome_bool, 
                         num_categoricals_pref, 
                         act_input_index, 
                         tss_index):
    """Convert the original dataset (tuple of tensors) into the format 
    needed for training and evaluating the SEP LSTM benchmarks. Note that 
    this function makes the crucial underlying assumption that remaining 
    runtime labels are included in the original data. 

    This includes:

    * For the prefix event tokens: selecting only the tensors 
      containing the activity integers, and the two numerical features 
      serving as the timestamp proxy (time since start aka tss, 
      and time since previous event aka tsp).

    * Discarding the tensors containing the suffix activity integers, and 
      the two numerical features serving as the timestamp proxy, since 
      the SEP-LSTM only updates the prefix event token sequence upon 
      inference, and hence does not utilize suffix event tokens; 

    * selecting only the activity label suffix, remaining runtime (rrt) 
      label suffix, and the time till next event (ttne) label suffix 
      out of all available labels. (Which means dropping the binary 
      outcome labels if included.)
    
    * converting the originally right-padded prefix tensors into 
      left-padded prefix tensors. 
    
    The padding mask, which indicates which prefix event tokens (in the 
    right-padded version) correspond to padded events, is not needed 
    as input for the SEP-LSTM benchmark, but is used upon inference 
    for deriving the prefix lengths. Therefore, it is retained. 

    Parameters
    ----------
    data : tuple of torch.Tensor 
        Tensors corresponding to the original dataset, including 
        prefix event token features, a boolean padding mask, suffix event 
        token features and three or four label tensors, depending on 
        whether or not outcome labels were included. (Three if 
        `outcome_label=False`, four otherwise.)
    outcome_bool : bool 
        Indicating whether binary outcome target is included in the data. 
    mean_std_rrt : list of float
        List consisting of two floats, the training mean and standard 
        deviation of the remaining runtime labels.
    num_categoricals_pref : int
        The number of categorical features (including the activity 
        labels) in the prefix event tokens. This pertains to the fully 
        DA version of the data. For the NDA SEP-LSTM benchmark, 
        only the activity label will ultimately be retained. 
    num_numericals_pref : int 
        The amount of numerical features contained within each prefix 
        event token. This pertains to the fully 
        DA version of the data. For the NDA SEP-LSTM benchmark, 
        only the two numerical features serving as the timestamp 
        proxy (time since start aka tss, and time since previous event 
        aka tsp).
    act_input_index : int 
        Index of the tensor containing the actvity vectors of the 
        prefix events. 
    tss_index : int 
        Index of the time since start (tss) feature inside of the 
        tensor containing all numeric event features. The time 
        since previous event (tsp) feature is located at the next index.

    Returns
    -------
    lstm_dataset : tuple of torch.Tensor
        The tuple solely containing the needed tensors for the 
        SEP_LSTM benchmark. 
    """
    # Subsetting the needed tensors: 
    # Tensor containing the numerical features of the prefix events. 
    num_ftrs_pref = data[num_categoricals_pref] # (N, window_size, num_num)
    # Tensor containing the padding mask for the prefix events. 
    padding_mask_input = data[num_categoricals_pref+1] # (N, window_size)
    cat_inputs = data[:num_categoricals_pref] # tuple of torch.Tensor

    # Slicing out the activities of the prefix events
    new_cat_inputs = []
    new_cat_inputs.append(cat_inputs[act_input_index]) # (num_prefs, window_size, num_activities)
    cat_inputs = new_cat_inputs

    # Slicing out the numeric features, starting with tss and tsp
    numeric_inds = [tss_index, tss_index+1]
    
    numeric_inds_tens = torch.tensor(numeric_inds, dtype=torch.long) # (2, )
    num_ftrs_pref = torch.index_select(num_ftrs_pref, dim=-1, index=numeric_inds_tens) # (num_prefs, W, 2)
    

    # Selecting the activity suffix and remaining runtime suffix labels 
    if outcome_bool:
        act_labels = data[-2] # (num_prefs, window_size)
        rrt_labels = data[-3] # (num_prefs, window_size, 1)
        ttne_labels = data[-4] # (num_prefs, window_size, 1)
    else:
        act_labels = data[-1] # (num_prefs, window_size)
        rrt_labels = data[-2] # (num_prefs, window_size, 1)
        ttne_labels = data[-3] # (num_prefs, window_size, 1)

    
    # Converting left padded prefix inputs to right padded ones 
    cat_inputs, num_ftrs_pref = left_pad(cat_inputs=cat_inputs, 
                                         numeric_inputs=num_ftrs_pref, 
                                         pad_mask=padding_mask_input, 
                                         num_nums=2)

    
    lstm_dataset = cat_inputs + (num_ftrs_pref,) + (padding_mask_input,) + (ttne_labels,) + (rrt_labels, ) + (act_labels,)

    return lstm_dataset