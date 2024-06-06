"""Module containing functionality to convert original Tensor Data 
into format required by LSTM Encoder Decoder Benchmark."""

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
        torch.int64 and shape (num_prefs, window_size). For the NDA ED-
        LSTM benchmark, only the activity labels of the prefix events 
        are retained. 
    numeric_inputs : torch.Tensor 
        Dtype torch.float32. Contains all selected numeric prefix 
        features in one tensor of shape 
        (num_prefs, window_size, num_nums). 
        For the NDA ED-LSTM benchmark, this amount is always equal to 2, 
        pertaining to the two numerical features serving as the timestamp 
        proxy (time since start aka tss, and time since previous event 
        aka tsp).
    pad_mask : torch.Tensor
        Tensor of dtype torch.bool and shape 
        (num_prefs, window_size). Indicates for each of the 
        'num_prefs' instances which of the indices pertain to padded 
        prefix events (True). Actual event indices are indicated with 
        False. 
    num_numericals_pref : int 
        The amount of numerical prefix features selected. For the NDA 
        ED-LSTM benchmark, this amount is always equal to 2, pertaining 
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

def convert_to_lstm_data(data, 
                         outcome_bool, 
                         num_categoricals_pref, 
                         num_numericals_pref):
    """Convert the original dataset (tuple of tensors) into the format 
    needed for training and evaluating the DNA ED-LSTM benchmark. Note that 
    this function makes the crucial underlying assumption that remaining 
    runtime labels are included in the original data. 

    In contrast to the functionality for converting the data into the 
    format required by the SEP-LSTM benchmark, this function already 
    requires all features of the prefix event tokens (i.e. all features 
    except for the activity integers and two numerical timestamp 
    proxies), to be discarded. 

    Therefore, `num_categoricals_pref` and `num_numericals_pref` should 
    always be equal to 1 and 2 respectively. IF one wishes to implement 
    DA versions that require the prefix event tokens to be left-padded 
    instead of right-padded, the data can be converted while retaining 
    all additional payload data in the prefix event tokens by specifying 
    the appropriate number of categorical and numerical prefix event 
    features, while not subsetting the data beforehand. 

    This includes:

    * selecting only the activity label suffix, remaining runtime (rrt) 
      label suffix, and the time till next event (ttne) label suffix 
      out of all available labels. (Which means dropping the binary 
      outcome labels if included.)
    
    * converting the originally right-padded prefix tensors into 
      left-padded prefix tensors. 
    
    The padding mask, which indicates which prefix event tokens (in the 
    right-padded version) correspond to padded events, is not needed 
    as input for the ED-LSTM benchmark, but is used upon inference 
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
    num_categoricals_pref : int
        The number of categorical features (including the activity 
        labels) in the prefix event tokens. Each categorical has its own 
        integer-encoded tensor in both the original data, as well as the 
        modified data format for the ED-LSTM benchmark.
    num_numericals_pref : int 
        The amount of numerical features contained within each prefix 
        event token. 

    Returns
    -------
    lstm_dataset : tuple of torch.Tensor
    """
    # Subsetting the needed tensors: 
    # Tensor containing the numerical features of the prefix events. 
    num_ftrs_pref = data[num_categoricals_pref] # (N, window_size, num_num)
    # Tensor containing the padding mask for the prefix events. 
    padding_mask_input = data[num_categoricals_pref+1] # (N, window_size)
    cat_inputs = data[:num_categoricals_pref] # tuple of torch.Tensor
    acts_suf = data[num_categoricals_pref+2].clone()
    num_ftrs_suf = data[num_categoricals_pref+3].clone()

    # Selecting the activity suffix and remaining runtime suffix labels 
    if outcome_bool:
        act_labels = data[-2]
        rrt_labels = data[-3]
        ttne_labels = data[-4]
    else:
        act_labels = data[-1]
        rrt_labels = data[-2]
        ttne_labels = data[-3]

    
    # Converting left padded prefix inputs to right padded ones
    cat_inputs, num_ftrs_pref = left_pad(cat_inputs=cat_inputs, 
                                         numeric_inputs=num_ftrs_pref, 
                                         pad_mask=padding_mask_input, 
                                         num_nums=num_numericals_pref)
    
    
        
    lstm_dataset = cat_inputs + (num_ftrs_pref,) + (padding_mask_input,) + (acts_suf,) + (num_ftrs_suf,) + (ttne_labels,) + (rrt_labels, ) + (act_labels,)
    return lstm_dataset