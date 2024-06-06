"""This module contains functionality to adapt the tensor data, created 
for SuTraN, towards the format needed for the CRTP_LSTM 
benchmark. 
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
        categorical being a integer encoded torch.Tensor of dtype 
        torch.int64 and shape (num_prefs, window_size).
    numeric_inputs : torch.Tensor 
        Dtype torch.float32. Contains all selected numeric prefix 
        features in one tensor of shape 
        (num_prefs, window_size, num_nums). 
    pad_mask : torch.Tensor
        Tensor of dtype torch.bool and shape 
        (num_prefs, window_size). Indicates for each of the 
        'num_prefs' instances which of the indices pertain to padded 
        prefix events (True). Actual event indices are indicated with 
        False. 
    num_nums : int 
        The amount of numerical features contained within each prefix 
        event token. 
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

def replace_rrt_pads(rrt_labels, 
                     mean_std_rrt):
  """Replaces the original padding values of -100 with the 
  zero-equivalent value. The rrt_labels are standardized such that they 
  have a mean of 0 and standardeviation of 1 (~N(0,1)). Therefore, 
  the zero-equivalent of the standardized labels is equal to the 
  negative of the original mean (used for standardization) divided by the 
  original standard deviation. 

  Parameters
  ----------
  rrt_labels : torch.Tensor 
      Shape (N, window_size, 1) and dtype torch.float32. Contains the 
      standardized remaining runtime labels. 
  mean_std_rrt : list of float
      List of two elements, the training mean and standard deviation 
      used for standardizing the original remaining runtime values. 
  """
  stand_0_eq = -mean_std_rrt[0]/mean_std_rrt[1]
  rrt_labels = torch.where(condition=(rrt_labels != -100), input=rrt_labels, other=stand_0_eq)
  return rrt_labels

def convert_to_lstm_data(data, 
                         outcome_bool, 
                         mean_std_rrt, 
                         num_categoricals_pref, 
                         num_numericals_pref, 
                         masking = True):
    """Convert the original dataset (tuple of tensors) into the format 
    needed for training and evaluating the CRTP_LSTM benchmark. Note that 
    this function makes the crucial underlying assumption that remaining 
    runtime labels are included in the original data. 

    This includes:

    * selecting only the tensors pertaining to the prefix 
      features. The CRTP-LSTM does not generate suffixes 
      auto-regressively (AR) and hence does not utilize 
      suffix event tokens. 
    
    * converting the originally right-padded prefix tensors into 
      left-padded prefix tensors. 
    
    * in the original RRT suffix labels, starting from the index after 
      which the activity label suffix contained the END token, the RRT 
      suffix labels were padded with -100 values to be able to mask out 
      the loss contributions of the padded suffix events (after END 
      token should have been predicted). 
      In the original implementation of the CRTP-LSTM benchmark, masking 
      was not applied and loss functions included contributions by 
      padded suffix events, pertaining to target values of 0 (or 0-
      equivalent). The authors however indicated that after further 
      exploration, masking improved the results and resulted in faster 
      convergence. Therefore, in the re-implementation in the SuTraN 
      paper, masking was implemented, indicated by `masking=True`. 
      By specifying `masking=False`, one could train without masking, 
      and in that case, the -100 padding 
      values are replaced by 0-equivalent values. 
    
    The padding mask, which indicates which prefix event tokens (in the 
    right-padded version) correspond to padded events, is not needed 
    as input for the CRTP-LSTM benchmark, but is used upon inference 
    for deriving the prefix lengths. Therefore, it is retained. 

    Parameters
    ----------
    data : tuple of torch.Tensor 
        Tensors corresponding to the original dataset, including 
        prefix event token features, a boolean padding mask, suffix event 
        token features and three or four label tensors, depending on 
        whether or not outcome labels were included. (Three if 
        `outcome_label=False`, four otherwise.)
    outcome : bool 
        Indicating whether binary outcome target is included in the data. 
    mean_std_rrt : list of float
        List consisting of two floats, the training mean and standard 
        deviation of the remaining runtime labels.
    num_categoricals_pref : int
        The number of categorical features (including the activity 
        labels) in the prefix event tokens. Each categorical has its own 
        integer-encoded tensor in both the original data, as well as in  
        the modified data format for the CRTP_LSTM benchmark 
        (`lstm_dataset`).
    num_numericals_pref : int 
        The amount of numerical features contained within each prefix 
        event token. 
    masking : bool 
        If `True`, the masking of the padded labels will be retained. 

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

    # Selecting the activity suffix and remaining runtime suffix labels 
    # as well as the ttne labels 

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
    
    # Replacing -100. padding values of rrt_labels with 0 equivalent value 
    if not masking:
      rrt_labels = replace_rrt_pads(rrt_labels=rrt_labels, 
                                    mean_std_rrt=mean_std_rrt)
    
    lstm_dataset = cat_inputs + (num_ftrs_pref,) + (padding_mask_input,) + (ttne_labels,) + (rrt_labels, ) + (act_labels,)
    return lstm_dataset