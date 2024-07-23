"""Functionality for conducting parallel inference over batches for the 
SuTraN model. Returning all metrics both aggregated 
over all prefix lengths, as well as individually for each prefix length.

SuTraN conducts inference in an AR manner, 
within the forward method of the initialized model itself when set to 
eval mode.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from SuTraN.inference_environment import BatchInference

from torch.utils.data import TensorDataset, DataLoader
import os 
import pickle

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def inference_loop(model, 
                   inference_dataset,
                   remaining_runtime_head, 
                   outcome_bool, 
                   num_categoricals_pref,
                   mean_std_ttne, 
                   mean_std_tsp, 
                   mean_std_tss,
                   mean_std_rrt, 
                   results_path=None, 
                   val_batch_size=8192):
    """Inference loop, both for validition set and ultimate test set.

    Parameters
    ----------
    model : SuTraN or SuTraN_no_context
        The initialized and current version of a SuTraN neural network. 
        Should be set to evaluation mode to trigger the AR decoding loop 
        within SuTraN's forward method. 
    inference_dataset : tuple of torch.Tensor
        Contains the tensors comprising the inference dataset, including 
        the labels for all prediction heads. 
    remaining_runtime_head : bool
        Whether or not the model is also trained to directly predict the 
        remaining runtime given a prefix. See Notes for further remarks 
        regarding the `remaining_runtime_head` parameter. 
    outcome_bool : bool, optional 
        Whether or not the model should also include a prediction 
        head for binary outcome prediction. If 
        `outcome_bool=True`, a prediction head for predicting 
        the binary outcome given a prefix is added. This prediction 
        head, in contrast to the time till next event and activity 
        suffix predictions, will only be trained to provide a 
        prediction at the first decoding step. Note that the 
        value of `outcome_bool` should be aligned with the 
        `outcome_bool` parameter of the model and train 
        procedure, as well as with the preprocessing pipeline that 
        produces the labels. See Notes for further remarks 
        regarding the `outcome_bool` parameter. 
    num_categoricals_pref : int
        The number of categorical features (including the activity label) 
        contained within each prefix event token. For the NDA 
        implementation of SuTraN (`SuTraN_no_context`), this parameter 
        should be set to `1`. 
    mean_std_ttne : list of float
        Training mean and standard deviation used to standardize the time 
        till next event (in seconds) target. Needed for re-converting 
        ttne predictions to original scale. Mean is the first entry, 
        std the second.
    mean_std_tsp : list of float
        Training mean and standard deviation used to standardize the time 
        since previous event (in seconds) feature of the decoder suffix 
        tokens. Needed for re-converting time since previous event values 
        to original scale (seconds). Mean is the first entry, std the 2nd.
    mean_std_tss : list of float
        Training mean and standard deviation used to standardize the time 
        since start (in seconds) feature of the decoder suffix tokens. 
        Needed for re-converting time since start to original scale 
        (seconds). Mean is the first entry, std the 2nd. 
    mean_std_rrt : list of float
        List consisting of two floats, the training mean and standard 
        deviation of the remaining runtime labels (in seconds). Needed 
        for de-standardizing remaining runtime predictions and labels, 
        such that the MAE can be expressed in seconds (and minutes). 
    results_path : None or str, optional
        The absolute path name of the folder in which the final 
        evaluation results should be stored. The default of None should 
        be retained for intermediate validation set computations.
    val_batch_size : int, optional
        Batch size for iterating over inference dataset. By default 8192. 

    Notes
    -----
    Additional remarks regarding parameters: 

    * `remaining_runtime_head` : This parameter has become redundant, and 
      should always be set to `True`. SuTraN by default accounts for an 
      additional direct remaining runtime prediction head. 

    * `outcome_bool` : For the paper implementation, this boolean should 
      be set to `False`. For future work, already included for extending 
      the multi-task PPM setup to simultaneously predict a binary outcome 
      target for each prefix as well.  
      

    Additional explanations commonly referred tensor dimensionalities: 

    * `num_prefs` : the integer number of instances, aka 
        prefix-suffix pairs, contained within the inference dataset 
        for which this `BatchInference` instance is initialized. Also 
        often referred to as `batch_size` in the comment lines 
        complementing the code. 

    * `window_size` : the maximum sequence length of both the prefix 
        event sequences, as well as the generated suffix event 
        predictions. 

    * `num_activities` : the total number of possible activity labels 
        to be predicted. This includes the padding and end token. The 
        padding token will however always be masked, such that it 
        cannot be predicted. Also referred to as `num_classes`. 
    """

    # Creating TensorDataset and corresponding DataLoader out of 
    # `inference_dataset`. 
    inf_tensordataset = TensorDataset(*inference_dataset)
    inference_dataloader = DataLoader(inf_tensordataset, batch_size=val_batch_size, shuffle=False, drop_last=False, pin_memory=True)

    # Define auxiliary booleans specifying SuTraN's multi-task setup 
    only_rrt = (not outcome_bool) & remaining_runtime_head
    only_out = outcome_bool & (not remaining_runtime_head)
    both_not = (not outcome_bool) & (not remaining_runtime_head)
    both = outcome_bool & remaining_runtime_head

    # If the only additional prediction head (on top of activity and ttne 
    # suffix) is remaining time prediction
    if only_rrt:
        # number of prediction targets (and hence labels) to be 
        # simultaneously predicted 
        num_target_tens = 3

        # Index of the ground-truth activity label tensor in the dataset 
        act_label_index = -1 

    # If the only additional prediction head (on top of activity and ttne 
    # suffix) is (binary) outcome prediction
    elif only_out:
        # number of prediction targets (and hence labels) to be 
        # simultaneously predicted 
        num_target_tens = 3

        # Index of the ground-truth activity label tensor in the dataset 
        act_label_index = -2

    # If the additional prediction heads (on top of activity and ttne 
    # suffix) comprise both rrt and outcome prediction
    elif both:
        # number of prediction targets (and hence labels) to be 
        # simultaneously predicted 
        num_target_tens = 4

        # Index of the ground-truth activity label tensor in the dataset 
        act_label_index = -2

    # If their are no additional prediction heads (on top of activity and 
    # ttne suffix)
    elif both_not: 
        # number of prediction targets (and hence labels) to be 
        # simultaneously predicted 
        num_target_tens = 2

        # Index of the ground-truth activity label tensor in the dataset 
        act_label_index = -1
    
    # Retrieving labels 
    labels_global = inference_dataset[-num_target_tens:] 
    
    # Retrieving seq length (`window_size`, also referred to as W) 
    window_size = labels_global[act_label_index].shape[-1]


    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        # Initializing global tensors for storing model outputs on CPU
        # The two directly underneath have shape 
        # (num_prefs, window_size) after inference loop
        suffix_acts_decoded_global = torch.empty((0, window_size), dtype=torch.int64)
        suffix_ttne_preds_global = torch.empty((0, window_size), dtype=torch.float32)

        if remaining_runtime_head:
            # Shape (number of instances, ) after inference loop
            rrt_pred_global = torch.tensor(data=[], dtype=torch.float32)
        
        if outcome_bool:
            # Shape (number of instances, ) after inference loop
            out_pred_global = torch.tensor(data=[], dtype=torch.float32)


        # Initializing a global tensor to store the prefix lengths of all inference instances 
        pad_mask_global = inference_dataset[num_categoricals_pref+1] # (num_prefs, window_size)
        pref_len_global = torch.argmax(pad_mask_global.to(torch.int64), dim=-1) # (batch_size,)

        # Prefixes of maximum length (window_size) have no True values in 
        # the padding mask and hence prefix length 0 is (falsely) derived 
        # replacing 0s with window_size 
        pref_len_global = torch.where(pref_len_global == 0, window_size, pref_len_global) # (num_prefs,)

        act_labels_global = inference_dataset[act_label_index] # (num_prefs, window_size)
        num_classes = torch.max(act_labels_global).item() + 1


        # Total number of test or validation set instances / 
        # prefix-suffix pairs 
        num_prefs = act_labels_global.shape[0]

        # Derive ground-truth suffix length of each instance 
        suf_len_global = torch.argmax((act_labels_global == (num_classes-1)).to(torch.int64), dim=-1) + 1 # (num_prefs,)

        # Iterating over the inference batches 
        for valbatch_num, vdata in tqdm(enumerate(inference_dataloader), desc="Validation batch calculation"):
                vinputs = vdata[:-num_target_tens]
                # Assign all input tensors to GPU
                vinputs = [vinput_tensor.to(device) for vinput_tensor in vinputs]


                # Decoding the batch_size instances. 
                # NOTE that the model should be set in evaluation mode 
                # in order for it to handle the entire (greedy) decoding 
                # process itself. 
                voutputs = model(vinputs, 
                                 window_size, 
                                 mean_std_ttne, 
                                 mean_std_tsp, 
                                 mean_std_tss)
                
                # Retrieving the different outputs and adding them to 
                # their respective global tensors on the CPU

                #   - Greedily decoded activity suffix 
                suffix_acts_decoded = voutputs[0] # (B, W) torch.int64
                suffix_acts_decoded_global = torch.cat((suffix_acts_decoded_global, suffix_acts_decoded.cpu()), dim=0)

                #   - Predicted TTNE suffix (in standardized scale)
                #     Note: only the predictions up until the decoding 
                #     step in which the END token is predicted in 
                #     `suffix_acts_decoded` should be taken into account. 
                suffix_ttne_preds = voutputs[1] # (B, W) torch.float32
                suffix_ttne_preds_global = torch.cat((suffix_ttne_preds_global, suffix_ttne_preds.cpu()), dim=0)

                #   - remaining runtime and / or outcome predictions if 
                #     trained for. 
                if only_rrt: 
                    # - (direct) remaining runtime predictions. 
                    #   Still in standardized scale. 
                    rrt_pred = voutputs[-1] # (B,) torch.float32
                    rrt_pred_global = torch.cat((rrt_pred_global, rrt_pred.cpu()), dim=-1)

                elif only_out:
                    # - (binary) outcome prediction
                    out_pred = voutputs[-1] # (B, ) torch.float32
                    out_pred_global = torch.cat((out_pred_global, out_pred.cpu()), dim=-1)
                
                elif both:
                    # - (direct) remaining runtime predictions. 
                    #   Still in standardized scale. 
                    rrt_pred = voutputs[-2] # (B,) torch.float32
                    rrt_pred_global = torch.cat((rrt_pred_global, rrt_pred.cpu()), dim=-1)

                    # - (binary) outcome prediction
                    out_pred = voutputs[-1] # (B, ) torch.float32
                    out_pred_global = torch.cat((out_pred_global, out_pred.cpu()), dim=-1)
        
        # Consolidating all predictions 
        outputs_global = (suffix_acts_decoded_global, suffix_ttne_preds_global)

        if remaining_runtime_head:
            outputs_global += (rrt_pred_global,)
        
        if outcome_bool:
            outputs_global += (out_pred_global, )
        
        # Write away results for final test set inference if specified 
        if results_path: 
            subfolder_path = results_path
            os.makedirs(subfolder_path, exist_ok=True)

            # Specifying paths to save the prediction tensors and writing 
            # them to disk. 

            #   Activity suffix predictions 
            suffix_acts_decoded_path = os.path.join(subfolder_path, 'suffix_acts_decoded.pt')
            torch.save(suffix_acts_decoded_global, suffix_acts_decoded_path)

            #   Timestamp suffix predictions 
            suffix_ttne_preds_path = os.path.join(subfolder_path, 'suffix_ttne_preds.pt')
            torch.save(suffix_ttne_preds_global, suffix_ttne_preds_path)

            if remaining_runtime_head:
                rrt_pred_path = os.path.join(subfolder_path, 'rrt_pred.pt')
                torch.save(rrt_pred_global, rrt_pred_path)
            if outcome_bool:
                out_pred_path = os.path.join(subfolder_path, 'out_pred.pt')
                torch.save(out_pred_global, out_pred_path)
            
            # Prefix length and suffix length 
            pref_len_path = os.path.join(subfolder_path, 'pref_len.pt')
            torch.save(pref_len_global, pref_len_path)

            suf_len_path = os.path.join(subfolder_path, 'suf_len.pt')
            torch.save(suf_len_global, suf_len_path)

            # Labels 
            # Note, this is a tuple of tensors instead of just a tensor
            labels_path = os.path.join(subfolder_path, 'labels.pt')
            torch.save(labels_global, labels_path)


        # Initializing BatchInference object for computing 
        # inference metrics 
        infer_env = BatchInference(preds=outputs_global, 
                                   labels=labels_global, 
                                   mean_std_ttne=mean_std_ttne, 
                                   mean_std_tsp=mean_std_tsp, 
                                   mean_std_tss=mean_std_tss, 
                                   mean_std_rrt=mean_std_rrt, 
                                   remaining_runtime_head=remaining_runtime_head, 
                                   outcome_bool=outcome_bool)
        # Retrieving individual validation metric components for each of 
        # the 'num_prefs' instances, for all prediction targets. 

        # Compute initial TTNE metrics
        # both of shape (num_prefs, window_size). 
        # Only the MAE values pertaining to the non-padded suffix event  
        # tokens (in the two initial TTNE metrics) should still be   
        # selected before computing global averages (see infra). 
        MAE_ttne_stand, MAE_ttne_seconds = infer_env.compute_ttne_results()

        # (normalized) Damerau-Levenshtein similarity activity suffix prediction
        dam_lev = infer_env.damerau_levenshtein_distance_tensors() # (num_prefs, )

        # MAE remaining runtime predictions (standardized scale and in seconds)
        if remaining_runtime_head:
            MAE_rrt_stand, MAE_rrt_seconds = infer_env.compute_rrt_results() # (num_prefs,)
            
        if outcome_bool:
            inference_BCE = infer_env.compute_outcome_BCE() # (num_prefs,)

        # Length differences between predicted and ground-truth suffixes. 
        length_diff, length_diff_too_early, length_diff_too_late, amount_right = infer_env.compute_suf_length_diffs()


        # Averages computation

        # Time Till Next Event (TTNE) suffix 
        #     Retain only MAE contributions pertaining to 
        #     non-padded suffix events
        counting_tensor = torch.arange(window_size, dtype=torch.int64) # (window_size,)
        #       Repeat the tensor along the first dimension to match the desired shape
        counting_tensor = counting_tensor.unsqueeze(0).repeat(num_prefs, 1) # (num_prefs, window_size)
        #       Compute boolean indexing tensor to, for each of the 
        #       'num_prefs' instances, slice out only the absolute 
        #       errors pertaining to actual non-padded suffix events. 
        before_end_token = counting_tensor <= (suf_len_global-1).unsqueeze(-1) # (num_prefs,)

        avg_MAE_ttne_stand = MAE_ttne_stand[before_end_token] # shape (torch.sum(suf_len_global), )
        avg_MAE_ttne_stand = (torch.sum(avg_MAE_ttne_stand) / avg_MAE_ttne_stand.shape[0]).item()

        avg_MAE_ttne_seconds = MAE_ttne_seconds[before_end_token] # shape (torch.sum(suf_len_global), )
        avg_MAE_ttne_seconds = (torch.sum(avg_MAE_ttne_seconds) / avg_MAE_ttne_seconds.shape[0]).item()

        avg_MAE_ttne_minutes = avg_MAE_ttne_seconds / 60

        # Activity suffix 
        #   normalized Damerau Levenshtein similarity Activity Suffix 
        #   prediction 
        dam_lev_similarity = 1. - dam_lev # (num_prefs,)
        avg_dam_lev = (torch.sum(dam_lev_similarity) / dam_lev_similarity.shape[0]).item() # Scalar

        # Remaining Runtime (RRT)
        if remaining_runtime_head:
            avg_MAE_stand_RRT = (torch.sum(MAE_rrt_stand) / MAE_rrt_stand.shape[0]).item() # Scalar 
            avg_MAE_seconds_RRT = (torch.sum(MAE_rrt_seconds) / MAE_rrt_seconds.shape[0]).item() # Scalar 
            avg_MAE_minutes_RRT = avg_MAE_seconds_RRT / 60 # Scalar 
            # Without averaging
            MAE_rrt_minutes = MAE_rrt_seconds / 60 # (num_prefs, )
        
        if results_path:
            # Writing the tensors containing the DLS and MAE RRT for each individual 
            # test set instance / test set prefix-suffix pair, to disk. 
            dam_lev_sim_path = os.path.join(subfolder_path, 'dam_lev_similarity.pt')
            torch.save(dam_lev_similarity, dam_lev_sim_path)
            
            if remaining_runtime_head:
                MAE_rrt_minutes_path = os.path.join(subfolder_path, 'MAE_rrt_minutes.pt')
                torch.save(MAE_rrt_minutes, MAE_rrt_minutes_path)


        # Binary outcome 
        if outcome_bool:
             avg_BCE_out = (torch.sum(inference_BCE) / inference_BCE.shape[0]).item()
             # AUC-ROC and AUC-PR computations
             auc_roc, auc_pr = infer_env.compute_AUC()

        # Length differences: 
        total_num = length_diff.shape[0]
        num_too_early = length_diff_too_early.shape[0]
        num_too_late = length_diff_too_late.shape[0]
        percentage_too_early = num_too_early / total_num
        percentage_too_late = num_too_late / total_num
        percentage_correct = amount_right.item() / total_num
        mean_absolute_length_diff = (torch.sum(torch.abs(length_diff)) / total_num).item()
        mean_too_early = (torch.sum(torch.abs(length_diff_too_early)) / num_too_early).item()
        mean_too_late = (torch.sum(torch.abs(length_diff_too_late)) / num_too_late).item()


    return_list = [avg_MAE_ttne_stand, avg_MAE_ttne_minutes]
    return_list += [avg_dam_lev, percentage_too_early, percentage_too_late]
    return_list += [percentage_correct, mean_absolute_length_diff, mean_too_early, mean_too_late]

    if remaining_runtime_head:
         return_list += [avg_MAE_stand_RRT, avg_MAE_minutes_RRT]
    if outcome_bool:
         return_list += [avg_BCE_out, auc_roc, auc_pr]
    
    # Making dictionaries of the results for over both prefix and suff length. 
    results_dict_pref = {}
    for i in range(1, window_size+1):
        bool_idx = pref_len_global==i
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
        bool_idx = suf_len_global==i
        dam_levs = dam_lev_similarity[bool_idx].clone()
        MAE_rrt_i = MAE_rrt_minutes[bool_idx].clone()
        num_inst = dam_levs.shape[0]
        if num_inst > 0:
            avg_dl = (torch.sum(dam_levs) / num_inst).item()
            avg_mae = (torch.sum(MAE_rrt_i) / num_inst).item()
            results_i = [avg_dl, avg_mae, num_inst]
            results_dict_suf[i] = results_i
    
    return_list += [results_dict_pref, results_dict_suf]


    if results_path:
        # Writing the prefix and suffix dictionaries to disk 

        #   Prefix length results dictionary 
        preflen_results_path = os.path.join(subfolder_path, 'prefix_length_results_dict.pkl')
        with open(preflen_results_path, 'wb') as file:
            pickle.dump(results_dict_pref, file)

        #   Suffix length results dictionary 
        suflen_results_path = os.path.join(subfolder_path, 'suffix_length_results_dict.pkl')
        with open(suflen_results_path, 'wb') as file:
            pickle.dump(results_dict_suf, file)

    return return_list