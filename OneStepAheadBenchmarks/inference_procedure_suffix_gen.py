"""Functionality for the iterative feedback loop needed for leveraging 
the SEP benchmarks, which are trained for predicting the next event's 
activity label and timestamp (Time Till Next Event aka TTNE), for suffix 
generation. This includes the computation of remaining runtime as the sum 
of time till next event predictions."""

import torch
# import torch.nn as nn
from tqdm import tqdm
from OneStepAheadBenchmarks.inference_environment import BatchInference
from torch.utils.data import TensorDataset, DataLoader
import os 


# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def inference_loop(model, 
                   inference_dataset, 
                   mean_std_ttne, 
                   mean_std_tss_pref, 
                   mean_std_tsp_pref,
                   mean_std_rrt, 
                   num_categoricals_pref, 
                   num_numericals_pref, 
                   results_path=None, 
                   val_batch_size=8192):
    """Inference loop, both for validition set and ultimate test set. 
    Facilitates external validation loop needed for SEP-LSTM to generate 
    suffixes, by, for each batch, iterating over the `window_size` 
    decoding steps, querying the current modified version of the prefix 
    event tokens from the initialized `BatchInference` instance, feeding 
    it to the model as if it were a completely new batch of instances, 
    and returning the model's predictions to the `BatchInference` 
    instance, which processes the predictions and uses them to add a 
    new prefix event token to the sequence. 

    Parameters
    ----------
    model : SEP_Benchmarks_time
        Initialized instance of the SEP-LSTM benchmark implementation. 
    inference_dataset : tuple of torch.Tensor
        Contains the tensors comprising the inference dataset, including 
        the labels for all prediction heads. 
    num_categoricals_pref : int
        The number of categorical features (including the activity label) 
        contained within each prefix event token. Given the fact that 
        SEP-LSTM is non-data aware (NDA), this should be set to 1. 
    mean_std_ttne : list of float
        Training mean and standard deviation used to standardize the time 
        till next event (in seconds) target. Needed for re-converting 
        ttne predictions to original scale. Mean is the first entry, 
        std the second.
    mean_std_tss_pref : list of float
        Training mean and standard deviation used to standardize the 
        time since start (in seconds) feature. In contrast 
        to both SuTraN and ED-LSTM, the mean and standard deviation 
        here pertains to the one of the prefix event tokens, instead 
        of the suffix event tokens. SEP-LSTM progressively adds new 
        prefix event tokens to the input sequence by means of the 
        external feedback loop, and hence does not produce suffix 
        event tokens. Needed for re-converting time since start 
        event values to original scale (seconds). Mean is the first 
        entry, std the 2nd.
    mean_std_tsp_pref : list of float
        Training mean and standard deviation used to standardize the 
        time since previous event (in seconds) feature. In contrast 
        to both SuTraN and ED-LSTM, the mean and standard deviation 
        here pertains to the one of the prefix event tokens, instead 
        of the suffix event tokens. SEP-LSTM progressively adds new 
        prefix event tokens to the input sequence by means of the 
        external feedback loop, and hence does not produce suffix 
        event tokens. Needed for re-converting time since previous 
        event values to original scale (seconds). Mean is the first 
        entry, std the 2nd.
    mean_std_rrt : list of float, optional
        List consisting of two floats, the training mean and standard 
        deviation of the remaining runtime labels (in seconds). Needed 
        for de-standardizing remaining runtime predictions and labels, 
        such that the MAE can be expressed in seconds (and minutes). 
    num_categoricals_pref : int 
        The number of categorical features (including the activity label) 
        contained within each prefix event token. Given the fact that 
        SEP-LSTM is non-data aware (NDA), this should be set to 1. 
    num_numericals_pref : int 
        The number of numerical features (including the two numerical 
        timestamp proxies) contained within each prefix event token. 
        Given the fact that SEP-LSTM is non-data aware (NDA), this should 
        be set to 2. 
    results_path : None or str, optional
        The absolute path name of the folder in which the final evaluation results 
        should be stored. The default of None should be retained for 
        intermediate validation set computations. 
    val_batch_size : int, optional
        Batch size for iterating over inference dataset. By default 8192. 
    """
    # Creating TensorDataset and corresponding DataLoader out of 
    # `inference_dataset`. 
    inf_tensordataset = TensorDataset(*inference_dataset)
    inference_dataloader = DataLoader(inf_tensordataset, batch_size=val_batch_size, shuffle=False, drop_last=False, pin_memory=True)

    # Retrieving labels 
    labels_global = inference_dataset[-3:] 

    # Retrieving seq length (`window_size`, also referred to as W) 
    act_label_index = -1
    window_size = labels_global[act_label_index].shape[-1]
    num_classes = torch.max(labels_global[act_label_index]).item() + 1
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():

        # Initializing global tensors for storing model outputs on CPU
        # The two directly underneath have shape 
        # (number of instances, window_size) after inference loop
        suffix_acts_decoded_global = torch.empty((0, window_size), dtype=torch.int64)
        suffix_ttne_preds_global = torch.empty((0, window_size), dtype=torch.float32)

        # Initializing tensors storing the evaluation metrics for 
        # suffix prediction

        # NOTE: due to legacy implementation computing distorted MAE TTNE metrics, 
        # standardized TTNE MAE metrics (MAE_1_stand_global & MAE_2_stand_global) 
        # are computed twice, while being exactly the same. Same for the metrics 
        # in seconds and minutes. Will be resolved soon. 
        MAE_1_stand_global = torch.tensor(data=[], dtype=torch.float32).to(device)
        MAE_1_seconds_global = torch.tensor(data=[], dtype=torch.float32).to(device)
        MAE_2_stand_global = torch.tensor(data=[], dtype=torch.float32).to(device)
        MAE_2_seconds_global = torch.tensor(data=[], dtype=torch.float32).to(device)

        MAE_rrt_sum_global = torch.tensor(data=[], dtype=torch.float32).to(device)

        length_diff_global = torch.tensor(data=[], dtype=torch.int64).to(device)
        length_diff_too_early_global = torch.tensor(data=[], dtype=torch.int64).to(device)
        length_diff_too_late_global = torch.tensor(data=[], dtype=torch.int64).to(device)
        amount_right_global = torch.tensor(data=0, dtype=torch.int64).to(device) # scalar tensor

        # Initializing a global tensor to store the prefix lengths of all inference instances 
        pref_len_global = torch.tensor(data=[], dtype=torch.int64).to(device)

        # Initializing a global tensor to store the suffix lengths of all inference instances 
        suf_len_global = torch.tensor(data=[], dtype=torch.int64).to(device)

        dam_lev_global = torch.tensor(data=[], dtype=torch.int64).to(device)

        for valbatch_num, vdata in tqdm(enumerate(inference_dataloader), desc="Validation batch calculation"):
                vinputs = vdata[:-3]
                vlabels = vdata[-3:]

                pad_mask = vdata[num_categoricals_pref+1]
                pad_mask = pad_mask.to(device) # (batch_size, window_size)
                # Sending inputs and labels to GPU
                vinputs = [vinput_tensor.clone().to(device) for vinput_tensor in vinputs]
                vlabels = [vlabel_tensor.clone().to(device) for vlabel_tensor in vlabels]

                act_labels = vlabels[-1]
                # Deriving suffix length of each instance 
                suf_len = torch.argmax((act_labels == (num_classes-1)).to(torch.int64), dim=-1) + 1 # (batch_size,)
                suf_len_global = torch.cat(tensors=(suf_len_global, suf_len), dim=-1)

                # Deriving the prefix length of each instance 
                padding_idx = torch.argmax(pad_mask.to(torch.int64), dim=-1) # (batch_size,)
                pref_len_global = torch.cat(tensors=(pref_len_global, padding_idx), dim=-1)

                # Initializing an inference object to keep track of the 
                # decoding progress (in the iteratvie feedback loop) for 
                # the whole batch of instances 
                infer_env = BatchInference(inputs=vinputs, 
                                           labels=vlabels, 
                                           num_categoricals_pref=num_categoricals_pref, 
                                           num_numericals_pref=num_numericals_pref, 
                                           mean_std_ttne=mean_std_ttne, 
                                           mean_std_tsp=mean_std_tsp_pref, 
                                           mean_std_tss=mean_std_tss_pref, 
                                           mean_std_rrt=mean_std_rrt)

                # Iterative feedback loop. If model has not predicted 
                # end token after window_size decoding steps, the end 
                # token is enforced automatically. 
                for i in range(window_size):
                    # Model predictions NEXT activity and timestamp
                    voutputs = model(infer_env.inputs) 
                    act_preds, ttne_preds = voutputs[0], voutputs[1] # (batch_size, num_classes) and (bs, 1)

                    # Instructing inference object to process predictions 
                    infer_env.decode_step(act_preds=act_preds, 
                                          ttne_preds=ttne_preds, 
                                          dec_step=i)
                
                # Retrieving the predicted activity and ttne suffixes 
                suffix_acts_decoded = infer_env.suffix_acts_decoded.clone() # (batch_size, window_size)
                suffix_acts_decoded_global = torch.cat((suffix_acts_decoded_global, suffix_acts_decoded.cpu()), dim=0) 

                suffix_ttne_preds = infer_env.suffix_ttne_preds.clone() # (batch_size, window_size)
                suffix_ttne_preds_global = torch.cat((suffix_ttne_preds_global, suffix_ttne_preds.cpu()), dim=0) 

                # TTNE metrics (MAE)
                MAE_1_stand, MAE_1_seconds, MAE_2_stand, MAE_2_seconds = infer_env.compute_ttne_results()
                MAE_1_stand_global = torch.cat(tensors=(MAE_1_stand_global, MAE_1_stand), dim=-1)
                MAE_1_seconds_global = torch.cat(tensors=(MAE_1_seconds_global, MAE_1_seconds), dim=-1)
                MAE_2_stand_global = torch.cat(tensors=(MAE_2_stand_global, MAE_2_stand), dim=-1)
                MAE_2_seconds_global = torch.cat(tensors=(MAE_2_seconds_global, MAE_2_seconds), dim=-1)

                # (normalized) Damerau-Levenshtein similarity activity suffix prediction
                dam_lev = infer_env.damerau_levenshtein_distance_tensors() # (batch_size, )
                dam_lev_global = torch.cat(tensors=(dam_lev_global, dam_lev), dim=-1)

                # Computing the Remaining Runtime (RRT) MAE (in seconds). 
                # The Remaining Runtime prediction is computed by the sum 
                # of TTNE predictions (until the END token was predicted)
                MAE_rrt_sum = infer_env.compute_rrt_results()
                MAE_rrt_sum_global = torch.cat(tensors=(MAE_rrt_sum_global, MAE_rrt_sum), dim=-1) # (batch_size,)

                # Length differences between predicted and ground-truth suffixes. 
                length_diff, length_diff_too_early, length_diff_too_late, amount_right = infer_env.compute_suf_length_diffs()
                length_diff_global = torch.cat(tensors=(length_diff_global, length_diff), dim = -1)
                length_diff_too_early_global = torch.cat(tensors=(length_diff_too_early_global, length_diff_too_early), dim=-1)
                length_diff_too_late_global = torch.cat(tensors=(length_diff_too_late_global, length_diff_too_late), dim=-1)
                amount_right_global += amount_right

        # Correction pref len 
        # replacing 0s with max_pref_len 
        pref_len_global = torch.where(pref_len_global == 0, window_size, pref_len_global) # (num_prefs,)
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
            
            # Prefix length and suffix length 
            pref_len_path = os.path.join(subfolder_path, 'pref_len.pt')
            torch.save(pref_len_global.cpu(), pref_len_path)

            suf_len_path = os.path.join(subfolder_path, 'suf_len.pt')
            torch.save(suf_len_global.cpu(), suf_len_path)

            # Labels 
            # Note, this is a tuple of tensors instead of just a tensor
            labels_path = os.path.join(subfolder_path, 'labels.pt')
            torch.save(labels_global, labels_path)

        # Final validation metrics:
        # ------------------------- 
        #   TTNE predictions 
        avg_MAE1_stand = (torch.sum(MAE_1_stand_global) / MAE_1_stand_global.shape[0]).item()
        avg_MAE1_seconds = (torch.sum(MAE_1_seconds_global) / MAE_1_seconds_global.shape[0]).item()
        avg_MAE1_minutes = avg_MAE1_seconds / 60
        avg_MAE2_stand = (torch.sum(MAE_2_stand_global) / MAE_2_stand_global.shape[0]).item()
        avg_MAE2_seconds = (torch.sum(MAE_2_seconds_global) / MAE_2_seconds_global.shape[0]).item()
        avg_MAE2_minutes = avg_MAE2_seconds / 60

        #   normalized Damerau Levenshtein Similarity Activity Suffix prediction 
        avg_dam_lev = (torch.sum(dam_lev_global) / dam_lev_global.shape[0]).item()
        avg_dam_lev = 1 - avg_dam_lev

        #   Without averaging 
        dam_lev_similarity = 1. - dam_lev_global # (num_prefs,)

        #   Remaining Runtime Predictions (RRT)
        avg_MAE_rrt_sum = (torch.sum(MAE_rrt_sum_global) / MAE_rrt_sum_global.shape[0]).item()
        avg_MAE_rrt_sum_minutes = avg_MAE_rrt_sum / 60

        #   Without averaging 
        MAE_rrt_minutes = MAE_rrt_sum_global / 60 # (num_prefs, )

        if results_path:
            # Writing the tensors containing the DLS and MAE RRT for each individual 
            # test set instance / test set prefix-suffix pair, to disk. 
            dam_lev_sim_path = os.path.join(subfolder_path, os.path.join(subfolder_path, 'dam_lev_similarity.pt'))
            torch.save(dam_lev_similarity, dam_lev_sim_path)

            MAE_rrt_minutes_path = os.path.join(subfolder_path, os.path.join(subfolder_path, 'MAE_rrt_minutes.pt'))
            torch.save(MAE_rrt_minutes, MAE_rrt_minutes_path)

        # Length differences: 
        total_num = length_diff_global.shape[0]
        num_too_early = length_diff_too_early_global.shape[0]
        num_too_late = length_diff_too_late_global.shape[0]
        percentage_too_early = num_too_early / total_num
        percentage_too_late = num_too_late / total_num
        percentage_correct = amount_right_global.item() / total_num
        mean_absolute_length_diff = (torch.sum(torch.abs(length_diff_global)) / total_num).item()
        mean_too_early = (torch.sum(torch.abs(length_diff_too_early_global)) / num_too_early).item()
        mean_too_late = (torch.sum(torch.abs(length_diff_too_late_global)) / num_too_late).item()

    return_list = [avg_MAE1_stand, avg_MAE1_seconds, avg_MAE1_minutes, avg_MAE2_stand, avg_MAE2_seconds, avg_MAE2_minutes]
    return_list += [avg_dam_lev, percentage_too_early, percentage_too_late]
    return_list += [percentage_correct, mean_absolute_length_diff, mean_too_early, mean_too_late, avg_MAE_rrt_sum_minutes]

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
    return return_list
