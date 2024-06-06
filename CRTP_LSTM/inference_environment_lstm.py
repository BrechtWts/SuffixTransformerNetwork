"""Inference environment `BatchInference` for the CRTP-LSTM benchmark. 
For each batch in the inference loop (contained within the 
`CRTP_LSTM\inference_procedure_lstm.py` module), a new instance of the 
`BatchInference` class is initialized. 
"""

import torch
import torch.nn as nn
import numpy as np
# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BatchInference():
    def __init__(self,
                 preds,
                 labels, 
                 mean_std_ttne, 
                 mean_std_tsp, 
                 mean_std_tss, 
                 mean_std_rrt):
        """
        Inference environment for the CRTP-LSTM benchmark. Contains 
        methods for computing the different evaluation metrics for 
        activity suffix, timestamp suffix and remaining runtime 
        predictions. 
        
        Takes as 
        inputs the predictions (activity suffix and remaining 
        runtime suffix) made by CRTP-LSTM for a whole batch 
        of test (or validation) set instances, as well as the 
        corresponding activity and remaining runtime suffix labels. 

        Preprocesses both predictions and labels in different scales 
        and representations, upon which the methods for computing the 
        metrics can be called. 

        Parameters
        ----------
        preds : tuple of torch.Tensor 
            Containing the two tensors containing CRTP-LSTM's activity 
            suffix predictions (i.e. unnormalized probability 
            distributions), as well as its remaining runtime suffix 
            predictions, in that order. The former has shape 
            (batch_size, window_size, num_activities), the latter 
            has shape (batch_size, window_size, 1). 
        labels : list of torch.Tensor
            List of three tensors containing the timestamp suffix, 
            activity suffix and remaining runtime labels. The tensors 
            have shape (batch_size, window_size, 1), 
            (batch_size, window_size, 1) and (batch_size, window_size) 
            respectively, and dtypes torch.float32, torch.float32 
            and torch.int64. 
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
            since start (in seconds) feature of the decoder suffix 
            tokens. Needed for re-converting time since previous event values 
            to original scale (seconds). Mean is the first entry, std the 2nd.
        mean_std_rrt : list of float, optional
            List consisting of two floats, the training mean and standard 
            deviation of the remaining runtime labels (in seconds). Needed 
            for de-standardizing remaining runtime predictions and labels, 
            such that the MAE can be expressed in seconds (and minutes). 

        Notes
        -----
        * `batch_size` : the integer number of instances, aka 
          prefix-suffix pairs, contained within the batch for which the 
          `BatchInference` instance is initialized. 

        * `window_size` : the maximum sequence length of both the prefix 
          event sequences, as well as the generated suffix event 
          predictions. 

        * `num_activities` : the total number of possible activity labels 
          to be predicted. This includes the padding and end token. The 
          padding token will however always be masked, such that it 
          cannot be predicted. Also referred to as `num_classes`. 
        """

        self.rrt_labels = labels[-2] # (batch_size, window_size, 1)
        self.rrt_labels = self.rrt_labels[:,:,0] # (batch_size, window_size)

        self.act_labels = labels[-1] # (batch_size, window_size)
        self.ttne_labels = labels[-3]
        self.ttne_labels = self.ttne_labels[:, :, 0] # (batch_size, window_size)


        self.window_size = self.act_labels.shape[-1]
        self.batch_size = self.act_labels.shape[0]
        # End Token gets last index, and should be contained 
        # within the label tensor for all of the prefix-suffix 
        # pairs. That's how we derive the num_classes from it. 
        self.num_classes = torch.max(self.act_labels).item() + 1

        self.mean_std_ttne = mean_std_ttne
        self.mean_std_tsp = mean_std_tsp
        self.mean_std_tss = mean_std_tss
        self.mean_std_rrt = mean_std_rrt

        self.act_preds = preds[0] # (batch_size, window_size, num_classes)
        self.rrt_preds = preds[1] # (batch_size, window_size, 1)
        self.rrt_preds = self.rrt_preds[:,:,0] # (batch_size, window_size)
        # rrt predictions tensor converted in seconds (original scale)
        self.rrt_preds_seconds = self.convert_to_seconds(time_string='rrt', input_tensor=self.rrt_preds.clone()) # (batch_size, window_size)

        # Creating rrt_labels in original scale 
        self.rrt_labels_seconds = self.convert_to_seconds(time_string='rrt', input_tensor=self.rrt_labels.clone()) # (batch_size, window_size)

        self.ttne_labels_seconds = self.convert_to_seconds(time_string='ttne', input_tensor=self.ttne_labels)
        self.suffix_acts_decoded, self.pred_length = self.decode_activities() # (batch_size, window_size) & (batch_size,)

        # Getting (batch_size, )-shaped torch.int64 tensor 
        # ``self.actual_length``containing for each of the batch_size 
        # instances the suffix length (0-based)
        self.actual_length = self.get_actual_length()

    def decode_activities(self):
        """Decodes the unnormalized predicted activity logits of shape 
        (batch_size, window_size, num_classes) into an integer encoded 
        suffix tensor of shape (batch_size, window_size) and of 
        dtype torch.int64. The decoding happens greedily, by taking the 
        largest predicted probability (equivalent with max logit) at each 
        decoding step. The first index, index 0, corresponds to the 
        padding token, and is masked out beforehand. 
        """
        # masking out padding token such that it is not chosen:
        act_preds_copy = self.act_preds.clone()
        act_preds_copy[:,:,0] = -1000000
        suffix_acts_decoded = torch.argmax(act_preds_copy, dim=-1) # (batch_size, window_size)
        # Initialize tensor that contains 1 where end token 
        # (idx num_classes-1) is predicted, 0 otherwise. 
        end_pred_tens = (suffix_acts_decoded== (self.num_classes-1)).to(torch.int64) # (batch_size, window_size)
        # Init tensor that contains for each instance the number of times 
        # an end token is predicted 
        sum_tens = torch.sum(input=end_pred_tens, dim=-1) # (batch_size,)

        # Artificial end token inserted on the last time step, for all 
        # instances for which the model never produced an END prediction
        suffix_acts_dec_help = suffix_acts_decoded.clone() # (batch_size, window_size)
        suffix_acts_dec_help[:,-1] = self.num_classes-1
        # (batch_size, window_size)
        suffix_acts_decoded = torch.where(condition=(sum_tens==0).unsqueeze(-1), input=suffix_acts_dec_help, other=suffix_acts_decoded)

        # Right padding decoded activity predictions with index 0 after
        # starting from the first END token prediction
        pred_length = torch.argmax((suffix_acts_decoded== (self.num_classes-1)).to(torch.int64), dim=-1) 
        counting_tensor = torch.arange(self.window_size, dtype=torch.int64).to(device) # (window_size,)
        #       Repeat the tensor along the first dimension to match the desired shape
        counting_tensor = counting_tensor.unsqueeze(0).repeat(self.batch_size, 1).to(device) # (batch_size, window_size)
        padding_bool = counting_tensor > pred_length.unsqueeze(-1) # (batch_size, window_size)
        padding_inds = torch.nonzero(input=padding_bool, as_tuple=True)
        suffix_acts_decoded[padding_inds] = 0

        return suffix_acts_decoded, pred_length


        

    
    def convert_to_seconds(self, time_string, input_tensor):
        """Convert a tensor of any shape, containing time-related 
        features, predictions or labels, that are standardized (and 
        potentially log-transformed before being standardized) back 
        into the original scale of seconds. 

        Parameters
        ----------
        time_string : {'ttne', 'tsp', 'tss', 'rrt'}
            String indicating which type of time feature / target / 
            prediction needs to be converted. 
        input_tensor : torch.Tensor
            Tensor to be converted into the original seconds scale. 
        """
        if time_string == 'ttne':
            train_mean = self.mean_std_ttne[0]
            train_std = self.mean_std_ttne[1]
        elif time_string == 'tsp':
            train_mean = self.mean_std_tsp[0]
            train_std = self.mean_std_tsp[1]
        elif time_string == 'tss':
            train_mean = self.mean_std_tss[0]
            train_std = self.mean_std_tss[1]
        elif time_string == 'rrt':
            train_mean = self.mean_std_rrt[0]
            train_std = self.mean_std_rrt[1]

        # De-standardizing the time-related tensors back into 
        # the scale of seconds. 
        converted_tensor = input_tensor*train_std + train_mean
        # Disscarding negative predictions (if any) by imposing lower 
        # bound of 0. 
        converted_tensor = torch.clamp(converted_tensor, min=0)

        return converted_tensor # Same shape as input_tensor

    def convert_to_transf(self, time_string, input_tensor):
        """Convert a tensor of any shape, containing time-related 
        features, predictions or labels, that are expressed in seconds, 
        back into the transformed scale, which is obtained by 
        standardization.

        Parameters
        ----------
        time_string : {'ttne', 'tsp', 'tss', 'rrt'}
            String indicating which type of time feature / target / 
            prediction needs to be converted. 
        input_tensor : torch.Tensor
            Tensor to be converted back into the transformed scale. 
        """
        if time_string == 'ttne':
            train_mean = self.mean_std_ttne[0]
            train_std = self.mean_std_ttne[1]
        elif time_string == 'tsp':
            train_mean = self.mean_std_tsp[0]
            train_std = self.mean_std_tsp[1]
        elif time_string == 'tss':
            train_mean = self.mean_std_tss[0]
            train_std = self.mean_std_tss[1]
        elif time_string == 'rrt':
            train_mean = self.mean_std_rrt[0]
            train_std = self.mean_std_rrt[1]

        # Standardizing the given tensor containing time-related 
        # information in the scale of seconds. 
        converted_tensor = (input_tensor - train_mean) / train_std

        return converted_tensor # Same shape as input_tensor
    
    def get_actual_length(self):
        """Get the ground truth suffix length of each of the batch_size 
        instances. Computed based on the act_labels, by determining 
        the suffix index where the ground truth 'END TOKEN' is located. 
        Note: this is 0-based. E.g., a value of 0 would indicate that the 
        END TOKEN should be predicted immediately at the very first 
        decoding step. 
        """
        actual_length = torch.argmax((self.act_labels == (self.num_classes-1)).to(torch.int64), dim=-1) # (batch_size,)
        return actual_length

    def compute_ttne_results(self):
        """Derive implicit timestamp suffix predictions based on 
        predicted remaining time suffix CRTP-LSTM in the original 
        scale of seconds, and subsequently compute the absolute errors 
        for the timestamp (Time Till Next Event, aka TTNE) suffix 
        predictions in the original (in seconds) scale. 

        Each of the `num_prefs` prefix-suffix pairs 
        (inference instances), the TTNE suffix is comprised of a sequence 
        of `window_size` TTNE values, both the TTNE predictions and the 
        TTNE labels. Given one particular inference instance, its 
        `window_size` absolute differences are hence computed by 
        taking the absolute value of the difference between each  
        ground-truth TTNE value, and the corresponding prediction. For 
        the majority of the cases however, a significant portion of the 
        `window_size` TTNE values are (right-)padded values, and hence do 
        not correspond to actual suffix events. Therefore, only the 
        absolute differences pertaining to actual (ground-truth) suffix 
        events are selected and returned. 
        
        Furthermore, to resemble real-
        life situations, in which timestamp predictions for positions 
        after the position at which the END token (activity suffix) is 
        predicted are ignored, as closely as possible, the TTNE 
        predictions after the index of the predicted END token are 
        replaced with zero. Additionally, since decoded 
        activity labels fed to the decoder after the prediction of the 
        END token do not contain any informational value, these 
        timestamp predictions would not make any sense anyway. 
        
        Returns
        -------
        MAE_ttne_seconds : torch.Tensor 
            Absolute errors timestamp predictions in seconds for all 
            actual / non-padded suffix events. Shape (NP,), with NP being 
            equal to the total number of actual suffix events (including 
            the added END tokens) over all `batch_size` 
            prefix-suffix pairs in the validation or test set. 
        """
        #       Generate a tensor with values counting from 0 to window_size
        counting_tensor = torch.arange(self.window_size, dtype=torch.int64).to(device) # (window_size,)
        #       Repeat the tensor along the first dimension to match the desired shape
        counting_tensor = counting_tensor.unsqueeze(0).repeat(self.batch_size, 1).to(device) # (batch_size, window_size)

        # Deriving boolean tensor of shape (batch_size, window_size) 
        # containing True for the indices after its END token prediction 
        # in the predicted activity suffix 
        pad_preds = counting_tensor > self.pred_length.unsqueeze(-1) # (batch_size, window_size)

        # Pad RRT predictions in seconds with 0 from the first index 
        # after the predicted end token. 
        self.rrt_preds_seconds[pad_preds] = 0 # (batch_size, window_size)

        # Derive timestamp suffix predictions (in seconds)
        self.rrt_preds_shifted = self.rrt_preds_seconds.clone() # (batch_size, window_size)
        self.rrt_preds_shifted = torch.cat((self.rrt_preds_shifted[:, 1:], torch.zeros(size=(self.batch_size,1), dtype=torch.float32, device=device)), dim=-1)
        self.ttne_preds_seconds = self.rrt_preds_seconds - self.rrt_preds_shifted # (batch_size, window_size)

        # Imposing non-negative lower boundary 
        self.ttne_preds_seconds = torch.clamp(self.ttne_preds_seconds, min=0) # (batch_size, window_size)

        

        # Deriving boolean tensor of shape (batch_size, window_size) with 
        # True on the indices before and on the ground truth END token 
        before_end_token = counting_tensor <= self.actual_length.unsqueeze(-1)

        # Computing MAE ttne elements and slicing out the considered ones 
        MAE_ttne_seconds = torch.abs(self.ttne_preds_seconds - self.ttne_labels_seconds) # (batch_size, window_size)
        MAE_ttne_seconds = MAE_ttne_seconds[before_end_token] # shape (torch.sum(self.actual_length+1), )

        return MAE_ttne_seconds


    def damerau_levenshtein_distance_tensors(self):
        """Compute the (normalized) damerau-levenshtein distance for each of 
        the `batch_size` predicted suffixes in parallel. 

        Notes
        -----
        Leverages the following `BatchInference` attributes:

        * `self.suffix_acts_decoded` : torch.Tensor containing the 
          integer-encoded sequence of predicted activities. Shape 
          (self.batch_size, self.window_size) and dtype torch.int64. The 
          predicted suffix activities are derived greedily, by taking 
          the activity label index pertaining to the highest probability 
          in `self.act_preds`. 

        * `self.act_labels` : torch.Tensor containing the integer-encoded 
          sequence of ground-truth activity label suffixes. Shape 
          (self.batch_size, self.window_size) and dtype torch.int64. 
        
        * `self.pred_length` : torch.Tensor of dtype torch.int64 and 
          shape (self.batch_size, ). Contains the (0-based) index at 
          which the model predicted the END token during decoding, for 
          each of the self.batch_size instances. Consequently, for each 
          instance i (i=0, ..., self.batch_size-1), the predicted suffix 
          length is equal to self.pred_length[i] + 1. 
        
        * `self.actual_length` : torch.Tensor of dtype torch.int64 and 
          shape (self.batch_size, ). Contains the (0-based) index of the 
          END token in the ground-truth activity suffix for each of the 
          (self.batch_size,) instances. Consequently, for each instance i 
          (i=0, ..., self.batch_size-1), the ground-truth suffix length 
          is equal to self.actual_length[i] + 1. 

        Returns
        -------
        dam_lev_dist : torch.Tensor
            Tensor containing the normalized Damerau-Levenshtein distance 
            for each of the `batch_size` prefix-suffix pairs / instances. 
            Shape (batch_size, ), dtype torch.float32. 
        """
        len_pred = self.pred_length+1 # (self.batch_size, )
        len_actual = self.actual_length+1 # (self.batch_size, )
        # max_length between the ground-truth and predicted activity sequence
        max_len = torch.maximum(len_pred, len_actual) # (self.batch_size,)

        # Initializing the (self.batch_size, self.window_size+1, 
        # self.window_size+1)-shaped distance matrix, with the innermost 
        # dimension representing the activity labels, the central 
        # dimension the predicted activities, and the outermost dimension 
        # the self.batch_size instances. (self.window_size+1 because 
        # first row and column stand for empty strings)
        d = torch.full(size=(self.batch_size, self.window_size+1, self.window_size+1), fill_value=0, dtype=torch.int64).to(device) # (B, WS+1, WS+1)

        # Initialize distances first row and column for each of the 
        # self.batch_size instances (empty strings)
        arange_tens = torch.arange(start=0, end=self.window_size+1, dtype=torch.int64).unsqueeze(0).to(device) # (1, WS+1)
        # First row for each instance 
        d[:, 0, :] = arange_tens
        # First column for each instance 
        d[:, :, 0] = arange_tens

        # Outer loop over predicted sequence (rows) for all of the instances
        # Note that in this loop, both the index for the rows and the 
        # index for the columns refer to the previous letters in both 
        # words. I.e. index 1 refers to 0th letter (1st, 0-based).
        for i in range(1, self.window_size+1): 
            # Inner loop over the ground-truth sequence (columns) for all of the instances
            for j in range(1, self.window_size+1):
                # At each position, make a (self.batch_size, ) shaped 
                # tensor containing the integer distances for each of the 
                # 4 possible operations. Then derive the minimum cost 
                # into a integer tensor 'min_cost' of shape 
                # (self.batch_size, ). Then d[:, i, j] = min_cost. 
                
                # Get (self.batch_size, )-shaped cost tensor for the 
                # substitution cost (0 or 1)
                cost = torch.where(self.suffix_acts_decoded[:, i-1]==self.act_labels[:, j-1], 0, 1) 
                
                # Get (self.batch_size, )-shaped distances for the 
                # respective cell, in case of deletion.
                deletion = d[:, i-1, j] + 1 # (self.batch_size, )

                # Get (self.batch_size, )-shaped distances for the 
                # respective cell, in case of insertion.
                insertion = d[:, i, j-1] + 1 # (self.batch_size, )

                # Get (self.batch_size, )-shaped distances for the 
                # respective cell, in case of substitution.
                substition = d[:, i-1, j-1] + cost # (self.batch_size, )

                # Update distance respective cell based on the 
                # cheapest option (deletion, insertion, substitution)
                d[:, i, j] = torch.minimum(torch.minimum(deletion, insertion), substition)

                # Check whether transposition would be cheaper. 
                if i > 1 and j > 1:
                    # Derive boolean tensor of shape (batch_size,), with 
                    # True indicating transposition is possible for 
                    # that respective instance. False if not. 
                    tpos_true = (self.suffix_acts_decoded[:, i-1]==self.act_labels[:, j-2]) & (self.suffix_acts_decoded[:, i-2]==self.act_labels[:, j-1])

                    # Computing minimum cost between 
                    # min(deletion, insertion, substitution), and 
                    # superposition, for all instances, even for the ones 
                    # for which superposition would not be possible. 
                    min_og_tpos = torch.minimum(d[:, i, j], d[:, i-2, j-2]+cost) # (self.batch_size, )

                    # Updating distance respective cell for those 
                    # instances for which transposition is possible and 
                    # cheaper. 
                    d[:, i, j] = torch.where(tpos_true, min_og_tpos, d[:, i, j])

        # Derive integer indexing tensor for the outermost (batch) 
        # dimension
        batch_arange = torch.arange(start=0, end=self.batch_size, dtype=torch.int64).to(device) # (self.batch_size, )

        # For each of the batch_size instances i, the dam lev distance is 
        # contained in the cell with row index len_pred[i] and column 
        # index len_actual[i]. Normalize by the longest distance between 
        # predicted and ground-truth suffix for each of the instances too.
        dam_lev_dist = d[batch_arange, len_pred, len_actual] / max_len  # (self.batch_size, )

        return dam_lev_dist
    
    def compute_rrt_results(self):
        """Compute MAE for the remaining runtime predictions, both in 
        the standardized scale, as well as the origianl scale of seconds. 
        """
        # Selecting only the labels and preds corresponding to the 
        # first decoding step. 
        rrt_labels_fin = self.rrt_labels[:, 0].clone() # (batch_size,)
        rrt_labels_fin_seconds = self.rrt_labels_seconds[:,0].clone() # (batch_size,)
        rrt_preds_fin = self.rrt_preds[:, 0].clone() # (batch_size, )
        rrt_preds_fin_seconds = self.rrt_preds_seconds[:, 0].clone() 

        # Computing absolute errors as-is, and in seconds. 
        abs_errors = torch.abs(rrt_preds_fin - rrt_labels_fin)
        abs_errors_seconds = torch.abs(rrt_preds_fin_seconds- rrt_labels_fin_seconds)

        return abs_errors, abs_errors_seconds # both (batch_size,)
    
    def compute_suf_length_diffs(self):
        """Compute different measures concerning the difference between 
        the predicted and ground-truth suffix lengths. 
        """
        too_early_bool = self.pred_length < self.actual_length # (batch_size,)
        too_late_bool = self.pred_length > self.actual_length # (batch_size, )
        length_diff = self.pred_length - self.actual_length # (batch_size,)
        length_diff_too_early = length_diff[too_early_bool].clone() 
        length_diff_too_late = length_diff[too_late_bool].clone()

        amount_right = torch.sum(length_diff == 0) # scalar tensor 

        return length_diff, length_diff_too_early, length_diff_too_late, amount_right