"""Contains the Inference Environment supporting the iterative feedback 
loop needed for leveraging the SEP benchmark model(s) for suffix 
generation (activity suffix and Time Till Next Event aka TTNE suffix). 
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BatchInference():
    def __init__(self,
                 inputs, 
                 labels,
                 num_categoricals_pref, 
                 num_numericals_pref, 
                 mean_std_ttne, 
                 mean_std_tsp, 
                 mean_std_tss, 
                 mean_std_rrt):
        """Initialize instance of inference environment for the SEP-LSTM 
        benchmark. Contains methods for computing the different 
        evaluation metrics for activity suffix, timestamp suffix and 
        remaining runtime predictions. Also contains functionality 
        facilitating the external feedback loop needed for deploying SEP-
        LSTM for suffix generation. In particular, keeps track of the 
        generation progress over the iterations, and creates new event 
        tokens based on the activity and timestamp predictions, adding 
        them to the current prefix and thereby updating the prefix events 
        to be inputted for the next decoding step. 

        Parameters
        ----------
        inputs : list of torch.Tensor 
            Containing tensors pertaining to prefix event tokens. For the 
            NDA SEP-LSTM benchmark, this only includes an integer-encoded 
            tensor containing the prefix events' activities, and a tensor 
            containing both numerical timestamp proxies (time since start 
            aka tss, and time since previous event aka tsp).
        labels : list of torch.Tensor 
            List of torch.Tensors containing the labels for the three 
            prediction targets, being the timestamp suffix, the remaining 
            runtime and the activity label suffix. 
        num_categoricals_pref : int
            The number of categorical features (including the activity 
            labels) in the prefix event tokens. Given the NDA character 
            of SEP-LSTM, this should be equal to 1. Not fixed to retain 
            flexibility for future research. 
        num_numericals_pref : int
            The number of numerical features in the prefix event tokens. 
            Given the NDA character of SEP-LSTM, this should be equal to 
            2, referring to the two numerical features serving as the  
            timestamp proxy (time since start aka tss, and time since 
            previous event aka tsp). Not fixed to retain flexibility for 
            future research. 
        mean_std_ttne : list of float
            Training mean and standard deviation used to standardize the time 
            till next event (in seconds) target. Needed for re-converting 
            ttne predictions to original scale. Mean is the first entry, 
            std the second.
        mean_std_tsp : list of float
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
        mean_std_tss : list of float
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
        mean_std_rrt : list of float
            List consisting of two floats, the training mean and standard 
            deviation of the remaining runtime labels (in seconds). Needed 
            for de-standardizing remaining runtime predictions and labels, 
            such that the MAE can be expressed in seconds (and minutes). 
        """
        
        self.include_ttne_end = False
        
        self.inputs = inputs 

        self.ttne_labels = labels[0].clone() # (batch_size, window_size, 1)
        self.ttne_labels = self.ttne_labels[:, :, 0] # (batch_size, window_size)
        self.rrt_labels = labels[1].clone() # (batch_size, window_size, 1)
        self.act_labels = labels[2].clone() # (batch_size, window_size)

        self.window_size = self.act_labels.shape[-1]
        self.batch_size = self.act_labels.shape[0]
        # End Token gets last index, and should be contained 
        # within the label tensor for all of the prefix-suffix 
        # pairs. That's how we derive the num_classes from it. 
        self.num_classes = torch.max(self.act_labels).item() + 1
        self.num_categoricals_pref = num_categoricals_pref
        self.num_numericals_pref = num_numericals_pref

        self.mean_std_ttne = mean_std_ttne
        self.mean_std_tsp = mean_std_tsp
        self.mean_std_tss = mean_std_tss
        self.mean_std_rrt = mean_std_rrt

        # Getting (batch_size, )-shaped torch.int64 tensor 
        # ``self.actual_length``containing for each of the batch_size 
        # instances the suffix length (0-based)
        self.actual_length = self.get_actual_length()

        # Initialize predicted_length with window_size -1 (0-based) values.
        self.pred_length = torch.full(size=(self.batch_size,), fill_value=self.window_size-1, dtype=torch.int64).to(device)

        # Initialize boolean tensor to track of the instances for which 
        # the model already predicted them to be finished during 
        # decoding. True if model predicted them to be over. 
        self.end_predicted = torch.full(size=(self.batch_size,), fill_value=False).to(device) # (batch_size,) torch.bool

        # Initialize (batch_size, window_size)-shaped torch.int64 tensor 
        # to keep track of the decoded activities for each of the  
        # batch_size instances as integers. 
        self.suffix_acts_decoded = torch.full(size=(self.batch_size, self.window_size), fill_value=0, dtype=torch.int64).to(device)

        # Initialize (batch_size, window_size)-shaped torch.float32 tensor 
        # to keep track of the standardized ttne predictions 
        self.suffix_ttne_preds = torch.full(size=(self.batch_size, self.window_size), fill_value=0, dtype=torch.float32).to(device) 

        # Converting the ttne suffix labels from standardized back into original scale. 
        self.ttne_labels_seconds = self.convert_to_seconds(time_string='ttne', input_tensor=self.ttne_labels.clone())

        # Extending left paddings in the prefix feature tensors 
        self.extend_prefix_leftpaddings()

    def extend_prefix_leftpaddings(self):
        """Extend the leftpadding of the prefix inputs from window_size 
        to 2*window_size. This is needed for the iterative feedback loop 
        being able to read the full prefix, including its own predictions 
        made prior to a certain decoding step. 

        This is done prior to the start of the iterative feedback loop. 
        """
        # Updating activity labels in prefix events
        # act_pref_inputs = self.inputs[0].clone() # (B, W, num_classes-2)
        act_pref_inputs = self.inputs[0].clone() # (B, W)
        zeros_tensor = torch.zeros_like(act_pref_inputs, dtype=torch.int64, device=device) # (B, W)
        act_pref_inputs = torch.cat((zeros_tensor, act_pref_inputs), dim=-1) # (B, 2*W)
        
        # Inserting it again in the inputs list 
        self.inputs[0] = act_pref_inputs

        # Updating the categorical case features (if any)
        for i in range(1, self.num_categoricals_pref):
            cat_ftr = self.inputs[i].clone() # (batch_size, W)
            zeros_tensor = torch.zeros_like(cat_ftr, dtype=torch.int64, device=device) # (B, W)
            cat_ftr = torch.cat((zeros_tensor, cat_ftr), dim=-1) # (B, 2*W)

            # Inserting it in the inputs list 
            self.inputs[i] = cat_ftr # (B, 2*W)
        
        # Updating numerics tensor prefix events 
        num_ftrs_pref = self.inputs[self.num_categoricals_pref].clone() # (B, W, num_numericals_pref)
        zeros_tensor = torch.zeros_like(num_ftrs_pref, dtype=torch.float32, device=device) # (B, W, num_numericals_pref)
        num_ftrs_pref = torch.cat((zeros_tensor, num_ftrs_pref), dim=1) # (B, 2*W, num_numericals_pref)
        self.inputs[self.num_categoricals_pref] = num_ftrs_pref



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
    

    def convert_to_seconds(self, time_string, input_tensor):
        """Convert a tensor of any shape, containing time-related 
        features, predictions or labels, that are standardized, back 
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
        
        converted_tensor = input_tensor*train_std + train_mean
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
        
        converted_tensor = (input_tensor - train_mean) / train_std

        return converted_tensor # Same shape as input_tensor
    
    def decode_step(self, 
                    act_preds,
                    ttne_preds,
                    dec_step):
        """Computes the values for the next decoder suffix token (activity 
        label, time since previous event and time since start) based on 
        decoding the predicted activity logits (according to ``decode_type``) 
        and the time till next event predictions. Thereby also updates 
        the corresponding tensors in ``self.inputs``, which are used for 
        the next decoding step by the external inference loop.

        Parameters
        ----------
        act_preds : torch.Tensor 
            Dtype torch.float32; the unnormalized logits for the next 
            activity prediction. Shape (batch_size, num_classes).
        ttne_preds : torch.Tensor 
            Dtype torch.float32; predictions for time till next event. 
            Shape (batch_size, 1).
        dec_step : int
            The decoding step, 0-based. First decodign step corresponds 
            to ``decode_step=0``. 
        """
        # if decode_type == 'greedy':
        #     if dec_step < (self.window_size-1):

        ttne_preds = ttne_preds[:, 0] # (batch_size, )

        # Keeping track of (standardized) ttne_preds 
        self.suffix_ttne_preds[:, dec_step] = ttne_preds


        # # Keeping track of the unnormalized activity logits 
        # self.act_preds_batch[:, dec_step, :] = act_preds

        #--------------- start 
        # Masking out the padding token 
        act_preds[:, 0] = -1e9 # (B, W)

        #--------------- end

        if dec_step < (self.window_size-1):
            
            # Selecting for each instance the next activity based on max prob
            act_selected = torch.argmax(act_preds, dim=-1) # (batch_size,), torch.int64

            # Appending the predicted next activities only for those 
            # instances that have already ended 
            unfinished = self.end_predicted==False # (batch_size,)
            self.suffix_acts_decoded[unfinished, dec_step] = act_selected[unfinished]

            # Deriving which instances have just finished 
            just_finished = (act_selected==self.num_classes-1) & unfinished

            # Updating pred_length
            self.pred_length[just_finished] = dec_step
            
            self.end_predicted[just_finished] = True 

            # Updating the prefix activity labels 
            # act_pref_inputs = self.inputs[0].clone() # (B, W, num_classes-2)
            act_pref_inputs = self.inputs[0].clone() # (B, W)


            # Discarding first (left-padded) padding token 
            # act_pref_inputs_new = act_pref_inputs[:, 1:, :].clone() # (B, W-1, C-2)
            act_pref_inputs_new = act_pref_inputs[:, 1:].clone() # (B, W-1)

            # Deriving activity indices pertaining to the 
            # selected activities for the derived next suffix 
            # event to be fed to the decoder in the next decoding 
            # step. 
            act_pref_updates = act_selected.clone() # (batch_size, )
            
            #   There is no artificially added END token present in the 
            #   prefix activity representations, and hence there is no 
            #   end token index in the prefix activity representations 
            #   on index num_classes-1. Therefore, we clamp 
            #   it on num_classes-2. Predictions for already finished 
            #   instances will not be taken into account at the end. 
            act_pref_updates = torch.clamp(act_pref_updates, max=self.num_classes-2) # (batch_size,)

            #   Finally concatenating those updates at the end 
            act_pref_inputs_new = torch.cat((act_pref_inputs_new, act_pref_updates.unsqueeze(-1)), dim=-1) # (batch_size, W)

            self.inputs[0] = act_pref_inputs_new


            # Auxiliary zeros tensor to shift current prefix features to 
            # the left 
            for i in range(1, self.num_categoricals_pref):
                # Update each of the categorical case features if any 
                cat_ftr = self.inputs[i].clone() # (batch_size, W, card_i)

                # Discarding the first (left-padded) padding token 
                cat_ftr_new = cat_ftr[:, 1:, :].clone() # (batch_size, W-1, card_i)

                # Concatenating duplicated case feature at the end 
                cat_ftr_new = torch.cat((cat_ftr_new, cat_ftr_new[:, -1, :][:, None, :]), dim=1) # (batch_size, W, card_i)
                self.inputs[i] = cat_ftr_new
            



            # Update tss and tsp time features 
            num_ftrs_pref = self.inputs[self.num_categoricals_pref].clone() # (B, W, num_numericals_pref)

            tss_tsp = num_ftrs_pref[:, :, :2].clone() # (B, W, 2)

            # Discarding first (left-padded) padding token 
            tss_tsp_new = tss_tsp[:, 1:, :].clone() # (B, W-1, 2)


            # Computing new time since start (tss) and time since 
            # previous (tsp) features based on TTNE predictions 
            ttne_preds_seconds = self.convert_to_seconds(time_string='ttne', input_tensor=ttne_preds.clone()) # (batch_size, )

            # Getting the latest standardized tss and tsp 
            tss_tsp_latest = tss_tsp[:, -1, :] # (B, 2)

            tss_stand = tss_tsp_latest[:, 0].clone() # (B, )

            # Converting into seconds 
            tss_seconds = self.convert_to_seconds(time_string='tss', input_tensor=tss_stand.clone()) # (B,)

            # Computing new tss feature predicted next event (in seconds, 
            # before standardizing it again based on the training set mean 
            # and standard deviation of the tss prefix event feature).
            tss_seconds_plus1 = tss_seconds + ttne_preds_seconds # (batch_size, )
            tss_plus1_stand = self.convert_to_transf(time_string='tss', input_tensor=tss_seconds_plus1) # (batch_size, )

            # Computing new tsp feature (equal to ttne prediction in seconds)
            tsp_plus1_stand = self.convert_to_transf(time_string='tsp', input_tensor=ttne_preds_seconds.clone())  # (batch_size,)

            tss_tsp_plus1 = torch.cat((tss_plus1_stand[:, None], tsp_plus1_stand[:, None]), dim=-1) # (batch_size, 2)

            tss_tsp_new = torch.cat((tss_tsp_new, tss_tsp_plus1[:, None, :]), dim=1) # (batch_size, W, 2)

            # Updating the numerics 
            self.inputs[self.num_categoricals_pref] = tss_tsp_new # (B, W, num_numericals_pref) = (B, W, 2)
            

            # For last decoding step, we just 'force' the model to predict end token
            # (I.e. we just ignore its preds if it has not predicted end token yet.)
        elif dec_step == (self.window_size-1):

            unfinished = self.end_predicted==False # (batch_size,)
            self.suffix_acts_decoded[unfinished, dec_step] = self.num_classes-1


            # Updating pred_length
            self.pred_length[unfinished] = dec_step # Not needed. Initialized at window_size-1

    def compute_ttne_results(self):
        """NOTE: 
        for the SEP-LSTM, this method, since it also computes the 
        remaining runtime (RRT) predictions based on the sum of TTNE 
        predictions (in seconds) until the decoding index at which the 
        ED-LSTM predicted the END token for the activity suffix, should 
        always be called prior to the `compute_rrt_results()` method. 
        
        Compute the absolute errors for the timestamp (Time Till Next 
        Event, aka TTNE) suffix predictions in both the standardized 
        (~N(0,1)) and the original (in seconds) scale. Furthermore, 
        compute the RRT predictions (in seconds) derived from the sum of  
        TTNE predictions up until the index of the predicted END token 
        and save it as a variable of the class instance. Will be used 
        in the `compute_rrt_results()` method, which should always 
        be called after this method. 

        Each of the `batch_size` prefix-suffix pairs 
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
        replaced with a zero(-equivalent). Additionally, since decoded 
        activity labels fed to the decoder after the prediction of the 
        END token do not contain any informational value, these 
        timestamp predictions would not make any sense anyway. 

        Returns
        -------
        MAE_ttne_stand : torch.Tensor 
            Absolute errors standardized timestamp predictions for all 
            actual / non-padded suffix events. Shape (NP,), with NP being 
            equal to the total number of actual suffix events (including 
            the added END tokens) over all `batch_size` 
            prefix-suffix pairs in the validation or test set. 
        MAE_ttne_seconds : torch.Tensor 
            The absolute errors pertaining to the same predictions as the 
            errors contained within `MAE_ttne_stand`, but after first 
            having converted their respective predictions and labels back 
            into the original scale (in seconds). (This is done based on 
            the training mean and standard deviation of the actual / non-
            padded TTNE values, contained within `self.mean_std_ttne`.)
        rrt_sum_seconds : torch.Tensor 
            The remaining runtime (RRT) predictions derived from the 
            predicted TTNE suffix (converted in seconds). The RRT 
            prediction is derived by computing the sum of the, in seconds 
            converted, TTNE predictions until the index at which the 
            model predicted the END token in the activity suffix. 
        """
        # De-standardizing the decoded predictions
        self.suffix_ttne_preds_seconds = self.convert_to_seconds(time_string='ttne', input_tensor=self.suffix_ttne_preds.clone())
        # self.suffix_ttne_preds_seconds = self.suffix_ttne_preds_seconds*self.mean_std_ttne[1] + self.mean_std_ttne[0] # (batch_size, window_size)
        
        # Defining the 0-equivalent value for the standardized predictions and labels
        stand_0_eq = -self.mean_std_ttne[0]/self.mean_std_ttne[1]


        # --------------------------------------------
        #       Generate a tensor with values counting from 0 to window_size
        counting_tensor = torch.arange(self.window_size, dtype=torch.int64).to(device) # (window_size,)
        #       Repeat the tensor along the first dimension to match the desired shape
        counting_tensor = counting_tensor.unsqueeze(0).repeat(self.batch_size, 1).to(device) # (batch_size, window_size)

        # Deriving boolean tensor of shape (batch_size, window_size) 
        # containing True for the indices after its END token prediction 
        # in the predicted activity suffix 
        pad_preds = counting_tensor > self.pred_length.unsqueeze(-1) # (batch_size, window_size)

        # Padding both prediction tensors 
        self.suffix_ttne_preds_seconds[pad_preds] = 0 # (batch_size, window_size)
        self.suffix_ttne_preds[pad_preds] = stand_0_eq # (batch_size, window_size)

        # Deriving boolean tensor of shape (batch_size, window_size) with 
        # True on the indices before and on the ground truth END token 
        before_end_token = counting_tensor <= self.actual_length.unsqueeze(-1)

        # Computing MAE ttne elements and slicing out the considered ones 
        MAE_ttne_stand = torch.abs(self.suffix_ttne_preds - self.ttne_labels) # (batch_size, window_size)
        MAE_ttne_stand = MAE_ttne_stand[before_end_token] # shape (torch.sum(self.actual_length+1), )

        MAE_ttne_seconds = torch.abs(self.suffix_ttne_preds_seconds - self.ttne_labels_seconds) # (batch_size, window_size)
        MAE_ttne_seconds = MAE_ttne_seconds[before_end_token] # shape (torch.sum(self.actual_length+1), )

      # Deriving an RRT prediction in seconds based on the SUM of TTNE 
        # predictions until (and including) the index of the PREDICTED 
        # end token. 
        # Given the 0 padding of TTNE predictions after predicted END 
        # token, this is obtained by simply computing the sum over the 
        # trailing dimension of the predictions. 
        self.rrt_sum_seconds = self.suffix_ttne_preds_seconds.clone() # (batch_size, window_size)

        if self.include_ttne_end:
            self.rrt_sum_seconds = torch.sum(self.rrt_sum_seconds, dim=-1) # (batch_size, )
        else: 
            # Not include ttne predition on index predicted END token activity suffix
            # in sum TTNE suffix computed to derive RRT prediction. This boils down 
            # to padding the ttne prediction (in seconds) on that index with 0 as well. 
            
            # Replacing the specified indices with 0
            self.rrt_sum_seconds[torch.arange(self.batch_size), self.pred_length] = 0.0 # (batch_size, window_size)
            self.rrt_sum_seconds = torch.sum(self.rrt_sum_seconds, dim=-1) # (batch_size, )

        # Due to original procedure computing distorted TTNE metrics, the 
        # metrics are returned twice. Will be resolved soon
        return MAE_ttne_stand, MAE_ttne_seconds, MAE_ttne_stand, MAE_ttne_seconds
    
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
        """Compute MAE for remaining runtime predictions (in seconds). 
        NOTE, 
        the RRT predictions are implicitly derived from the predicted 
        timestamp suffix, and computed within the 
        `compute_ttne_results()` method. Therefore, the 
        `compute_ttne_results()` should always be called before 
        executing the `compute_rrt_results()` method. 
        """
        # Selecting only the labels and preds corresponding to the 
        # first decoding step. 
        self.rrt_labels = self.rrt_labels[:, 0, 0] # (batch_size,)

        # De-standardizing to seconds. 
        self.rrt_labels_seconds = self.convert_to_seconds(time_string='rrt', input_tensor=self.rrt_labels.clone())

        abs_errors_sum = torch.abs(self.rrt_sum_seconds - self.rrt_labels_seconds)

        return abs_errors_sum # both (batch_size,)

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