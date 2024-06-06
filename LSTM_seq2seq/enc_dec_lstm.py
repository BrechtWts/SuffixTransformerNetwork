import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncDecLSTM_no_context(nn.Module):
    def __init__(self, 
                 num_activities, 
                 d_model, 
                 n_layers, 
                 dropout = 0.2):
        """Initialize instance of the ED-LSTM benchmark. 

        Parameters
        ----------
        num_activities : int
            Number of distinct activities present in the event log. 
            This does include the end token and padding token 
            used for the activity labels. For the categorical activity 
            label features in the prefix and suffix event tokens, no END 
            token is included. Hence, the amount of distinct levels there 
            is equal to `num_activities`-1. 
        d_model : int 
            Dimensionality of the hidden layers in the LSTM encoder and 
            decoder layers. 
        n_layers : int
            Number of LSTM layers in the LSTM encoder and decoder. 
        dropout : float, optional
            Dropout, by default 0.2
        """
        super(EncDecLSTM_no_context, self).__init__()
        self.num_activities = num_activities
        self.d_model = d_model
        self.dropout = dropout

        # Deriving embedding dimensionality for the prefix (and suffix) 
        # event tokens' activity embedding. 
        self.activity_emb_size = min(600, round(1.6 * (self.num_activities-2)**0.56))
        
        # Initialize activity embedding layer shared between prefix and 
        # suffix event tokens 
        self.act_emb = nn.Embedding(num_embeddings=num_activities-1, embedding_dim=self.activity_emb_size, padding_idx=0)

        # Dimensionality of initial encoder events after the prefix categoricals are fed to the dedicated entity embeddings and everything, including the numericals 
        # are concatenated
        self.dim_init_prefix = self.activity_emb_size + 2

        # The dimensionality of the initial suffix event tokens, consisting of the concatenation 
        # of the activity embedding + the 2 time features
        self.dim_init_suffix = self.activity_emb_size + 2


        # Encoder LSTM 
        self.encoder = nn.LSTM(input_size=self.dim_init_prefix, hidden_size=d_model, num_layers=n_layers, batch_first=True, dropout=dropout)

        # Decoder LSTM 
        self.decoder = nn.LSTM(input_size=self.dim_init_suffix, hidden_size=d_model, num_layers=n_layers, batch_first=True, dropout=dropout)


        # Initialize the additional activity output layer
        self.fc_out_act = nn.Linear(self.d_model, self.num_activities) # (batch_size, window_size, num_activities)

        # Initialize the additional time till next event prediction layer
        self.fc_out_ttne = nn.Linear(self.d_model, 1)


        
        
    # window_size : number of decoding steps during inference (model.eval())
    def forward(self, 
                inputs, 
                window_size=None, 
                mean_std_ttne=None, 
                mean_std_tsp=None, 
                mean_std_tss=None):
        # Tensor containing the numerical features of the prefix events. 
        num_ftrs_pref = inputs[1] # (batch_size, window_size, N)

        num_ftrs_suf = inputs[4] # (batch_size, window_size, 2)

        act_emb_pref = self.act_emb(inputs[0]) # (batch_size, window_size, self.activity_emb_size)

        # Concatenate cat_emb with the numerical features to get initial vector representations prefix events. 
        x = torch.cat((act_emb_pref, num_ftrs_pref), dim = -1) # (batch_size, window_size, sum(embed_sz_categ)+N)

        # Feeding prefix sequence to encoder and retrieving final hidden 
        # and cell states 
        _, (hidden, cell) = self.encoder(x) # both of shape (num_layers, batch_size, d_model)

        # Teacher forcing during training. 
        if self.training:
            # Using the activity embedding layer shared with the encoder 
            cat_emb_suf = self.act_emb(inputs[3]) # (batch_size, window_size, self.activity_emb_size)
            # Concatenate cat_emb with the numerical features to get initial vector representations suffix event tokens.
            target_in = torch.cat((cat_emb_suf, num_ftrs_suf), dim = -1) # (batch_size, window_size, self.dim_init_suffix)

            output, _ = self.decoder(target_in, (hidden, cell)) # (batch_size, window_size, d_model)

            act_logits = self.fc_out_act(output) # (batch_size, window_size, num_activities)
            time_preds = self.fc_out_ttne(output) # (batch_size, window_size, 1)

            return act_logits, time_preds


        else:
            # Inference mode
            # Retrieving first suffix activity one-hot vector 
            act_inputs = inputs[3] # (batch_size, window_size)
            act_inputs = act_inputs[:, 0].unsqueeze(1) # (batch_size, 1)

            # Retrieving time information start event 
            num_ftrs_suf = num_ftrs_suf[:, 0, :].unsqueeze(1) # (batch_size, 1, 2)

            batch_size = act_inputs.size(0)

            # Initialize tensors for storing final activity and timestamp suffix 
            # predictions. 
            suffix_acts_decoded = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.int64).to(device)
            suffix_ttne_preds = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.float32).to(device)
            
            for dec_step in range(0, window_size):
                # Leveraging learned embedding 
                cat_emb_suf = self.act_emb(act_inputs) # (batch_size, 1, self.activity_emb_size)

                # Concatenating both
                target_in = torch.cat((cat_emb_suf, num_ftrs_suf), dim = -1) # (batch_size, 1, self.dim_init_suffix)

                output, (hidden, cell) = self.decoder(target_in, (hidden, cell)) # (batch_size, 1, d_model)
                act_logits = self.fc_out_act(output) # (batch_size, 1, num_activities)
                time_preds = self.fc_out_ttne(output) # (batch_size, 1, 1)
                
                # Adding time pred as-is 
                suffix_ttne_preds[:, dec_step] = time_preds[:,0,0]

                act_outputs = act_logits[:,0,:] # (batch_size, num_activities)

                # Masking out the padding token (positioned at index 0)
                act_outputs[:, 0] = -1e9 # (batch_size, num_activities)

                # Greedy selection
                act_selected = torch.argmax(act_outputs, dim=-1) # (batch_size,), torch.int64

                # Adding selected activity integers to suffix_acts_decoded
                suffix_acts_decoded[:, dec_step] = act_selected

                # Deriving activity indices pertaining to the 
                # selected activities for the derived next suffix 
                # event to be fed to the decoder in the next decoding 
                # step. 
                act_suf_updates = act_selected.clone() # (batch_size, )
                
                #   There is no artificially added END token present in the 
                #   suffix activity representations, and hence there is no 
                #   end token index in the suffix activity representations 
                #   on index num_activities-1. Therefore, we clamp 
                #   it on num_activities-2. Predictions for already finished 
                #   instances will not be taken into account at the end. 
                act_suf_updates = torch.clamp(act_suf_updates, max=self.num_activities-2) # (batch_size,)

                # Updating the activity label fed to the decoder of the model 
                # in the next decoding step. 
                act_inputs = act_suf_updates.unsqueeze(-1) # (batch_size, 1)

                # Computing the time since start and time since previous event 
                # time features of created suffix event. 
                
                #   Since they features and timestamp labels are each 
                #   standardized indvidually, they have to be converted 
                #   into their original scale first, after which they 
                #   are standardized again. 

                time_preds_seconds = time_preds[:,0,0]*mean_std_ttne[1] + mean_std_ttne[0] # (batch_size,)
                # Truncating at zero (no negatives allowed)
                time_preds_seconds = torch.clamp(time_preds_seconds, min=0)

                tss_seconds = num_ftrs_suf[:, 0, 0]*mean_std_tss[1] + mean_std_tss[0] # (batch_size,)
                tss_seconds = torch.clamp(tss_seconds, min=0)
                tss_seconds_new = tss_seconds + time_preds_seconds # (batch_size,)
                tss_stand_new = (tss_seconds_new - mean_std_tss[0]) / mean_std_tss[1] # (batch_size,)

                # The new time since previous event is the rescaled prediction
                tsp_stand_new = (time_preds_seconds - mean_std_tsp[0]) / mean_std_tsp[1] # (batch_size,)

                # Finally updating the new time features of the decoder suffix 
                num_ftrs_suf = torch.cat((tss_stand_new.unsqueeze(-1), tsp_stand_new.unsqueeze(-1)), dim=-1).unsqueeze(1) # (batch_size, 1, 2)
            
            
            return suffix_acts_decoded, suffix_ttne_preds