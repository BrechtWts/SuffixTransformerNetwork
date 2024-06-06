"""Contains the code for the re-implemented CRTP-LSTM benchmark 
(`CRTP_LSTM`), as well as for the non-data aware (NDA) version 
(`CRTP_LSTM_no_context`). 
"""

import torch
import torch.nn as nn
import torch.nn.init as init



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class CRTP_LSTM(nn.Module):
    def __init__(self, 
                 num_activities, 
                 d_model, 
                 cardinality_categoricals_pref, 
                 num_numericals_pref, 
                 dropout = 0.2, 
                 num_shared_LSTMlayers = 1,
                 num_dedicated_LSTMlayers = 1,
                 ):
        """Initialize an instance of the CRTP LSTM benchmark model [1]_.

        Parameters
        ----------
        num_activities : int
            Number of distinct activities present in the event log. 
            This does include the end token and padding token 
            used for the activity labels. For the categorical activity 
            label features in the prefix event tokens, no END token is 
            included. Hence, the amount of distinct levels there is 
            equal to `num_activities`-1. 
        d_model : int
            The total dimension of the vectors outputted by the 
            bi-directional LSTM layers. The hidden size for the bi-LSTM 
            blocks is hence d_model / 2. Therefore, d_model should be 
            an even integer. Original implementation [1]_ set this 
            parameter to 80. 
        cardinality_categoricals_pref : list of int
            List of `num_categoricals` integers. Each integer entry 
            i (i = 0, ..., `num_categoricals`-1) contains the cardinality 
            of the i'th categorical feature of the encoder prefix events. 
            The order of the cardinalities should match the order in 
            which the categoricals are fed as inputs. 
        num_numericals_pref : int 
            Number of numerical features of the prefix events. 
        dropout : float, optional
            by default 0.2.
        num_shared_LSTMlayers : int, optional
            Number of bidirectional LSTM layers shared between both 
            prediction heads. 1 by default, adhering to the 
            parameters of the original implementation [1]__. 
        num_dedicated_LSTMlayers : int, optional
            Number of dedicated bidirectional LSTM layers for each  
            prediction heads. 1 by default, adhering to the 
            parameters of the original implementation [1]__. 
        References
        ----------
        .. [1] B. R. Gunnarsson, S. v. Broucke and J. De Weerdt, "A 
               Direct Data Aware LSTM Neural Network Architecture for 
               Complete Remaining Trace and Runtime Prediction," in IEEE 
               Transactions on Services Computing, vol. 16, no. 4, pp. 
               2330-2342, 1 July-Aug. 2023, doi: 10.1109/TSC.2023.3245726.
        """
        super(CRTP_LSTM, self).__init__()
        self.num_activities = num_activities

        self.d_model = d_model

        # Cardinality categoricals encoder prefix events
        self.cardinality_categoricals_pref = cardinality_categoricals_pref
        self.num_categoricals_pref = len(self.cardinality_categoricals_pref)

        # Number of numerical features encoder prefix events 
        self.num_numericals_pref = num_numericals_pref

        self.dropout = dropout

        self.num_shared_LSTMlayers = num_shared_LSTMlayers
        self.num_dedicated_LSTMlayers = num_dedicated_LSTMlayers

        self.cardinality_categoricals_pref = [card+1 for card in self.cardinality_categoricals_pref]

        # Initializing the categorical embeddings for the encoder inputs: 
        self.embed_sz_categ_pref = [min(600, round(1.6 * n_cat**0.56)) for n_cat in self.cardinality_categoricals_pref]
        self.cat_embeds_pref = nn.ModuleList([nn.Embedding(num_embeddings=self.cardinality_categoricals_pref[i], embedding_dim=self.embed_sz_categ_pref[i]) for i in range(self.num_categoricals_pref)])

        # Dimensionality of initial prefix event tokens after concatenating embeddings 
        # and numericals
        self.dim_init_prefix = sum(self.embed_sz_categ_pref) + self.num_numericals_pref

        # Initialize dropout instance             
        self.dropout = nn.Dropout(self.dropout)


        # CRTP-LSTM components: 
        assert d_model % 2 == 0, "d_model must be an even number"
        self.hidden_size = self.d_model // 2 
        # Shared lstm block 
        self.lstm_shared = nn.LSTM(input_size=self.dim_init_prefix, hidden_size=self.hidden_size, num_layers=self.num_shared_LSTMlayers, batch_first=True, bidirectional=True)

        # Batchnorm over outputs 
        self.bn_shared = nn.BatchNorm1d(self.d_model)


        # activity trace lstm block 
        self.lstm_act = nn.LSTM(input_size=d_model, hidden_size=self.hidden_size, num_layers=self.num_dedicated_LSTMlayers, batch_first=True, bidirectional=True)

        # Batchnorm over outputs 
        self.bn_act = nn.BatchNorm1d(self.d_model)

        # Dense output layer activity predictions 
        self.fc_out_act = nn.Linear(self.d_model, self.num_activities)

        # remaining time sequence lstm block 
        self.lstm_rrt = nn.LSTM(input_size=d_model, hidden_size=self.hidden_size, num_layers=self.num_dedicated_LSTMlayers, batch_first=True, bidirectional=True)

        # Batchnorm over outputs 
        self.bn_rrt = nn.BatchNorm1d(self.d_model)

        # Dense output layer rrt predictions 
        self.fc_out_rrt = nn.Linear(self.d_model, 1)  

        # Initialize LSTM parameters with Glorot_uniform (as in benchmark)
        self.reset_parameters_lstm()

    def reset_parameters_lstm(self):
        # Manually initialize LSTM weights to mimic glorot_uniform
        # (like it was done in the original implementation)

        # Shared lstm layer(s)
        for name, param in self.lstm_shared.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)

        # Specialized Activity LSTM layer(s)
        for name, param in self.lstm_act.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)

        # Specialized Time Till Next Event (ttne) LSTM layer(s)
        for name, param in self.lstm_rrt.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)


    def forward(self, inputs):
        # Tensor containing the numerical features of the prefix events. 
        num_ftrs_pref = inputs[self.num_categoricals_pref] # (batch_size, window_size, N)
        cat_inputs = inputs[:self.num_categoricals_pref]


        # Constructing categorical embeddings prefix (encoder)
        cat_emb_pref = self.cat_embeds_pref[0](cat_inputs[0]) # (batch_size, window_size, embed_sz_categ[0])
        for i in range(1, self.num_categoricals_pref):
            cat_emb_help = self.cat_embeds_pref[i](cat_inputs[i]) # (batch_size, window_size, embed_sz_categ[i])
            cat_emb_pref = torch.cat((cat_emb_pref, cat_emb_help), dim = -1) # (batch_size, window_size, sum(embed_sz_categ[:i+1]))
        
        # Concatenate cat_emb with the numerical features to get initial vector representations prefix events. 
        x = torch.cat((cat_emb_pref, num_ftrs_pref), dim = -1) # (batch_size, window_size, sum(embed_sz_categ)+N)

        x = self.dropout(x)

        # Shared LSTM 
        shared_out, _ = self.lstm_shared(x) # (batch_size, window_size, d_model)

        # Applying batchnorm over the shared output
        shared_out = shared_out.permute(0, 2, 1) # (batch_size, d_model, window_size)
        shared_out = self.bn_shared(shared_out) # (batch_size, d_model, window_size)
        #   permuting again to original shape 
        shared_out = shared_out.permute(0, 2, 1) # (batch_size, window_size, self.d_model)


        # Activity LSTM 
        act_outputs, _ = self.lstm_act(shared_out) # (batch_size, window_size, d_model)

        # Applying batchnorm over the activity lstm's outputs 
        act_outputs = act_outputs.permute(0, 2, 1) # (batch_size, d_model, window_size)
        act_outputs = self.bn_act(act_outputs) # (batch_size, d_model, window_size)
        #   permute again to original shape 
        act_outputs = act_outputs.permute(0, 2, 1) # (batch_size, window_size, d_model)

        # Activity suffix prediction
        act_probs = self.fc_out_act(act_outputs) # (batch_size, window_size, num_classes)


        # Remaining RunTime (rrt) LSTM
        rrt_outputs, _ = self.lstm_rrt(shared_out) # (batch_size, window_size, d_model)

        # Applying batchnorm over rrt lstm outputs
        rrt_outputs = rrt_outputs.permute(0, 2, 1) # (batch_size, d_model, window_size)
        rrt_outputs = self.bn_rrt(rrt_outputs) # (batch_size, d_model, window_size)

        #   permute again to the original shape 
        rrt_outputs = rrt_outputs.permute(0, 2, 1) # (batch_size, window_size, d_model)

        # rrt suffix prediction 
        rrt_pred = self.fc_out_rrt(rrt_outputs) # (batch_size, window_size, 1)

        return act_probs, rrt_pred
    

class CRTP_LSTM_no_context(nn.Module):
    def __init__(self, 
                 num_activities, 
                 d_model, 
                 dropout = 0.1, 
                 num_shared_LSTMlayers = 1,
                 num_dedicated_LSTMlayers = 1,
                 ):
        """Initialize an instance of the CRTP LSTM benchmark model [1]_.

        Parameters
        ----------
        num_activities : int
            Number of distinct activities present in the event log. 
            This does include the end token and padding token 
            used for the activity labels. For the categorical activity 
            label features in the prefix, no END token is 
            included. Hence, the amount of distinct levels there is 
            equal to `num_activities`-1. 
        d_model : int
            The total dimension of the vectors outputted by the 
            bi-directional LSTM layers. The hidden size for the bi-LSTM 
            blocks is hence d_model / 2. Therefore, d_model should be 
            an even integer. Original implementation [1]_ set this 
            parameter to 80. 
        dropout : float, optional
            Dropout rate. By default 0.2.
        num_shared_LSTMlayers : int, optional
            Number of bidirectional LSTM layers shared between both 
            prediction heads. 1 by default, adhering to the 
            parameters of the original implementation [1]__. 
        num_dedicated_LSTMlayers : int, optional
            Number of dedicated bidirectional LSTM layers for each  
            prediction heads. 1 by default, adhering to the 
            parameters of the original implementation [1]__. 
        References
        ----------
        .. [1] B. R. Gunnarsson, S. v. Broucke and J. De Weerdt, "A 
               Direct Data Aware LSTM Neural Network Architecture for 
               Complete Remaining Trace and Runtime Prediction," in IEEE 
               Transactions on Services Computing, vol. 16, no. 4, pp. 
               2330-2342, 1 July-Aug. 2023, doi: 10.1109/TSC.2023.3245726.
        """
        super(CRTP_LSTM_no_context, self).__init__()
        self.num_activities = num_activities

        self.d_model = d_model

        self.dropout = dropout

        self.num_shared_LSTMlayers = num_shared_LSTMlayers
        self.num_dedicated_LSTMlayers = num_dedicated_LSTMlayers

        self.activity_emb_size = min(600, round(1.6 * (self.num_activities-1)**0.56))
        self.act_emb = nn.Embedding(num_embeddings=num_activities-1, embedding_dim=self.activity_emb_size)

        # Dimensionality of initial prefix event tokens after 
        # concatenating activty embedding with the two time-related 
        # features of the prefix events.  
        self.dim_init_prefix = self.activity_emb_size + 2
            
        self.dropout = nn.Dropout(self.dropout)


        # CRTP-LSTM components: 
        assert d_model % 2 == 0, "d_model must be an even number"
        self.hidden_size = self.d_model // 2 
        # Shared lstm block 
        self.lstm_shared = nn.LSTM(input_size=self.dim_init_prefix, hidden_size=self.hidden_size, num_layers=self.num_shared_LSTMlayers, batch_first=True, bidirectional=True)

        # Batchnorm over outputs 
        self.bn_shared = nn.BatchNorm1d(self.d_model)


        # activity trace lstm block 
        self.lstm_act = nn.LSTM(input_size=d_model, hidden_size=self.hidden_size, num_layers=self.num_dedicated_LSTMlayers, batch_first=True, bidirectional=True)

        # Batchnorm over outputs 
        self.bn_act = nn.BatchNorm1d(self.d_model)

        # Dense output layer activity predictions 
        self.fc_out_act = nn.Linear(self.d_model, self.num_activities)

        # remaining time sequence lstm block 
        self.lstm_rrt = nn.LSTM(input_size=d_model, hidden_size=self.hidden_size, num_layers=self.num_dedicated_LSTMlayers, batch_first=True, bidirectional=True)

        # Batchnorm over outputs 
        self.bn_rrt = nn.BatchNorm1d(self.d_model)

        # Dense output layer rrt predictions 
        self.fc_out_rrt = nn.Linear(self.d_model, 1)  

        # Initialize LSTM parameters with Glorot_uniform (as in benchmark)
        self.reset_parameters_lstm()

    def reset_parameters_lstm(self):
        # Manually initialize LSTM weights to mimic glorot_uniform
        # (like it was done in the original implementation. )

        # Shared lstm layer(s)
        for name, param in self.lstm_shared.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)

        # Specialized Activity LSTM layer(s)
        for name, param in self.lstm_act.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)

        # Specialized Time Till Next Event (ttne) LSTM layer(s)
        for name, param in self.lstm_rrt.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)


    def forward(self, inputs):
        # Tensor containing the numerical features of the prefix events. 
        num_ftrs_pref = inputs[1] # (batch_size, window_size, 2)
        act_emb_pref = self.act_emb(inputs[0]) # (batch_size, window_size, self.activity_emb_size)
        
        # Concatenate cat_emb with the numerical features to get initial vector representations prefix events. 
        x = torch.cat((act_emb_pref, num_ftrs_pref), dim = -1) # (batch_size, window_size, sum(embed_sz_categ)+N)

        x = self.dropout(x)

        # Shared LSTM 
        shared_out, _ = self.lstm_shared(x) # (batch_size, window_size, d_model)

        # Applying batchnorm over the shared output
        shared_out = shared_out.permute(0, 2, 1) # (batch_size, d_model, window_size)
        shared_out = self.bn_shared(shared_out) # (batch_size, d_model, window_size)
        #   permuting again to original shape 
        shared_out = shared_out.permute(0, 2, 1) # (batch_size, window_size, self.d_model)

        # Activity LSTM 
        act_outputs, _ = self.lstm_act(shared_out) # (batch_size, window_size, d_model)

        # Applying batchnorm over the activity lstm's outputs 
        act_outputs = act_outputs.permute(0, 2, 1) # (batch_size, d_model, window_size)
        act_outputs = self.bn_act(act_outputs) # (batch_size, d_model, window_size)
        #   permute again to original shape 
        act_outputs = act_outputs.permute(0, 2, 1) # (batch_size, window_size, d_model)

        # Activity suffix prediction
        act_probs = self.fc_out_act(act_outputs) # (batch_size, window_size, num_classes)


        # Remaining RunTime (rrt) LSTM
        rrt_outputs, _ = self.lstm_rrt(shared_out) # (batch_size, window_size, d_model)

        # Applying batchnorm over rrt lstm outputs
        rrt_outputs = rrt_outputs.permute(0, 2, 1) # (batch_size, d_model, window_size)
        rrt_outputs = self.bn_rrt(rrt_outputs) # (batch_size, d_model, window_size)

        #   permute again to the original shape 
        rrt_outputs = rrt_outputs.permute(0, 2, 1) # (batch_size, window_size, d_model)

        # rrt suffix prediction 
        rrt_pred = self.fc_out_rrt(rrt_outputs) # (batch_size, window_size, 1)

        return act_probs, rrt_pred