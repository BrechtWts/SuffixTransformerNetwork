import torch
import torch.nn as nn
import torch.nn.init as init


class SEP_Benchmarks_time(nn.Module):
    def __init__(self, 
                 d_model,
                 num_activities, 
                 dropout, 
                 num_shared_layers, 
                 num_specialized_layers):
        """Re-implementation of the SEP-benchmark based on the model 
        proposed by Tax et al. For a fair and controlled experiment, 
        the benchmark leverages the same activity embeddings compared 
        to all other models, instead of the one-hot encoding used 
        by Tax et al. Furthermore, identical to all other NDA 
        implementations, the prefix event tokens are represented by 
        the activity label, and two numerical time proxies. 

        Parameters
        ----------
        d_model : int
            Hidden size of the LSTM layers, and hence dimensionality 
            of the model. 
        num_activities : int
            Number of distinct activities present in the event log. 
            This does include the end token and padding token 
            used for the activity labels. For the categorical activity 
            label features in the prefix and suffix, no END token is 
            included. Hence, the amount of distinct levels there is 
            equal to `num_activities`-1. 
        dropout : float
            Dropout rate. 
        num_shared_layers : int
            Number of shared LSTM layers. Tax et al. used 1. 
        num_specialized_layers : int
            Number of specialized LSTM layers. Tax et al. used 1. 
        """
        super(SEP_Benchmarks_time, self).__init__()
        self.d_model = d_model
        self.num_activities = num_activities
        self.dropout = dropout
        self.num_shared_layers = num_shared_layers
        self.num_specialized_layers = num_specialized_layers

        emb_size_acts = min(600, round(1.6 * (num_activities-2)**0.56))
        self.act_embedding = nn.Embedding(num_embeddings=self.num_activities-1, embedding_dim=emb_size_acts, padding_idx=0)

        # Compute dimensionality of concatenated prefix event vectors 
        self.dim_init_prefix = emb_size_acts + 2 

        # Initialize dropout 
        self.dropout = nn.Dropout(self.dropout)

        # Initialize the shared LSTM layers 
        self.lstm_shared = nn.LSTM(input_size=self.dim_init_prefix, hidden_size=self.d_model, num_layers=self.num_shared_layers, batch_first=True, dropout=0.2)
        self.bn_shared = nn.BatchNorm1d(d_model)

        # Specialized activity layer(s)
        self.lstm_act = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.num_specialized_layers, batch_first=True, dropout=0.2)
        self.bn_act = nn.BatchNorm1d(d_model)

        # Specialized timestamp layer(s)
        self.lstm_ttne = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.num_specialized_layers, batch_first=True, dropout=0.2)
        self.bn_ttne = nn.BatchNorm1d(d_model)

        # Initialize LSTM parameters with Glorot_uniform 
        self.reset_parameters_lstm()

        # Dense activity prediction layer 
        self.fc_act = nn.Linear(self.d_model, self.num_activities)

        # Apply the Glorot (Xavier) uniform initialization 
        init.xavier_uniform_(self.fc_act.weight)

        # Dense ttne prediction layer 
        self.fc_ttne = nn.Linear(self.d_model, 1)

        # Apply the Glorot (Xavier) uniform initialization
        init.xavier_uniform_(self.fc_ttne.weight)


    def reset_parameters_lstm(self):
        # Manually initialize LSTM weights to mimic glorot_uniform

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
        for name, param in self.lstm_ttne.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
    
    def forward(self, inputs):
        # integer-encoded prefix activities 
        prefix_acts = inputs[0] # (batch_size, window_size)

        prefix_acts = self.act_embedding(prefix_acts) # (batch_size, window_size, emb_size_acts)
        
        time_features = inputs[1] # (batch_size, window_size, 2)

        # Concatenate prefix_acts with time features
        main_input = torch.cat((prefix_acts, time_features), dim=-1) # (batch_size, window_size, self.dim_init_prefix)

        # Applying dropout over input 
        main_input = self.dropout(main_input)
        
        # Feeding the concatenated input to the shared lstm layer(s)
        shared_out, _ = self.lstm_shared(main_input) # (batch_size, window_size, self.d_model)
        # Dropout over output (needed since dropout not automatically applied over 
        # outputs last lstm layer)
        shared_out = self.dropout(shared_out)

        # Applying BatchNormalization over the shared output 
        shared_out = shared_out.permute(0, 2, 1) # (batch_size, self.d_model, window_size)
        shared_out = self.bn_shared(shared_out)
        #   permuting again to original shape 
        shared_out = shared_out.permute(0, 2, 1) # (batch_size, window_size, self.d_model)
        
        # Feeding in the sequence of shared lstm output vectors to te specialized layer(s)

        #   Activity LSTM layer 
        act_outputs, _ = self.lstm_act(shared_out) # (batch_size, window_size, d_model)
    
        #   Selecting only the last output 
        act_outputs = act_outputs[:, -1, :] # (batch_size, d_model)
        # Applying dropout and batchnorm
        act_outputs = self.bn_act(self.dropout(act_outputs)) # (batch_size, d_model)

        # Feeding output vector to last dedicated dense layer
        act_probs = self.fc_act(act_outputs) # (batch_size, num_classes) = (batch_size, num_activities+2)

        #   TTNE LSTM layer 
        ttne_outputs, _ = self.lstm_ttne(shared_out) # (batch_size, window_size, d_model)
    
        #   Selecting only the last output 
        ttne_outputs = ttne_outputs[:, -1, :] # (batch_size, d_model)
        # Applying dropout and batchnorm
        ttne_outputs = self.bn_ttne(self.dropout(ttne_outputs)) # (batch_size, d_model)

        # Feeding output vector to last dedicated dense layer
        ttne_pred = self.fc_ttne(ttne_outputs) # (batch_size, 1)

        return act_probs, ttne_pred # (batch_size, num_classes) and (batch_size, 1)