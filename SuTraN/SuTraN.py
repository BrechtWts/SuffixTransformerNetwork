import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from SuTraN.transformer_prefix_encoder import EncoderLayer
from SuTraN.transformer_suffix_decoder import DecoderLayer

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    """Inject sequence information in the prefix or suffix embeddings 
    before feeding them to the stack of encoders or decoders respectively. 

    Predominantly based on the PositionalEncoding module defined in 
    https://github.com/pytorch/examples/tree/master/word_language_model. 
    This reimplemetation, in contrast to the original one, caters for 
    adding sequence information in input embeddings where the batch 
    dimension comes first (``batch_first=True`). 

    Parameters
    ----------
    d_model : int
        The embedding dimension adopted by the associated Transformer. 
    dropout : float
        Dropout value. Dropout is applied over the sum of the input 
        embeddings and the positional encoding vectors. 
    max_len : int
        the max length of the incoming sequence. By default 10000. 
    """


    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()

        # Check if d_model is an integer and is even
        assert isinstance(d_model, int), "d_model must be an integer"
        assert d_model % 2 == 0, "d_model must be an even number"
        
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        self.register_buffer('pe', pe) # shape (max_len, d_model)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Sequence of prefix event tokens or suffix event tokens fed 
            to the positional encoding module. Shape 
            (batch_size, window_size, d_model).

        Returns
        -------
        Updated sequence tensor of the same shape, with sequence 
        information injected into it, and dropout applied. 
        """
        x = x + self.pe[:x.size(1), :] # (batch_size, window_size, d_model)
        return self.dropout(x)

class SuTraN(nn.Module):
    def __init__(self, 
                 num_activities, 
                 d_model, 
                 cardinality_categoricals_pref, 
                 num_numericals_pref, 
                 num_prefix_encoder_layers = 3, 
                 num_decoder_layers = 2,
                 num_heads=8, 
                 d_ff = 128, 
                 dropout = 0.2, 
                 remaining_runtime_head = True, 
                 layernorm_embeds = True, 
                 outcome_bool = False,
                 ):
        """Initialize an instance of SuTraN. The learned 
        activity embedding weight matrix is shared between the encoder 
        and decoder. 

        Parameters
        ----------
        num_activities : int
            Number of distinct activities present in the event log. 
            This does include the end token and padding token 
            used for the activity labels. For the categorical activity 
            label features in the prefix and suffix, no END token is 
            included. Hence, the amount of distinct levels there is 
            equal to `num_activities`-1. 
        d_model : int
            Model dimension. Each sublayer of the encoder and decoder 
            blocks take as input a (batch_size, window_size, d_model) 
            shaped tensor, and output an updated tensor of the same 
            shape. 
        cardinality_categoricals_pref : list of int
            List of `num_categoricals` integers. Each integer entry 
            i (i = 0, ..., `num_categoricals`-1) contains the cardinality 
            of the i'th categorical feature of the encoder prefix events. 
            The order of the cardinalities should match the order in 
            which the categoricals are fed as inputs. Note that for each 
            categorical, an extra category should be included to account 
            for missing values.
        num_numericals_pref : int 
            Number of numerical features of the prefix events
        num_prefix_encoder_layers : int, optional
            The number of prefix encoder blocks stacked on top of each 
            other, by default 3.
        num_decoder_layers : int, optional
            Number of decoder blocks stacked on top of each other, 
            by default 2.
        num_heads : int, optional
            Number of attention heads for the Multi-Head Attention 
            sublayers in both the encoder and decoder blocks, by default 
            8.
        d_ff : int, optional
            The dimension of the hidden layer of the point-wise feed 
            forward sublayers in the transformer blocks , by default 128.
        dropout : float, optional
            Dropout rate during training. By default 0.2. 
        remaining_runtime_head : bool, optional
            If True, on top of the default time till next event suffix 
            prediction and the activity suffix prediction, also the 
            complete remaining runtime is predicted. By default True. 
            See Notes for further remarks 
            regarding the `remaining_runtime_head` parameter. 
        layernorm_embeds : bool, optional
            Whether or not Layer Normalization is applied over the 
            initial embeddings of the encoder and decoder. True by 
            default.
        outcome_bool : bool, optional 
            Whether or not the model should also include a prediction 
            head for binary outcome prediction. By default `False`. If 
            `outcome_bool=True`, a prediction head for predicting 
            the binary outcome given a prefix is added. This prediction 
            head, in contrast to the time till next event and activity 
            suffix predictions, will only be trained to provide a 
            prediction at the first decoding step. Note that the 
            value of `outcome_bool` should be aligned with the 
            `outcome_bool` parameter of the training and inference 
            procedure, as well as with the preprocessing pipeline that 
            produces the labels. See Notes for further remarks 
            regarding the `outcome_bool` parameter. 

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
        """
        super(SuTraN, self).__init__()

        self.num_activities = num_activities

        self.d_model = d_model

        # Cardinality categoricals encoder prefix events
        self.cardinality_categoricals_pref = cardinality_categoricals_pref
        self.num_categoricals_pref = len(self.cardinality_categoricals_pref)

        # Number of numerical features encoder prefix events 
        self.num_numericals_pref = num_numericals_pref

        self.num_prefix_encoder_layers = num_prefix_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.remaining_runtime_head = remaining_runtime_head
        self.layernorm_embeds = layernorm_embeds
        self.outcome_bool = outcome_bool

        # Initialize positional encoding layer 
        self.positional_encoding = PositionalEncoding(d_model)

        # Initializing the categorical embeddings for the encoder inputs: 
        # Shared activity embeddings prefix and suffix! So only for the remaining ones you should do it. 
        self.embed_sz_categ_pref = [min(600, round(1.6 * n_cat**0.56)) for n_cat in self.cardinality_categoricals_pref[:-1]]
        self.activity_emb_size = min(600, round(1.6 * self.cardinality_categoricals_pref[-1]**0.56))

        # Initializing a separate embedding layer for each categorical prefix feature 
        # (Incrementing the cardinality with 1 to account for the padding idx of 0.)
        self.cat_embeds_pref = nn.ModuleList([nn.Embedding(num_embeddings=self.cardinality_categoricals_pref[i]+1, embedding_dim=self.embed_sz_categ_pref[i], padding_idx=0) for i in range(self.num_categoricals_pref-1)])
        self.act_emb = nn.Embedding(num_embeddings=num_activities-1, embedding_dim=self.activity_emb_size, padding_idx=0)


        # Dimensionality of initial encoder events after the prefix categoricals are fed to the dedicated entity embeddings and everything, including the numericals 
        # are concatenated
        self.dim_init_prefix = sum(self.embed_sz_categ_pref) + self.activity_emb_size + self.num_numericals_pref
        # Initial input embedding prefix events (encoder)
        self.input_embeddings_encoder = nn.Linear(self.dim_init_prefix, self.d_model)

        # Dimensionality of initial decoder suffix event tokens after the suffix categoricals are fed to the dedicated entity embeddings and everything, 
        # including the numericals are concatenated
        self.dim_init_suffix = self.activity_emb_size + 2

        # Initial input embedding prefix events (encoder)
        self.input_embeddings_decoder = nn.Linear(self.dim_init_suffix, self.d_model)

        # Initializing the num_prefix_encoder_layers encoder layers 
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(self.num_prefix_encoder_layers)])
        # Initializing the num_decoder_layers decoder layers 
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(self.num_decoder_layers)])

        # Initializing the additional activity output layer
        self.fc_out_act = nn.Linear(self.d_model, self.num_activities) # (batch_size, window_size, num_activities)

        # Initializing the additional time till next event prediction layer
        self.fc_out_ttne = nn.Linear(self.d_model, 1)

        if self.remaining_runtime_head:
            # Additional remaining runtime layers
            self.fc_out_rrt = nn.Linear(self.d_model, 1)

        if self.outcome_bool:
            # Additional (binary) outcome head 
            self.fc_out_out = nn.Linear(self.d_model, 1)
            # Sigmoid activiation function
            self.sigmoid_out = nn.Sigmoid()
        
        
        if self.layernorm_embeds:
            self.norm_enc_embeds = nn.LayerNorm(self.d_model)
            self.norm_dec_embeds = nn.LayerNorm(self.d_model)

            
        self.dropout = nn.Dropout(self.dropout)

        # Creating forward call bools to know what to output 
        self.only_rrt = (not self.outcome_bool) & self.remaining_runtime_head
        self.only_out = self.outcome_bool & (not self.remaining_runtime_head)
        self.both_not = (not self.outcome_bool) & (not self.remaining_runtime_head)
        self.both = self.outcome_bool & self.remaining_runtime_head
    


    # window_size : number of decoding steps during inference (model.eval())
    def forward(self, 
                inputs, 
                window_size=None, 
                mean_std_ttne=None, 
                mean_std_tsp=None, 
                mean_std_tss=None):
        """Processing a batch of inputs. The activity labels of the 
        prefix events are (and should) always be located at 
        inputs[self.num_categoricals_pref-1].

        Parameters
        ----------
        inputs : list of torch.Tensor
            List of tensors containing the various components 
            of the inputs. 
        window_size : None or int, optional
            The (shared) sequence length of the prefix and suffix inputs. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        mean_std_ttne : None or list of int, optional 
            List of two integers representing the mean and standard 
            deviation of the Time Till Next Event (TTNE) prediction 
            targets in seconds, computed over the training set instances 
            and used to standardize the TTNE labels of the training set, 
            validation set and test set. Needed for converting 
            timestamp predictions back to seconds and vice versa, during 
            inference only. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        mean_std_tsp : None or list of int, optional 
            List of two integers representing the mean and standard 
            deviation of the Time Since Previous (TSP) event features of 
            the suffix event tokens, in seconds computed over the 
            training set instances and used to standardize the TSP values 
            of the training set, validation set and test set. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        mean_std_tss : None or list of int, optional 
            List of two integers representing the mean and standard 
            deviation of the Time Since Start (TSS) event features of 
            the suffix event tokens, in seconds computed over the 
            training set instances and used to standardize the TSS values 
            of the training set, validation set and test set. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        """
        # Tensor containing the numerical features of the prefix events. 
        num_ftrs_pref = inputs[(self.num_categoricals_pref-1)+1] # (batch_size, window_size, N)

        # Tensor containing the padding mask for the prefix events. 
        padding_mask_input = inputs[(self.num_categoricals_pref-1)+2] # (batch_size, window_size) = (B, W)

        # Just auxilary index for understandability
        idx = self.num_categoricals_pref+2

        # Tensor containing the numerical features of the suffix event tokens: 
        num_ftrs_suf = inputs[idx + 1] # (batch_size, window_size, 2)

        # Constructing categorical embeddings prefix (encoder)
        cat_emb_pref = self.cat_embeds_pref[0](inputs[0]) # (batch_size, window_size, embed_sz_categ[0])
        for i in range(1, self.num_categoricals_pref-1):
            cat_emb_help = self.cat_embeds_pref[i](inputs[i]) # (batch_size, window_size, embed_sz_categ[i])
            cat_emb_pref = torch.cat((cat_emb_pref, cat_emb_help), dim = -1) # (batch_size, window_size, sum(embed_sz_categ[:i+1]))
        act_emb_pref = self.act_emb(inputs[self.num_categoricals_pref-1])
        cat_emb_pref = torch.cat((cat_emb_pref, act_emb_pref), dim=-1)
        
        # Concatenate cat_emb with the numerical features to get initial vector representations prefix events. 
        x = torch.cat((cat_emb_pref, num_ftrs_pref), dim = -1) # (batch_size, window_size, sum(embed_sz_categ)+N)

        # Dropout over concatenated features: 
        x = self.dropout(x)

        # Initial embedding encoder (prefix events)
        x = self.positional_encoding(self.input_embeddings_encoder(x) * math.sqrt(self.d_model)) # (batch_size, window_size, d_model)
        if self.layernorm_embeds:
            x = self.norm_enc_embeds(x) # (batch_size, window_size, d_model)

        # Updating the prefix event embeddings with the encoder blocks 
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, padding_mask_input)

        # ---------------------------

        if self.training: # Teacher forcing (for now)

            # Using the activity embedding layer shared with the encoder 
            cat_emb_suf = self.act_emb(inputs[idx]) # (batch_size, window_size, embed_sz_categ[0])
            
            # Concatenate cat_emb with the numerical features to get initial vector representations suffix event tokens.
            target_in = torch.cat((cat_emb_suf, num_ftrs_suf), dim = -1) # (batch_size, window_size, self.dim_init_suffix)
            
            # Initial embeddings decoder suffix event tokens 
            # The positional encoding module applies dropout over the result 
            target_in = self.positional_encoding(self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)) # (batch_size, window_size, d_model)

            if self.layernorm_embeds:
                target_in = self.norm_dec_embeds(target_in) # (batch_size, window_size, d_model)

            # Activating the decoder
            dec_output = target_in
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, x, padding_mask_input) # (batch_size, window_size)

            # Next activity prediction head: 
            act_probs = self.fc_out_act(dec_output) # (batch_size, window_size, self.num_activities)

            # Time till next event prediction (ttne) head:
            ttne_pred = self.fc_out_ttne(dec_output) # (batch_size, window_size, 1)

            # if self.remaining_runtime_head:
            if self.only_rrt:
                # Complete remaining runtime prediction (rrt) head
                rrt_pred = self.fc_out_rrt(dec_output) # (batch_size, window_size, 1)

                return act_probs, ttne_pred, rrt_pred 
                # (batch_size, window_size, self.num_activities), (batch_size, window_size, 1), (batch_size, window_size, 1)
            elif self.only_out:
                out_pred = self.fc_out_out(dec_output) # (batch_size, window_size, 1)
                out_pred = self.sigmoid_out(out_pred) # (batch_size, window_size, 1)
                # Only first decoding step output needed 
                out_pred = out_pred[:, 0, :] # (batch_size, 1)
                return act_probs, ttne_pred, out_pred
            elif self.both:
                rrt_pred = self.fc_out_rrt(dec_output) # (batch_size, window_size, 1)

                out_pred = self.fc_out_out(dec_output) # (batch_size, window_size, 1)
                out_pred = self.sigmoid_out(out_pred) # (batch_size, window_size, 1)
                # Only first decoding step output needed 
                out_pred = out_pred[:, 0, :] # (batch_size, 1)
                return act_probs, ttne_pred, rrt_pred, out_pred
            else: 
                return act_probs, ttne_pred
                # (batch_size, window_size, self.num_activities), (batch_size, window_size, 1)

        else: # Inference mode greedy decoding activities 
            # NOTE: Considerations for future work, in which you adopt 
            # similar procedure during training for rescheduled sampling: 
            # Pay attention to gradient tracking, whether you should detach 
            # the next decoder suffix event token's derived features based 
            # on predictions current decoding step. Figure out whether 
            # the operations involved still maintain differentiability wrt 
            # predictions used for deriving new features. 

            # Retrieving suffix activity integer vector `act_inputs`.
            #   `act_inputs` still contains the ground truth activity 
            #   labels (shifted by 1) for the entire suffixes. However, 
            #   at each decoding step `dec_step`, we will only predict 
            #   based on the shifed suffix generated up till that point, 
            #   and use those predictions to update the activity labels  
            #   for the subsequent decoding step. Finally, the look-ahead  
            #   mask ensures that the decoder cannot incorporate any 
            #   information regarding ground-truth activity labels in the 
            #   suffix.
            #   NOTE: the same holds for the two time features of the 
            #   suffix event tokens (`num_ftrs_suf`).

            act_inputs = inputs[idx] # (B, W)

            batch_size = act_inputs.size(0) # B

            # Initializing zero filled tensors for storing the activity 
            # and timestamp predictions during decoding 
            suffix_acts_decoded = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.int64).to(device) # (B, W)
            suffix_ttne_preds = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.float32).to(device) # (B, W)

            for dec_step in range(0, window_size):
                # Leveraging learned embedding 
                cat_emb_suf = self.act_emb(act_inputs) # (B, W, self.activity_emb_size)

                # Concatenating both
                target_in = torch.cat((cat_emb_suf, num_ftrs_suf), dim = -1) # (B, W, dim_init_suffix)

                # Initial embeddings decoder suffix event tokens 
                target_in = self.positional_encoding(self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)) # (B, W, d_model)

                # Applying layernorm if specified 
                if self.layernorm_embeds:
                    target_in = self.norm_dec_embeds(target_in) # (B, W, d_model)

                # Activating the decoder
                dec_output = target_in
                for dec_layer in self.decoder_layers:
                    dec_output = dec_layer(dec_output, x, padding_mask_input) # (batch_size, window_size)

                # Next activity prediction head: 
                act_logits = self.fc_out_act(dec_output) # (B, W, self.num_activities)

                # Time till next event prediction (ttne) head:
                ttne_pred = self.fc_out_ttne(dec_output) # (B, W, 1)

                #   Selecting predictions for current decoding step
                act_outputs = act_logits[:, dec_step, :] # (B, C)
                ttne_outputs = ttne_pred[:, dec_step, 0] # (B, )

                # Adding time pred as-is 
                suffix_ttne_preds[:, dec_step] = ttne_outputs # (B, W)


                # Remaining Runtime Predictions and optional outcome 
                # prediction only performed at the very first decoding 
                # step 
                if dec_step == 0:
                    if self.remaining_runtime_head:
                        rrt_pred = self.fc_out_rrt(dec_output) # (B, W, 1)
                        # Slicing out first decoding step prediction only
                        rrt_pred = rrt_pred[:, 0, 0] # (B,)

                    if self.outcome_bool:
                        out_pred = self.fc_out_out(dec_output) # (B, W, 1)
                        out_pred = self.sigmoid_out(out_pred) # (B, W, 1)
                        # Slicing out first decoding step prediction only
                        out_pred = out_pred[:, 0, 0] # (batch_size, )

                # Decoding activity preditions (greedily)
                #   "Masking padding token"
                act_outputs[:, 0] = -1e9

                #   Greedy selection 
                act_selected = torch.argmax(act_outputs, dim=-1) # (batch_size,), torch.int64

                #   Adding selected activity integers to suffix_acts_decoded
                suffix_acts_decoded[:, dec_step] = act_selected

                if dec_step < (window_size-1):

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
                    act_suf_updates = torch.clamp(act_suf_updates, max=self.num_activities-2) # (batch_size,) aka (B,)

                    # Updating `act_inputs` for suffix decoder for next decoding step 

                    act_inputs[:, dec_step+1] = act_suf_updates # (B, W)

                    # Deriving TSS and TSP time features for next decoding 
                    # step based on the TTNE predictions 

                    #   Converting predictions standardized TTNE 
                    #   back to original scale (seconds)
                    time_preds_seconds = ttne_outputs*mean_std_ttne[1] + mean_std_ttne[0] # (batch_size,)

                    #   Truncating at zero (no negatives allowed)
                    time_preds_seconds = torch.clamp(time_preds_seconds, min=0)

                    #   Converting standardized TSS feature current decoding 
                    #   step's suffix event token to original scale (seconds) 
                    tss_stand = num_ftrs_suf[:, dec_step, 0].clone() # (batch_size,)
                    tss_seconds = tss_stand*mean_std_tss[1] + mean_std_tss[0] # (batch_size,)

                    #   Clamping at zero again 
                    tss_seconds = torch.clamp(tss_seconds, min=0)

                    #   Updating tss in seconds next decoding step based on 
                    #   converted TTNE predictions 
                    tss_seconds_new = tss_seconds + time_preds_seconds # (batch_size,)

                    #   Converting back to preprocessed scale based on 
                    #   training mean and std
                    tss_stand_new = (tss_seconds_new - mean_std_tss[0]) / mean_std_tss[1] # (batch_size,)

                    #   TSP: time since previous event next decoding step 
                    #   is equal to the ttne in seconds, standardized with 
                    #   the training mean and std of the Suffix TSP feature 
                    tsp_stand_new = (time_preds_seconds - mean_std_tsp[0]) / mean_std_tsp[1] # (batch_size,)


                    #   Concatenating both 
                    new_suffix_timefeats = torch.cat((tss_stand_new.unsqueeze(-1), tsp_stand_new.unsqueeze(-1)), dim=-1) # (B, 2)
                    #   Updating next decoding step's time feature
                    #   tensor for the suffix event tokens 
                    num_ftrs_suf[:, dec_step+1, :] = new_suffix_timefeats # (B, W, 2)
            
            if self.only_rrt:
                return suffix_acts_decoded, suffix_ttne_preds, rrt_pred
                # (B, W), (B, W) and (B,)
            elif self.only_out:
                return suffix_acts_decoded, suffix_ttne_preds, out_pred
                # (B, W), (B, W) and (B, )
            elif self.both:
                return suffix_acts_decoded, suffix_ttne_preds, rrt_pred, out_pred
                # (B, W), (B, W), (B,) and (B, )
            else:
                return suffix_acts_decoded, suffix_ttne_preds
                # (B, W), (B, W)




# ---------------------------------------


class SuTraN_no_context(nn.Module):
    def __init__(self, 
                 num_activities, 
                 d_model, 
                 num_prefix_encoder_layers = 3, 
                 num_decoder_layers = 2,
                 num_heads=8, 
                 d_ff = 128, 
                 dropout = 0.2, 
                 remaining_runtime_head = True, 
                 layernorm_embeds = True, 
                 outcome_bool = False,
                 ):
        """Initialize an instance of SuTraN. The learned 
        activity embedding weight matrix is shared between the encoder 
        and decoder. 

        Parameters
        ----------
        num_activities : int
            Number of distinct activities present in the event log. 
            This does include the end token and padding token 
            used for the activity labels. For the categorical activity 
            label features in the prefix and suffix, no END token is 
            included. Hence, the amount of distinct levels there is 
            equal to `num_activities`-1. 
        d_model : int
            Model dimension. Each sublayer of the encoder and decoder 
            blocks take as input a (batch_size, window_size, d_model) 
            shaped tensor, and output an updated tensor of the same 
            shape. 
        num_prefix_encoder_layers : int, optional
            The number of prefix encoder blocks stacked on top of each 
            other, by default 3.
        num_decoder_layers : int, optional
            Number of decoder blocks stacked on top of each other, 
            by default 2.
        num_heads : int, optional
            Number of attention heads for the Multi-Head Attention 
            sublayers in both the encoder and decoder blocks, by default 
            8.
        d_ff : int, optional
            The dimension of the hidden layer of the point-wise feed 
            forward sublayers in the transformer blocks , by default 128.
        dropout : float, optional
            Dropout rate during training. By default 0.2. 
        remaining_runtime_head : bool, optional
            If True, on top of the default time till next event suffix 
            prediction and the activity suffix prediction, also the 
            complete remaining runtime is predicted. By default True. 
            See Notes for further remarks 
            regarding the `remaining_runtime_head` parameter. 
        layernorm_embeds : bool, optional
            Whether or not Layer Normalization is applied over the 
            initial embeddings of the encoder and decoder. True by 
            default.
        outcome_bool : bool, optional 
            Whether or not the model should also include a prediction 
            head for binary outcome prediction. By default `False`. If 
            `outcome_bool=True`, a prediction head for predicting 
            the binary outcome given a prefix is added. This prediction 
            head, in contrast to the time till next event and activity 
            suffix predictions, will only be trained to provide a 
            prediction at the first decoding step. Note that the 
            value of `outcome_bool` should be aligned with the 
            `outcome_bool` parameter of the training and inference 
            procedure, as well as with the preprocessing pipeline that 
            produces the labels. See Notes for further remarks 
            regarding the `outcome_bool` parameter. 

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
        """
        super(SuTraN_no_context, self).__init__()

        self.num_activities = num_activities

        self.d_model = d_model
        self.num_prefix_encoder_layers = num_prefix_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.remaining_runtime_head = remaining_runtime_head
        self.layernorm_embeds = layernorm_embeds
        self.outcome_bool = outcome_bool

        # Initialize positional encoding layer 
        self.positional_encoding = PositionalEncoding(d_model)

        # Initializing the shared categorical activity embedding for the 
        # encoder and decoder input sequences (seq of prefix event tokens 
        # and seq of suffix event tokens): 
        self.activity_emb_size = min(600, round(1.6 * (self.num_activities-2)**0.56))
        self.act_emb = nn.Embedding(num_embeddings=num_activities-1, embedding_dim=self.activity_emb_size, padding_idx=0)


        # Dimensionality of initial prefix event tokens after prefix 
        # categoricals are fed to the dedicated entity embeddings and 
        # everything, including the numericals is concatenated
        self.dim_init_prefix = self.activity_emb_size + 2

        # Initial input embedding prefix events (encoder)
        self.input_embeddings_encoder = nn.Linear(self.dim_init_prefix, self.d_model)

        # Dimensionality of initial decoder suffix event tokens after the suffix categoricals are fed to the dedicated entity embeddings and everything, 
        # including the numericals are concatenated
        self.dim_init_suffix = self.activity_emb_size + 2

        # Initial input embedding prefix events (encoder)
        self.input_embeddings_decoder = nn.Linear(self.dim_init_suffix, self.d_model)

        # Initializing the num_prefix_encoder_layers encoder layers 
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(self.num_prefix_encoder_layers)])
        # Initializing the num_decoder_layers decoder layers 
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(self.num_decoder_layers)])

        # Initializing the additional activity output layer
        self.fc_out_act = nn.Linear(self.d_model, self.num_activities) # (batch_size, window_size, num_activities)

        # Initializing the additional time till next event prediction layer
        self.fc_out_ttne = nn.Linear(self.d_model, 1)

        if self.remaining_runtime_head:
            # Additional remaining runtime layers
            self.fc_out_rrt = nn.Linear(self.d_model, 1)

        if self.outcome_bool:
            # Additional (binary) outcome head 
            self.fc_out_out = nn.Linear(self.d_model, 1)
            # Sigmoid activiation function
            self.sigmoid_out = nn.Sigmoid()
        
        
        if self.layernorm_embeds:
            self.norm_enc_embeds = nn.LayerNorm(self.d_model)
            self.norm_dec_embeds = nn.LayerNorm(self.d_model)

            
        self.dropout = nn.Dropout(self.dropout)

        # Creating forward call bools to know what to output 
        self.only_rrt = (not self.outcome_bool) & self.remaining_runtime_head
        self.only_out = self.outcome_bool & (not self.remaining_runtime_head)
        self.both_not = (not self.outcome_bool) & (not self.remaining_runtime_head)
        self.both = self.outcome_bool & self.remaining_runtime_head


    # window_size : number of decoding steps during inference (model.eval())
    def forward(self, 
                inputs, 
                window_size=None, 
                mean_std_ttne=None, 
                mean_std_tsp=None, 
                mean_std_tss=None):
        """Processing a batch of inputs. The activity labels of the 
        prefix events are (and should) always be located at 
        inputs[self.num_categoricals_pref-1].

        Parameters
        ----------
        inputs : list of torch.Tensor
            List of tensors containing the various components 
            of the inputs. 
        window_size : None or int, optional
            The (shared) sequence length of the prefix and suffix inputs. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        mean_std_ttne : None or list of int, optional 
            List of two integers representing the mean and standard 
            deviation of the Time Till Next Event (TTNE) prediction 
            targets in seconds, computed over the training set instances 
            and used to standardize the TTNE labels of the training set, 
            validation set and test set. Needed for converting 
            timestamp predictions back to seconds and vice versa, during 
            inference only. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        mean_std_tsp : None or list of int, optional 
            List of two integers representing the mean and standard 
            deviation of the Time Since Previous (TSP) event features of 
            the suffix event tokens, in seconds computed over the 
            training set instances and used to standardize the TSP values 
            of the training set, validation set and test set. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        mean_std_tss : None or list of int, optional 
            List of two integers representing the mean and standard 
            deviation of the Time Since Start (TSS) event features of 
            the suffix event tokens, in seconds computed over the 
            training set instances and used to standardize the TSS values 
            of the training set, validation set and test set. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        """
        # Tensor containing the numerical features of the prefix events. 
        num_ftrs_pref = inputs[1] # (batch_size, window_size, N)

        # Tensor containing the padding mask for the prefix events. 
        padding_mask_input = inputs[2] # (batch_size, window_size) = (B, W)

        # Tensor containing the numerical features of the suffix event tokens: 
        num_ftrs_suf = inputs[4] # (batch_size, window_size, 2)

        # # Constructing categorical embeddings prefix (encoder)
        act_emb_pref = self.act_emb(inputs[0])
        # cat_emb_pref = torch.cat((cat_emb_pref, act_emb_pref), dim=-1)
        
        # Concatenate cat_emb with the numerical features to get initial vector representations prefix events. 
        x = torch.cat((act_emb_pref, num_ftrs_pref), dim = -1) # (batch_size, window_size, sum(embed_sz_categ)+N)

        # Dropout over concatenated features: 
        x = self.dropout(x)

        # Initial embedding encoder (prefix events)
        x = self.positional_encoding(self.input_embeddings_encoder(x) * math.sqrt(self.d_model)) # (batch_size, window_size, d_model)
        if self.layernorm_embeds:
            x = self.norm_enc_embeds(x) # (batch_size, window_size, d_model)

        # Updating the prefix event embeddings with the encoder blocks 
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, padding_mask_input)

        # ---------------------------

        if self.training: # Teacher forcing (for now)


            # Using the activity embedding layer shared with the encoder 
            cat_emb_suf = self.act_emb(inputs[3]) # (batch_size, window_size, embed_sz_categ[0])
            
            # Concatenate cat_emb with the numerical features to get initial vector representations suffix event tokens.
            target_in = torch.cat((cat_emb_suf, num_ftrs_suf), dim = -1) # (batch_size, window_size, self.dim_init_suffix)
            
            # Initial embeddings decoder suffix event tokens 
            target_in = self.positional_encoding(self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)) # (batch_size, window_size, d_model)

            if self.layernorm_embeds:
                target_in = self.norm_dec_embeds(target_in) # (batch_size, window_size, d_model)

            # Activating the decoder
            dec_output = target_in
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, x, padding_mask_input) # (batch_size, window_size)

            # Next activity prediction head: 
            act_probs = self.fc_out_act(dec_output) # (batch_size, window_size, self.num_activities)

            # Time till next event prediction (ttne) head:
            ttne_pred = self.fc_out_ttne(dec_output) # (batch_size, window_size, 1)

            # if self.remaining_runtime_head:
            if self.only_rrt:
                # Complete remaining runtime prediction (rrt) head
                rrt_pred = self.fc_out_rrt(dec_output) # (batch_size, window_size, 1)

                return act_probs, ttne_pred, rrt_pred 
                # (batch_size, window_size, self.num_activities), (batch_size, window_size, 1), (batch_size, window_size, 1)
            elif self.only_out:
                out_pred = self.fc_out_out(dec_output) # (batch_size, window_size, 1)
                out_pred = self.sigmoid_out(out_pred) # (batch_size, window_size, 1)
                # Only first decoding step output needed 
                out_pred = out_pred[:, 0, :] # (batch_size, 1)
                return act_probs, ttne_pred, out_pred
            elif self.both:
                rrt_pred = self.fc_out_rrt(dec_output) # (batch_size, window_size, 1)

                out_pred = self.fc_out_out(dec_output) # (batch_size, window_size, 1)
                out_pred = self.sigmoid_out(out_pred) # (batch_size, window_size, 1)
                # Only first decoding step output needed 
                out_pred = out_pred[:, 0, :] # (batch_size, 1)
                return act_probs, ttne_pred, rrt_pred, out_pred
            else: 
                return act_probs, ttne_pred
                # (batch_size, window_size, self.num_activities), (batch_size, window_size, 1)

        else: # Inference mode greedy decoding activities 

            # Retrieving suffix activity integer vector `act_inputs`.
            #   `act_inputs` still contains the ground truth activity 
            #   labels (shifted by 1) for the entire suffixes. However, 
            #   at each decoding step `dec_step`, we will only predict 
            #   based on the shifed suffix generated up till that point, 
            #   and use those predictions to update the activity labels  
            #   for the subsequent decoding step. Finally, the look-ahead  
            #   mask ensures that the decoder cannot incorporate any 
            #   information regarding ground-truth activity labels in the 
            #   suffix.
            #   NOTE: the same holds for the two time features of the 
            #   suffix event tokens (`num_ftrs_suf`).

            act_inputs = inputs[3] # (B, W)

            batch_size = act_inputs.size(0) # B

            # Initializing zero filled tensors for storing the activity 
            # and timestamp predictions during decoding 
            suffix_acts_decoded = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.int64).to(device) # (B, W)
            suffix_ttne_preds = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.float32).to(device) # (B, W)

            for dec_step in range(0, window_size):
                # Leveraging learned embedding 
                cat_emb_suf = self.act_emb(act_inputs) # (B, W, self.activity_emb_size)

                # Concatenating both
                target_in = torch.cat((cat_emb_suf, num_ftrs_suf), dim = -1) # (B, W, dim_init_suffix)

                # Initial embeddings decoder suffix event tokens 
                target_in = self.positional_encoding(self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)) # (B, W, d_model)

                # Applying layernorm if specified 
                if self.layernorm_embeds:
                    target_in = self.norm_dec_embeds(target_in) # (B, W, d_model)

                # Activating the decoder
                dec_output = target_in
                for dec_layer in self.decoder_layers:
                    dec_output = dec_layer(dec_output, x, padding_mask_input) # (batch_size, window_size)

                # Next activity prediction head: 
                act_logits = self.fc_out_act(dec_output) # (B, W, self.num_activities)

                # Time till next event prediction (ttne) head:
                ttne_pred = self.fc_out_ttne(dec_output) # (B, W, 1)

                #   Selecting predictions for current decoding step
                act_outputs = act_logits[:, dec_step, :] # (B, C)
                ttne_outputs = ttne_pred[:, dec_step, 0] # (B, )

                # Adding time pred as-is 
                suffix_ttne_preds[:, dec_step] = ttne_outputs # (B, W)


                # Remaining Runtime Predictions and optional outcome 
                # prediction only performed at the very first decoding 
                # step 
                if dec_step == 0:
                    if self.remaining_runtime_head:
                        rrt_pred = self.fc_out_rrt(dec_output) # (B, W, 1)
                        # Slicing out first decoding step prediction only
                        rrt_pred = rrt_pred[:, 0, 0] # (B,)

                    if self.outcome_bool:
                        out_pred = self.fc_out_out(dec_output) # (B, W, 1)
                        out_pred = self.sigmoid_out(out_pred) # (B, W, 1)
                        # Slicing out first decoding step prediction only
                        out_pred = out_pred[:, 0, 0] # (batch_size, )

                # Decoding activity preditions (greedily)
                #   "Masking padding token"
                act_outputs[:, 0] = -1e9

                #   Greedy selection 
                act_selected = torch.argmax(act_outputs, dim=-1) # (batch_size,), torch.int64

                #   Adding selected activity integers to suffix_acts_decoded
                suffix_acts_decoded[:, dec_step] = act_selected

                if dec_step < (window_size-1):

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
                    act_suf_updates = torch.clamp(act_suf_updates, max=self.num_activities-2) # (batch_size,) aka (B,)

                    # Updating `act_inputs` for suffix decoder for next decoding step 

                    act_inputs[:, dec_step+1] = act_suf_updates # (B, W)

                    # Deriving TSS and TSP time features for next decoding 
                    # step based on the TTNE predictions 

                    #   Converting predictions standardized TTNE 
                    #   back to original scale (seconds)
                    time_preds_seconds = ttne_outputs*mean_std_ttne[1] + mean_std_ttne[0] # (batch_size,)

                    #   Truncating at zero (no negatives allowed)
                    time_preds_seconds = torch.clamp(time_preds_seconds, min=0)

                    #   Converting standardized TSS feature current decoding 
                    #   step's suffix event token to original scale (seconds) 
                    tss_stand = num_ftrs_suf[:, dec_step, 0].clone() # (batch_size,)
                    tss_seconds = tss_stand*mean_std_tss[1] + mean_std_tss[0] # (batch_size,)

                    #   Clamping at zero again 
                    tss_seconds = torch.clamp(tss_seconds, min=0)

                    #   Updating tss in seconds next decoding step based on 
                    #   converted TTNE predictions 
                    tss_seconds_new = tss_seconds + time_preds_seconds # (batch_size,)

                    #   Converting back to preprocessed scale based on 
                    #   training mean and std
                    tss_stand_new = (tss_seconds_new - mean_std_tss[0]) / mean_std_tss[1] # (batch_size,)

                    #   TSP: time since previous event next decoding step 
                    #   is equal to the ttne in seconds, standardized with 
                    #   the training mean and std of the Suffix TSP feature 
                    tsp_stand_new = (time_preds_seconds - mean_std_tsp[0]) / mean_std_tsp[1] # (batch_size,)


                    #   Concatenating both 
                    new_suffix_timefeats = torch.cat((tss_stand_new.unsqueeze(-1), tsp_stand_new.unsqueeze(-1)), dim=-1) # (B, 2)
                    #   Updating next decoding step's time feature
                    #   tensor for the suffix event tokens 
                    num_ftrs_suf[:, dec_step+1, :] = new_suffix_timefeats # (B, W, 2)
            
            if self.only_rrt:
                # NOTE: rrt_pred already shape (B,). Don't have to subset anymore 
                # for metric computations 
                return suffix_acts_decoded, suffix_ttne_preds, rrt_pred
                # (B, W), (B, W) and (B,)
            elif self.only_out:
                return suffix_acts_decoded, suffix_ttne_preds, out_pred
                # (B, W), (B, W) and (B, )
            elif self.both:
                return suffix_acts_decoded, suffix_ttne_preds, rrt_pred, out_pred
                # (B, W), (B, W), (B,) and (B, )
            else:
                return suffix_acts_decoded, suffix_ttne_preds
                # (B, W), (B, W)