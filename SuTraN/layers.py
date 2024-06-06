import torch
import torch.nn as nn
import torch.utils.data as data
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot product attention for the num_heads heads. 

        Parameters
        ----------
        Q : torch.Tensor
            Projected and split up queries, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        K : torch.Tensor
            Projected and split up keys, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        V : torch.Tensor
            Projected and split up values, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        mask : torch.Tensor, optional
            Padding mask, by default None. 
            If not None, shape (batch_size, window_size) and of the 
            bool dtype, with True on the positions that correspond to 
            padded / masked events. 

        Returns
        -------
        output : torch.Tensor
            The result of the MHA. Shape 
            (batch_size, self.num_heads, window_size, self.d_k)
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, num_heads, W, W)
        if mask is not None: # (batch_size, window_size)
            # mask is also a boolean tensor. Same shape as attn_scores or 
            # broadcastable to it. 
            batch_size = Q.shape[0]
            window_size = Q.shape[2]
            # broadcast mask to shape (batch_size, window_size, window_size)
            mask = torch.broadcast_to(mask.unsqueeze(1), size = (batch_size, window_size, window_size))
            # In the masked_fill, an additional dimension to the mask is added (B, 1, W, W)
            # And hence, automatically broadcasted to (B, num_heads, W, W)
            attn_scores = attn_scores.masked_fill(mask = mask.unsqueeze(1), value = -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # x : shape (batch_size, window_size, d_model)
        batch_size, seq_length, d_model = x.size()
        # x.view(...) further subdivides the innermost dim (of size d_model) into 
        # num_heads vectors of size d_k. 
        # x.view(...).transpose(1,2) transposes axis 1 and axis 2. 
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) 
        # shape (batch_size, self.num_heads, window_size, self.d_k)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        """MHA

        Parameters
        ----------
        Q : torch.Tensor 
            Queries. Tensor of shape (batch_size, window_size, d_model)
        K : torch.Tensor 
            Keys. Tensor of shape (batch_size, window_size, d_model)
        V : torch.Tensor 
            Values. Tensor of shape (batch_size, window_size, d_model)
        mask : torch.Tensor, optional
            Boolean mask of shape (batch_size, window_size). Entries are 
            True for the embeddings that correspond to padded events. 

        Returns
        -------
        _type_
            _description_
        """
        Q = self.split_heads(self.W_q(Q)) # (batch_size, self.num_heads, window_size, self.d_k)
        K = self.split_heads(self.W_k(K)) # (batch_size, self.num_heads, window_size, self.d_k)
        V = self.split_heads(self.W_v(V)) # (batch_size, self.num_heads, window_size, self.d_k)

        # So your mask needs to be of the shape (batch_size, self.num_heads, window_size, self_d_k)
        # and it is fed into the MHA with shape (batch_size, window_size)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Self attention for in decoder. Seperate class because of fixed look-ahead mask. 
class MultiHeadSelfAttentionDecoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttentionDecoder, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V):
        """Scaled dot product attention for the num_heads heads. 

        Parameters
        ----------
        Q : torch.Tensor
            Projected and split up queries, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        K : torch.Tensor
            Projected and split up keys, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        V : torch.Tensor
            Projected and split up values, shape 
            (batch_size, self.num_heads, window_size, self.d_k).

        Returns
        -------
        output : torch.Tensor
            The result of the MHA. Shape 
            (batch_size, self.num_heads, window_size, self.d_k)
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, num_heads, W, W)

        window_size = Q.shape[2]
        look_ahead = torch.triu(torch.ones(1, 1, window_size, window_size), diagonal=1).bool()
        look_ahead = look_ahead.to(device)

        attn_scores = attn_scores.masked_fill(mask = look_ahead, value = -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # x : shape (batch_size, window_size, d_model)
        batch_size, seq_length, d_model = x.size()
        # x.view(...) further subdivides the innermost dim (of size d_model) into 
        # num_heads vectors of size d_k. 
        # x.view(...).transpose(1,2) transposes axis 1 and axis 2. 
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) 
        # shape (batch_size, self.num_heads, window_size, self.d_k)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V):
        """MHA

        Parameters
        ----------
        Q : torch.Tensor 
            Queries. Tensor of shape (batch_size, window_size, d_model)
        K : torch.Tensor 
            Keys. Tensor of shape (batch_size, window_size, d_model)
        V : torch.Tensor 
            Values. Tensor of shape (batch_size, window_size, d_model)

        Returns
        -------
        _type_
            _description_
        """
        Q = self.split_heads(self.W_q(Q)) # (batch_size, self.num_heads, window_size, self.d_k)
        K = self.split_heads(self.W_k(K)) # (batch_size, self.num_heads, window_size, self.d_k)
        V = self.split_heads(self.W_v(V)) # (batch_size, self.num_heads, window_size, self.d_k)

        # So your mask needs to be of the shape (batch_size, self.num_heads, window_size, self_d_k)
        # and it is fed into the MHA with shape (batch_size, window_size)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x))) # (batch_size, window_size, d_model)