"""
Custom loss functions for the CRTP-LSTM benchmark. In contrast to the 
loss function contained within `CRTP_LSTM\train_utils_lstm_masked.py`, 
in which the padded events are masked and hence do not contribute to the 
computation of losses, no masking is applied here. This is in line with 
the original implementation of the CRTP-LSTM benchmark. The authors 
however indicated that after further exploration, it became evident that 
masking lead to faster convergence. Therefore, in the SuTraN paper, 
masking is applied for the CRTP-LSTM benchmark as well. 
"""

import torch
import torch.nn as nn


#####################################
##    Individual loss functions    ##
#####################################

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(MaskedCrossEntropyLoss, self).__init__()
        # Number of activity output neurons. Includes padding token and end_token.
        self.num_classes = num_classes
        self.cross_entropy_crit = nn.CrossEntropyLoss()

        
    def forward(self, inputs, targets):
        """Compute the CrossEntropyLoss of the next activity prediction 
        head without masking the padding tokens, like it was done in the 
        CRTP-LSTM benchmark. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the unnormalized logits for each 
            activity class. Shape (batch_size, window_size, num_classes) 
            and dtype torch.float32.
        targets : torch.Tensor
            The activity labels, containing the indices. Shape 
            (batch_size, window_size), dtype torch.int64. 

        Returns
        -------
        loss: torch.Tensor
            The cross entropy loss for the activity prediction head. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Reshape inputs to shape (batch_size*window_size, num_classes)
        inputs = torch.reshape(input=inputs, shape=(-1, self.num_classes))
        # Reshape targets to shape (batch_size*window_size,)
        targets = torch.reshape(input=targets, shape=(-1,))

        # Compute masked loss 
        loss = self.cross_entropy_crit(inputs, targets) # scalar tensor

        return loss
    
class MaskedMeanAbsoluteErrorLoss(nn.Module):
    def __init__(self):
        super(MaskedMeanAbsoluteErrorLoss, self).__init__()
        
    def forward(self, inputs, targets):
        """Computes the Mean Absolute Error (MAE) of the CRTP-LSTM's 
        remaining runtime (rrt) prediction head without masking, 
        following the original implementation of the benchmark. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the continuous predictions for the  
            remaining runtime target. Shape (batch_size, window_size, 1) 
            and dtype torch.float32.
        targets : torch.Tensor
            The continuous time prediction targets. Shape 
            (batch_size, window_size, 1), dtype torch.float32. 

        Returns
        -------
        loss: torch.Tensor
            The masked MAE loss for one of the time prediction heads. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Reshape inputs to shape (batch_size*window_size,)
        inputs = torch.reshape(input=inputs, shape=(-1,))
        # Reshape targets to shape (batch_size*window_size,)
        targets= torch.reshape(input=targets, shape=(-1,))

        # Compute MAE rrt suffix predictions for all batch_size instances 
        absolute_errors = torch.abs(inputs-targets) # (batch_size * window_size,)


        # Return Mean Absolute Error 
        return torch.mean(absolute_errors)
    

class MultiOutputLoss(nn.Module):
    def __init__(self, num_classes):
        """Composite loss function for the following two jointly 
        learned prediction tasks: 

        #. activity suffix prediction 

        #. remaining runtime suffix predicion (default)
        
        Parameters
        ----------
        num_classes : int
            Number of output neurons (including padding and end tokens) 
            in the output layer of the activity suffix prediction task. 
        """
        super(MultiOutputLoss, self).__init__()
        self.cat_loss_fn = MaskedCrossEntropyLoss(num_classes)
        self.cont_loss_fn_ttne = MaskedMeanAbsoluteErrorLoss()

    def forward(self, outputs, labels):
        """Compute composite loss (for gradient updates) and return its 
        components as python floats for tracking training progress.

        Parameters
        ----------
        outputs : tuple of torch.Tensor
            Tuple consisting of two tensors, each containing the 
            model's predictions for one of the two tasks. 
        labels : tuple of torch.Tensor
            Tuple consisting of three tensors, each containing the 
            labels for one of the three tasks. However, since CRTP-LSTM 
            is trained on activity suffix and remaining runtime suffix 
            prediction only, only two out of three are needed. The last 
            tensor contains the activity suffix labels, the penultimate 
            tensor the remaining runtime suffix labels. 

        Returns
        -------
        loss : torch.Tensor
            Scalar tensor. Contains the composite loss that is used for 
            updating the gradients during training. Gradient tracking 
            turned on.
        cat_loss.item() : float
            Native python float. The cross entropy loss for 
            the next activity prediction head. Not used for gradient 
            updates during training, but for keeping track of the 
            different loss components during training and evaluation.
        cont_loss.item() : float
            Native python float. The MAE loss for the remaining  
            runtime suffix prediction head. Not (directly) used for 
            gradient updates during training, but for keeping track of 
            the different loss components during training and evaluation.
        """
        # Loss activity suffix prediction
        cat_loss = self.cat_loss_fn(outputs[0], labels[-1])
        
        # Remaining runtime (rrt) suffix prediction
        cont_loss = self.cont_loss_fn_ttne(outputs[1], labels[-2])

        # Composite loss (used for gradient updates)
        loss = cat_loss + cont_loss

        return loss, cat_loss.item(), cont_loss.item()
