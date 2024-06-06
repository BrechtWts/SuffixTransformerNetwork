"""Contains the training loss function components for the CRTP-LSTM 
benchmark without masking, tailored for inference. 

Contains modified versions of the CRTP-LSTM training loss functions 
without masking specified in `CRTP_LSTM\train_utils_lstm.py`. The 
difference being that the metrics here are not averaged to a scalar 
for each batch, but every single instance's loss contributions are 
returned. This is needed to compute valid global averages over the 
whole validation or test set, in which the last batch's size can 
differ. 
"""

import torch
import torch.nn as nn


#####################################
##    Individual loss functions    ##
#####################################

class CrossEntropyMetric(nn.Module):
    def __init__(self, num_classes):
        super(CrossEntropyMetric, self).__init__()
        # Number of activity output neurons. Includes padding token and end_token.
        self.num_classes = num_classes
        # self.cross_entropy_crit = nn.CrossEntropyLoss(ignore_index = 0)
        self.cross_entropy_crit = nn.CrossEntropyLoss(reduction='none')

        
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
            The masked cross entropy loss components. Shape 
            (batch_size*window_size, )
        """
        # Reshape inputs to shape (batch_size*window_size, num_classes)
        inputs = torch.reshape(input=inputs, shape=(-1, self.num_classes))
        # Reshape targets to shape (batch_size*window_size,)
        targets = torch.reshape(input=targets, shape=(-1,)) # (batch_size*window_size,)

        # Compute masked loss 
        loss = self.cross_entropy_crit(inputs, targets) # (batch_size*window_size,)

        return loss
    
class MeanAbsoluteErrorMetric(nn.Module):
    def __init__(self):
        super(MeanAbsoluteErrorMetric, self).__init__()
        
    def forward(self, inputs, targets):
        """Computes the Mean Absolute Error (MAE) loss without any 
        masking, like it was done in the CRTP_LSTM benchmark. 

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
            The MAE loss components. Shape (batch_size*window_size,).
        """
        # Reshape inputs to shape (batch_size*window_size,)
        inputs = torch.reshape(input=inputs, shape=(-1,))
        # Reshape targets to shape (batch_size*window_size,)
        targets= torch.reshape(input=targets, shape=(-1,))

        absolute_errors = torch.abs(inputs-targets) # (batch_size * window_size,)

        return absolute_errors # Shape (batch_size*window_size,)
    

class MultiOutputMetric(nn.Module):
    def __init__(self, num_classes):
        """Composite loss function for the following two jointly 
        learned prediction tasks (CRTP-LSTM): 

        #. activity suffix prediction 

        #. remaining runtime suffix prediction
        
        Parameters
        ----------
        num_classes : int
            Number of output neurons (including padding and end tokens) 
            in the output layer of the activity suffix prediction task. 
        """
        super(MultiOutputMetric, self).__init__()
        self.cat_loss_fn = CrossEntropyMetric(num_classes)
        self.cont_loss_fn_ttne = MeanAbsoluteErrorMetric()

    def forward(self, outputs, labels):
        """Compute composite loss components for validation inference. 
        Needed for the learning rate scheduler used in the CRTP_LSTM 
        benchmark. 

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
            Tensor of shape (batch_size, window_size), returning the 
            composite loss components pertaining to each suffix event 
            for all batch_size test or validation set instances. 
        """
        # Loss activity suffix prediction
        cat_loss = self.cat_loss_fn(outputs[0], labels[-1])
        
        # Remaining runtime (rrt) suffix prediction
        cont_loss = self.cont_loss_fn_ttne(outputs[1], labels[-2])

        # Composite loss (used for gradient updates)
        loss = cat_loss + cont_loss

        return loss #(batch_size*window_size, )