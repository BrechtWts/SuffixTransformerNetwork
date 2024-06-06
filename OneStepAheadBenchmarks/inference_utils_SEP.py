"""
Loss function and its components SEP benchmarks
"""

import torch
import torch.nn as nn

##################################################################
##    Individual inference metrics One-Step Ahead prediction    ##
##################################################################

class ActCrossEntropyMetric(nn.Module):
    def __init__(self, num_classes):
        super(ActCrossEntropyMetric, self).__init__()
        # Number of activity output neurons. Includes padding token and end_token.
        self.num_classes = num_classes
        self.cross_entropy_crit = nn.CrossEntropyLoss(reduction='none')

        
    def forward(self, inputs, targets):
        """Compute the CrossEntropyLoss of the next activity prediction 
        head.

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the unnormalized logits for each 
            activity class. Shape (batch_size, num_classes) 
            and dtype torch.float32.
        targets : torch.Tensor
            The activity labels, containing the indices. Shape 
            (batch_size, window_size), dtype torch.int64. Given the 
            one-step ahead training paradigm of the SEP-LSTM, only 
            the very first target of each instance will be used. 

        Returns
        -------
        loss: torch.Tensor
            The masked cross entropy loss for the activity prediction head. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Slice out only the first activity label 
        targets = targets[:, 0] # (batch_size,)

        # Compute loss 
        loss = self.cross_entropy_crit(inputs, targets) # (batch_size,)

        return loss

class NextActivityAccuracy(nn.Module):
    def __init__(self, num_classes):
        super(NextActivityAccuracy, self).__init__()
        # Number of activity output neurons. Includes padding token and end_token.
        self.num_classes = num_classes

        
    def forward(self, inputs, targets):
        """Compute a boolean tensor of shape (batch_size,), containing 
        `True` if the predicted next activity for a certain instance is 
        the correct one. `False` otherwise. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the unnormalized logits for each 
            activity class. Shape (batch_size, num_classes) 
            and dtype torch.float32.
        targets : torch.Tensor
            The activity labels, containing the indices. Shape 
            (batch_size, window_size), dtype torch.int64. 
            Given the one-step ahead training paradigm of the SEP-LSTM, 
            only the very first target of each instance will be used. 

        Returns
        -------
        loss: torch.Tensor
            The masked cross entropy loss for the activity prediction head. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Slice out only the first activity label 
        targets = targets[:, 0] # (batch_size,)

        # Masking the padding token 
        inputs[:, 0] = -1e9 # (batch_size, num_classes)

        # Greedily selecting the predicted activity 
        act_inds = torch.argmax(inputs, dim=-1) # (batch_size,)

        # Accuracy bool 
        accuracy_bool = act_inds == targets # (batch_size,)

        return accuracy_bool
    
class MeanAbsoluteErrorMetric(nn.Module):
    def __init__(self):
        super(MeanAbsoluteErrorMetric, self).__init__()
        
    def forward(self, inputs, targets):
        """Computes the Mean Absolute Error (MAE) loss in which the 
        target values of -100.0, corresponding to padded event tokens, 
        are ignored / masked and hence do not contribute to the input 
        gradient. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the continuous predictions for the  
            Time Till Next Event (TTNE) target. Shape (batch_size, 1) and 
            dtype torch.float32.
        targets : torch.Tensor
            The continuous time prediction targets. Shape 
            (batch_size, window_size, 1), dtype torch.float32. Given the 
            one-step ahead training paradigm of the SEP-LSTM, only 
            the very first target of each instance will be used. 

        Returns
        -------
        loss: torch.Tensor
            The masked MAE loss for one of the time prediction heads. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """

        # Reshape 
        inputs = inputs[:, 0] # (batch_size,)

        # Slice out only the first TTNE label 
        targets = targets[:, 0, 0] # (batch_size,)

        absolute_errors = torch.abs(inputs-targets) # (batch_size,)

        # Compute masked loss 
        return absolute_errors # (batch_size,)
    
    

class MultiOutputMetric(nn.Module):
    def __init__(self, num_classes):
        """Composite loss function for the following two jointly 
        learned prediction tasks: 

        #. NEXT activity label prediction

        #. Time Till Next Event (TTNE) prediction
        
        Parameters
        ----------
        num_classes : int
            Number of output neurons (including padding and end tokens) 
            in the output layer of the activity suffix prediction task. 
        """
        super(MultiOutputMetric, self).__init__()
        self.activity_accuracy = NextActivityAccuracy(num_classes)
        self.cat_loss_fn = ActCrossEntropyMetric(num_classes)
        self.cont_loss_fn_ttne = MeanAbsoluteErrorMetric()

    # def forward(self, cat_output, ttne_output, rrt_output, cat_target, ttne_target, rrt_target):
    def forward(self, outputs, labels):
        """Compute composite loss (for gradient updates) and return its 
        components as python floats for tracking training progress.

        Parameters
        ----------
        outputs : tuple of torch.Tensor
            Tuple consisting of two tensors, each containing the 
            model's predictions for one of the two tasks. 
        labels : tuple of torch.Tensor
            Tuple consisting of two tensors, each containing the 
            labels for one of the two tasks.

        Returns
        -------
        loss : torch.Tensor
            Scalar tensor. Contains the composite loss that is used for 
            updating the gradients during training. Gradient tracking 
            turned on.
        accuracy_bool : torch.Tensor
            Boolean tensor of shape (batch_size,), indicating for each 
            of the instances whether the predicted next activity was 
            the right one (`True`) or not (`False`).
        cat_loss : torch.Tensor
            Tensor of shape (batch_size,) and dtype torch.float32, 
            containing the cross entropy loss contributions for each of 
            batch_size instances (prefixes) (next activity prediction 
            head).
        cont_loss : torch.Tensor
            Tensor of shape (batch_size,) and dtype torch.float32, 
            containing the MAE loss contributions for each of 
            batch_size instances (prefixes) for the Time Till Next Event 
            (TTNE) prediction target. These MAE contributions are simply 
            computed by taking the absolute difference between the 
            standardized TTNE prediction, and the standardized 
            ground-truth. In order to compute the MAE in minutes, 
            the result should still be multiplied by the standard 
            deviation computed on the training set.
        """
        # Accuracy next activity prediction 
        accuracy_bool = self.activity_accuracy(outputs[0].clone(), labels[-1].clone()) # (batch_size,)

        # Loss next activity prediction
        cat_loss = self.cat_loss_fn(outputs[0], labels[-1]) # (batch_size,)
        
        # Loss TTNE prdiction
        cont_loss = self.cont_loss_fn_ttne(outputs[1], labels[0]) # (batch_size,)

        # Composite loss (used for gradient updates)
        loss = cat_loss + cont_loss # (batch_size,)

        # Composite loss, act suffix loss, ttne loss, rrt loss
        return loss, cat_loss, cont_loss, accuracy_bool