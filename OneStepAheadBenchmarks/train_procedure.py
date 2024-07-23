"""Functionality for one-step-ahead training loop SEP-LSTM benchmark. 
SEP-LSTM, in contrast to all other implementations, is only trained for 
one-step ahead prediction and not for suffix generation. The metrics for n
ext activity and next timestamp prediction are stored here."""

import torch
import torch.nn as nn
from OneStepAheadBenchmarks.train_utils_SEP import MultiOutputLoss
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.data import DataLoader
from OneStepAheadBenchmarks.inference_procedure import inference_loop

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, 
                training_loader, 
                optimizer,
                loss_fn, 
                batch_interval,
                epoch_number, 
                max_norm):

    # Tracking global loss over all prediction heads:
    running_loss_glb = []
    # Tracking loss of each prediction head separately: 
    running_loss_act = [] # Cross-Entropy
    running_loss_ttne = [] # MAE

    # Tracking gradient norms
    original_norm_glb = []
    clipped_norm_glb = []


    for batch_num, data in tqdm(enumerate(training_loader), desc="Batch calculation at epoch {}.".format(epoch_number)):
        # inputs, labels = data # Adjust this 
        inputs = data[:-3]
        labels = data[-3:]
        # Sending inputs and labels to GPU
        inputs = [input_tensor.to(device) for input_tensor in inputs]

        # Selecting only the TTNE and ACTIVITY LABELS 
        labels = [labels[0].to(device), labels[-1].to(device)]

        # Restoring gradients to 0 for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Computing composite loss and individual losses for track keeping
        loss_results = loss_fn(outputs, labels)
        
        # Compute the loss and its gradients
        loss = loss_results[0]
        loss.backward()

        # Keep track of original gradient norm 
        original_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        original_norm_glb.append(original_norm.item())

        # Clip gradient norm
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        clipped_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        clipped_norm_glb.append(clipped_norm.item())

        # Adjust learning weights
        optimizer.step()

        # Tracking losses and metrics
        running_loss_glb.append(loss.item())
        running_loss_act.append(loss_results[1])
        running_loss_ttne.append(loss_results[-1])
        if batch_num % batch_interval == (batch_interval-1):
                print("------------------------------------------------------------")
                print("Epoch {}, batch {}:".format(epoch_number, batch_num))
                print("Average original gradient norm: {} (over last {} batches)".format(sum(original_norm_glb[-batch_interval:])/batch_interval, batch_interval))
                print("Average clipped gradient norm: {} (over last {} batches)".format(sum(clipped_norm_glb[-batch_interval:])/batch_interval, batch_interval))
                print("Running average global loss: {} (over last {} batches)".format(sum(running_loss_glb[-batch_interval:])/batch_interval, batch_interval))
                print("Running average activity prediction loss: {} (Cross Entropy over last {} batches)".format(sum(running_loss_act[-batch_interval:])/batch_interval, batch_interval))
                print("Running average time till next event prediction loss: {} (MAE over last {} batches)".format(sum(running_loss_ttne[-batch_interval:])/batch_interval, batch_interval))
                print("------------------------------------------------------------")

    print("=======================================")
    print("End of epoch {}".format(epoch_number))
    print("=======================================")
    last_running_avg_glob = sum(running_loss_glb[-batch_interval:])/batch_interval
    print("Running average global loss: {} (over last {} batches)".format(last_running_avg_glob, batch_interval))
    last_running_avg_act = sum(running_loss_act[-batch_interval:])/batch_interval
    print("Running average activity prediction loss: {} (Cross Entropy over last {} batches)".format(last_running_avg_act, batch_interval))
    last_running_avg_ttne = sum(running_loss_ttne[-batch_interval:])/batch_interval
    print("Running average time till next event prediction loss: {} (MAE over last {} batches)".format(last_running_avg_ttne, batch_interval))
    last_running_avgs = last_running_avg_glob, last_running_avg_act, last_running_avg_ttne, loss

    return model, optimizer, last_running_avgs
            

def train_model(model, 
                optimizer, 
                train_dataset, 
                val_dataset, 
                start_epoch, 
                num_epochs, 
                num_classes, 
                batch_interval, 
                path_name, 
                mean_std_ttne, 
                batch_size, 
                lr_scheduler_present=False, 
                lr_scheduler=None, 
                lstm_scheduler = True,
                best_MAE_ttne = 1e9, 
                best_act_acc = 0,
                max_norm = 2.):
    """_summary_

    Parameters
    ----------
    model : SEP_Benchmarks_time
        Initialized instance of the SEP-LSTM benchmark implementation. 
    optimizer : torch optimizer
        NAdam torch optimizer. Already initialized when feeding it to 
        ``train_model()``.
    train_dataset : torch.utils.data.dataset.TensorDataset
        TensorDataset containing the training data. 
    val_dataset : torch.utils.data.dataset.TensorDataset
        TensorDataset containing the validation data. 
    start_epoch : int
        Number of the epoch from which the training loop is started. 
        First call to ``train_model()`` should be done with 
        ``start_epoch=0``.
    num_epochs : int
        Number of epochs to train. When resuming training with another 
        loop of num_epochs, for the new ``train_model()``, the new 
        ``start_epoch`` argument should be equal to the current one 
        plus the current value for ``num_epochs``.
    num_classes : int
        The number of output neurons for the activity prediction head. 
        This includes the padding token (0) and the END token. 
    batch_interval : int
        The periodic amount of batches trained for which the moving average 
        losses and metrics are printed and recorded. E.g. if 
        ``batch_interval=100``, then after every 100 batches, the 
        moving averages of all metrics and losses during training are 
        recorded, printed and reset to 0. 
    path_name : str 
        Needed for saving results and callbacks in the 
        appropriate subfolders. This is the path name 
        of the subfolder for which all the results 
        and callbacks (model copies) should be 
        stored for the current event log and 
        model configuration.
    mean_std_ttne : list of float
        Training mean and standard deviation used to standardize the time 
        till next event (in seconds) target. Needed for re-converting 
        ttne predictions to original scale. Mean is the first entry, 
        std the second.
    batch_size : int 
        Batch size used during training. 
    lr_scheduler_present : bool, optional
        Indicates whether we work with a learning rate scheduler wrapped 
        around the optimizer. If True, learning rate scheduler 
        included. If False (default), not. 
    lr_scheduler : torch lr_scheduler or None
        If ``lr_scheduler_present=True``, a lr_scheduler that is wrapped 
        around the optimizer should be provided as well. The SEP-LSTM 
        benchmark should be ran with the 
        `torch.optim.lr_scheduler.ReduceLROnPlateau` learning rate 
        scheduler to adhere to the parameter settings of the original 
        implementation by Tax et al. 
    best_MAE_ttne : float 
        Best validation Mean Absolute Error for the time till 
        next event prediction. The defaults apply if the training 
        loop is initialized for the first time for a given configuration. 
        If the training loop is resumed from a certain checkpoint, the 
        best results of the previous training loop should be given. 
    best_act_acc : float, optional 
        Best validation next event prediction accuracy so far. 0 by 
        default. 
    max_norm : float, optional
        Max gradient norm used for clipping during training. By default 2.
    """
    if lr_scheduler_present:
        if lr_scheduler==None:
            print("No lr_scheduler provided.")
            return -1, -1, -1, -1

    # Checking whether GPU is used
    print("Device: {}".format(device))

    # Assigning SEP-LSTM
    model.to(device)

    # Creating train and validation dataloaders 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4096, shuffle=False, drop_last=False, pin_memory=True)
    # Tracking running averages over last ``batch_interval`` batches of each epoch
    # & tracking average validation losses
    train_losses_global = []
    train_losses_act = []
    train_losses_ttne = []


    avg_MAE_standardized_glob, avg_MAE_minutes_glob, avg_CE_inference_glob, avg_accuracy_glob = [], [], [], []

    val_loss_glob = []
    val_loss_best = 1e9
    
    loss_fn = MultiOutputLoss(num_classes)

    # Early stopping with patience 42 epochs for validation loss 
    # (in line with best parameter settings Tax et al)
    num_epochs_not_improved = 0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Setting seed for shuffling of training dataloader's shuffling 
        # of the instances, such that each epoch is shuffled differently, 
        # while still maintaining reproducability. 
        torch.manual_seed(epoch) 
        print(" ")
        print("------------------------------------")
        print('EPOCH {}:'.format(epoch))
        print("____________________________________")

        # Activate gradient tracking
        model.train(True)
        model, optimizer, last_running_avgs = train_epoch(model, 
                                                          train_dataloader, 
                                                          optimizer, 
                                                          loss_fn, 
                                                          batch_interval, 
                                                          epoch, 
                                                          max_norm)

        train_losses_global.append(last_running_avgs[0])
        train_losses_act.append(last_running_avgs[1])
        train_losses_ttne.append(last_running_avgs[2])
        last_loss = last_running_avgs[-1]

        # Set the model to evaluation mode and disabling dropout
        model.eval()

        # Initialize inference loop (one-step ahead prediction only)
        inf_results = inference_loop(model=model, 
                                     inference_dataloader=val_dataloader, 
                                     mean_std_ttne=mean_std_ttne, 
                                     num_classes=num_classes)
        
        # Inference metrics ONE-STEP ahead prediction
        val_loss_avg, CE_avg, avg_MAE_stand, avg_MAE_min, act_accuracy = inf_results

        val_loss_glob.append(val_loss_avg)
        avg_CE_inference_glob.append(CE_avg)
        avg_MAE_standardized_glob.append(avg_MAE_stand)
        avg_MAE_minutes_glob.append(avg_MAE_min)
        avg_accuracy_glob.append(act_accuracy)
        
        # Saving a epoch callback of the model's weights if certain 
        # conditions are triggered. 

        if act_accuracy > best_act_acc:  
            best_act_acc = act_accuracy
        
        if avg_MAE_stand < best_MAE_ttne: 
            best_MAE_ttne = avg_MAE_stand
            
        if val_loss_avg < val_loss_best:
            num_epochs_not_improved = 0
            val_loss_best = val_loss_avg

        elif val_loss_avg > val_loss_best:
            num_epochs_not_improved += 1
        
            
        print("Avg validation loss: {}".format(val_loss_avg))
        print("Avg MAE TTNE Standardized: {}".format(avg_MAE_stand))
        print("Avg MAE TTNE minutes: {}".format(avg_MAE_min))
        print("Avg Accuracy Next activity prediction {}".format(act_accuracy))
        print("Avg CE validation set: {}".format(CE_avg))
        

        # Saving checkpoint after every epoch
        model_path = os.path.join(path_name, 'model_epoch_{}.pt'.format(epoch))
        checkpoint = {'epoch:' : epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'loss': last_loss}
        torch.save(checkpoint, model_path)
        
        if lr_scheduler_present:
            if lstm_scheduler:
                # Update the learning rate
                lr_scheduler.step(val_loss_avg)
            else:
                lr_scheduler.step()
        torch.cuda.empty_cache()

        if num_epochs_not_improved == 42:
            print("No improvements in validation loss for 42 consecutive epochs. Final epoch: {}".format(epoch))
            break

    # Writing training progress to csv at the end of the current training loop
    results_path = os.path.join(path_name, 'backup_results.csv')
    # if both:
    epoch_list = [i for i in range(len(train_losses_global))]
    results = pd.DataFrame(data = {'epoch' : epoch_list, 
                        'composite training loss' : train_losses_global, 
                        'activity training loss (cross entropy)': train_losses_act, 
                        'TTNE training loss (MAE)': train_losses_ttne, 
                        'composite validation loss' : val_loss_glob,
                        'TTNE - MAE standardized' : avg_MAE_standardized_glob, 
                        'TTNE - MAE minutes' : avg_MAE_minutes_glob,
                        'Cross Entropy validation set next activity' : avg_CE_inference_glob, 
                        'Next Activity Accuracy (validation)' : avg_accuracy_glob})
    results.to_csv(results_path, index=False)
    return model, optimizer, epoch, last_loss