"""Functionality training procedure for ED-LSTM benchmark"""


import torch
import torch.nn as nn
from LSTM_seq2seq.train_utils import MultiOutputLoss
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.data import DataLoader
from LSTM_seq2seq.inference_procedure import inference_loop

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

        # Keep track of clipped gradient norm 
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
                num_categoricals_pref,
                mean_std_ttne, 
                mean_std_tsp, 
                mean_std_tss, 
                mean_std_rrt, 
                batch_size, 
                patience = 24,
                lr_scheduler_present=False, 
                lr_scheduler=None, 
                best_MAE_ttne = 1e9, 
                best_DL_sim = -1, 
                best_MAE_rrt = 1e9, 
                max_norm = 2.):
    """Outer training loop for the ED-LSTM benchmark. 

    Parameters
    ----------
    model : EncDecLSTM_no_context
        Initialized and current version of the ED-LSTM benchmark model. 
    optimizer : torch optimizer
        torch.optim.AdamW optimizer. Should already be initialized and 
        wrapped around the parameters of `model`. 
    train_dataset : torch.utils.data.dataset.TensorDataset
        TensorDataset containing the training data. 
    val_dataset : tuple of torch.Tensor 
        Containing the tensors comprising the separate validation 
        dataset. Unlike `train_dataset`, the validation dataset should 
        still be the original tuple, and should hence not already be 
        transformed into a TensorDataset instance. 
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
    num_categoricals_pref : int
        The number of categorical features (including the activity label) 
        contained within each prefix event token. Given the NDA 
        characteristic of ED-LSTM, this should be specified as one. 
        Allowed for flexibility for future studies. 
    mean_std_ttne : list of float
        Training mean and standard deviation used to standardize the time 
        till next event (in seconds) target. Needed for re-converting 
        ttne predictions to original scale. Mean is the first entry, 
        std the second.
    mean_std_tsp : list of float
        Training mean and standard deviation used to standardize the time 
        since previous event (in seconds) feature of the decoder suffix 
        tokens. Needed for re-converting time since previous event values 
        to original scale (seconds). Mean is the first entry, std the 2nd.
    mean_std_tss : list of float
        Training mean and standard deviation used to standardize the time 
        since start (in seconds) feature of the decoder suffix tokens. 
        Needed for re-converting time since start to original scale 
        (seconds). Mean is the first entry, std the 2nd. 
    mean_std_rrt : list of float, optional
        List consisting of two floats, the training mean and standard 
        deviation of the remaining runtime labels (in seconds). Needed 
        for de-standardizing remaining runtime predictions and labels, 
        such that the MAE can be expressed in seconds (and minutes). 
    batch_size : int 
        Batch size used during training. 
    patience : int, optional. 
        Max number of epochs without any improvement in any of the 
        validation metrics. After `patience` epochs without any 
        improvement, the training loop is terminated early. By default 24.
    lr_scheduler_present : bool, optional
        Indicates whether we work with a learning rate scheduler wrapped 
        around the optimizer. If True, learning rate scheduler 
        included. If False (default), not. 
    lr_scheduler : torch lr_scheduler or None
        If ``lr_scheduler_present=True``, a lr_scheduler that is wrapped 
        around the optimizer should be provided as well. For ED-LSTM, 
        the ExponentialLR() scheduler should be used. 
    best_MAE_ttne : float 
        Best validation Mean Absolute Error for the time till 
        next event suffix prediction. The defaults apply if the training 
        loop is initialized for the first time for a given configuration. 
        If the training loop is resumed from a certain checkpoint, the 
        best results of the previous training loop should be given. 
    best_DL_sim : float 
        Best validation 1-'normalized Damerau-Levenshtein distance for 
        activity suffix prediction so far. The defaults apply if the  
        training loop is initialized for the first time for a given 
        configuration. If the training loop is resumed from a certain 
        checkpoint, the best results of the previous training loop should 
        be given. 
    best_MAE_rrt : float
        Best validation Mean Absolute Error for the remaining runtime 
        prediction so far. The defaults apply if the training 
        loop is initialized for the first time for a given configuration. 
        If the training loop is resumed from a certain checkpoint, the 
        best results of the previous training loop should be given. If 
        `remaining_runtime_head=False`, the defaults can be retained even 
        when resuming the training loop from a checkpoint. 
    max_norm : float, optional
        Max gradient norm used for clipping during training. By default 2.
    """
    if lr_scheduler_present:
        if lr_scheduler==None:
            print("No lr_scheduler provided.")
            return -1, -1, -1, -1

    # Checking whether GPU is used
    print("Device: {}".format(device))

    # Assigning ED-LSTM to gpu
    model.to(device)

    # Creating train and validation dataloaders 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    # Tracking running averages over last ``batch_interval`` batches of each epoch
    # & tracking average validation losses
    train_losses_global = []
    train_losses_act = []
    train_losses_ttne = []

    avg_MAE_ttne_stand_glob, avg_MAE_ttne_minutes_glob = [], []
    avg_dam_lev_glob, perc_too_early_glob, perc_too_late_glob, perc_correct_glob = ([] for _ in range(4))
    mean_absolute_length_diff_glob, mean_too_early_glob, mean_too_late_glob, avg_MAE_minutes_RRT_glob = ([] for _ in range(4))

    val_loss_glob = []
    val_loss_best = 1000
    
    loss_fn = MultiOutputLoss(num_classes)

    # Early stopping with patience for validation loss 
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

        inf_results = inference_loop(model,
                                    val_dataset, 
                                    num_categoricals_pref, 
                                    mean_std_ttne, 
                                    mean_std_tsp, 
                                    mean_std_tss, 
                                    mean_std_rrt, 
                                    results_path=None, 
                                    val_batch_size=2048)
        
        avg_MAE_ttne_stand, avg_MAE_ttne_minutes = inf_results[:2]
        avg_dam_lev, percentage_too_early, percentage_too_late = inf_results[2:5]
        percentage_correct, mean_absolute_length_diff, mean_too_early, mean_too_late, avg_MAE_rrt_sum_minutes = inf_results[5:10]

        better = False
        if avg_MAE_ttne_stand < best_MAE_ttne: 
            better = True
            best_MAE_ttne = avg_MAE_ttne_stand

        if avg_dam_lev > best_DL_sim:
            better = True 
            best_DL_sim = avg_dam_lev
        if avg_MAE_rrt_sum_minutes < best_MAE_rrt: 
            better = True
            best_MAE_rrt = avg_MAE_rrt_sum_minutes
        if better == False: 
            num_epochs_not_improved += 1
        else:
            num_epochs_not_improved = 0
        # Store other validation measures
        #   TTNE measures
        avg_MAE_ttne_stand_glob.append(avg_MAE_ttne_stand)
        avg_MAE_ttne_minutes_glob.append(avg_MAE_ttne_minutes)
        avg_dam_lev_glob.append(avg_dam_lev)
        perc_too_early_glob.append(percentage_too_early)
        perc_too_late_glob.append(percentage_too_late)
        perc_correct_glob.append(percentage_correct)
        mean_absolute_length_diff_glob.append(mean_absolute_length_diff)
        mean_too_early_glob.append(mean_too_early)
        mean_too_late_glob.append(mean_too_late)
        avg_MAE_minutes_RRT_glob.append(avg_MAE_rrt_sum_minutes)

        # Saving checkpoint every epoch
        model_path = os.path.join(path_name, 'model_epoch_{}.pt'.format(epoch))
        checkpoint = {'epoch:' : epoch, 
                      'model_state_dict': model.state_dict(), 
                      'optimizer_state_dict': optimizer.state_dict(), 
                      'loss': last_loss}
        torch.save(checkpoint, model_path)

        
        print("Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand, avg_MAE_ttne_minutes))
        print("Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev))
        print("Percentage of suffixes predicted to END: too early - {} ; right moment - {} ; too late - {}".format(percentage_too_early, percentage_correct, percentage_too_late))
        print("Too early instances - avg amount of events too early: {}".format(mean_too_early))
        print("Too late instances - avg amount of events too late: {}".format(mean_too_late))
        print("Avg absolute amount of events predicted too early / too late: {}".format(mean_absolute_length_diff))
        print("Avg MAE RRT prediction validation set:  {} (minutes)'".format(avg_MAE_rrt_sum_minutes))


        if lr_scheduler_present:
            # Update the learning rate
            lr_scheduler.step()
        
        # Empty GPU cache 
        torch.cuda.empty_cache()

        if num_epochs_not_improved >= patience:
            print("No improvements in validation loss for {} consecutive epochs. Final epoch: {}".format(patience, epoch))
            break
        

    # Writing training progress to csv at the end of the current training loop
    results_path = os.path.join(path_name, 'backup_results.csv')
    epoch_list = [i for i in range(len(train_losses_global))]

    results = pd.DataFrame(data = {'epoch' : epoch_list, 
                        'composite training loss' : train_losses_global, 
                        'activity training loss (cross entropy)': train_losses_act, 
                        'time till next event training loss (MAE)': train_losses_ttne, 
                        'TTNE - standardized MAE validation': avg_MAE_ttne_stand_glob, 
                        'TTNE - minutes MAE validation': avg_MAE_ttne_minutes_glob, 
                        'Activity suffix: 1-DL (validation)': avg_dam_lev_glob,  
                        'Percentage too early (validation)': perc_too_early_glob,    
                        'Percentage correct END prediction (validation)': perc_correct_glob,   
                        'Percentage too late (validation)': perc_too_late_glob,   
                        'Avg absolute amount of events predicted too early / too late (validation)': mean_absolute_length_diff_glob, 
                        'Avg too early (validation)': mean_too_early_glob, 
                        'Avg too late (validation)': mean_too_late_glob, 
                        'RRT - mintues MAE validation': avg_MAE_minutes_RRT_glob})
    results.to_csv(results_path, index=False)


    return model, optimizer, epoch, last_loss