"""Module containing the entire pipeline to train and evaluate SuTraN. 
"""
import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle 
import sys 




def load_checkpoint(model, path_to_checkpoint, train_or_eval, lr):
    """Loads already trained model into memory with the 
    learned weights, as well as the optimizer in its 
    state when saving the model.

    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html 

    Parameters
    ----------
    model : instance of the CRTP Transformer
        Should just be initialized with the correct initialization 
        arguments, like you would have done in the beginning. 
    path_to_checkpoint : string
        Exact path where the checkpoint is stored on disk. 
    train_or_eval : str, {'train', 'eval'}
        Indicating whether you want to resume training ('train') with the 
        loaded model, or you want to evaluate it ('eval'). The layers of 
        the model will be returned in the appropriate mode. 
    lr : float 
        Learning rate of the optimizer last used for training. 
    
    Returns
    -------
    model : ...
        With trained weights loaded. 
    optimizer : ... 
        With correct optimizer state loaded. 
    final_epoch_trained : int 
        Number of final epoch that the model is trained for. 
        If you want to resume training with the loaded model, 
        start from start_epoch = final_epoch_trained + 1. 
    final_loss : ... 
        Last loss of last epoch. Don't think you need it for resuming 
        training. 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(device)

    if (train_or_eval!= 'train') and (train_or_eval!= 'eval'):
        print("train_or_eval argument should be either 'train' or 'eval'.")
        return -1, -1, -1, -1

    checkpoint = torch.load(path_to_checkpoint)
    # Loading saved weights of the model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # Loading saved state of the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    # optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Loading number of last epoch that the saved model was trained for. 
    final_epoch_trained = checkpoint['epoch:']
    # Last loss of the trained model. 
    final_loss = checkpoint['loss']

    if train_or_eval == 'train':
        model.train()
    else: 
        model.eval()
        
    return model, optimizer, final_epoch_trained, final_loss


def train_eval(log_name):
    """Training and automatically evaluating SuTraN
    with the parameters used in the SuTraN paper. 

    Parameters
    ----------
    log_name : str
        Name of the event log on which the model is trained. Should be 
        the same string as the one specified for the `log_name` parameter 
        of the `log_to_tensors()` function in the 
        `Preprocessing\from_log_to_tensors.py` module. 
    """
    def load_dict(path_name):
        with open(path_name, 'rb') as file:
            loaded_dict = pickle.load(file)
        
        return loaded_dict


    # -----------------
    temp_string = log_name + '_cardin_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_dict = load_dict(temp_path)
    num_activities = cardinality_dict['concept:name'] + 2
    print(num_activities)

    # cardinality list prefix categoricals 
    temp_string = log_name + '_cardin_list_prefix.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_list_prefix = load_dict(temp_path)

    temp_string = log_name + '_cardin_list_suffix.pkl'
    temp_path = os.path.join(log_name, temp_string)
    # cardinality list suffix categoricals
    cardinality_list_suffix = load_dict(temp_path)

    temp_string = log_name + '_num_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    # To retrieve the number of numerical featrues in the prefix and suffix events respectively 
    num_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_cat_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cat_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_train_means_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    train_means_dict = load_dict(temp_path)

    temp_string = log_name + '_train_std_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)

    train_std_dict = load_dict(temp_path)

    mean_std_ttne = [train_means_dict['timeLabel_df'][0], train_std_dict['timeLabel_df'][0]]
    mean_std_tsp = [train_means_dict['suffix_df'][1], train_std_dict['suffix_df'][1]]
    mean_std_tss = [train_means_dict['suffix_df'][0], train_std_dict['suffix_df'][0]]
    # mean_std_tss_pref = [train_means_dict['prefix_df'][5], train_std_dict['prefix_df'][5]]
    # mean_std_tsp_pref = [train_means_dict['prefix_df'][6], train_std_dict['prefix_df'][6]]
    mean_std_rrt = [train_means_dict['timeLabel_df'][1], train_std_dict['timeLabel_df'][1]]
    num_numericals_pref = len(num_cols_dict['prefix_df'])
    num_numericals_suf = len(num_cols_dict['suffix_df'])

    num_categoricals_pref, num_categoricals_suf = len(cat_cols_dict['prefix_df']), len(cat_cols_dict['suffix_df'])

    # Fixed variables 
    d_model = 32 
    num_prefix_encoder_layers = 4
    num_decoder_layers = 4
    num_heads = 8 
    d_ff = 4*d_model 
    prefix_embedding = False
    layernorm_embeds = True
    outcome_bool = False
    remaining_runtime_head = True 
    # Creating auxiliary bools 
    only_rrt = (not outcome_bool) & remaining_runtime_head
    only_out = outcome_bool & (not remaining_runtime_head)
    both_not = (not outcome_bool) & (not remaining_runtime_head)
    both = outcome_bool & remaining_runtime_head
    log_transformed = False
    num_target_tens = 3

    dropout = 0.2
    batch_size = 128


    # specifying path results and callbacks 
    backup_path = os.path.join(log_name, "SUTRAN_DA_results")
    os.makedirs(backup_path, exist_ok=True)


    # Loading train, validation and test sets 
    # Training set 
    temp_path = os.path.join(log_name, 'train_tensordataset.pt')
    train_dataset = torch.load(temp_path)

    # Validation set
    temp_path = os.path.join(log_name, 'val_tensordataset.pt')
    val_dataset = torch.load(temp_path)

    # Test set 
    temp_path = os.path.join(log_name, 'test_tensordataset.pt')
    test_dataset = torch.load(temp_path)

    # Creating TensorDataset for the training set 
    train_dataset = TensorDataset(*train_dataset)

    # Setting up GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)


    # Initializing model 
    import random
    # Set a seed value
    seed_value = 24

    # Set Python random seed
    random.seed(seed_value)

    # Set NumPy random seed
    np.random.seed(seed_value)

    # Set PyTorch random seed
    torch.manual_seed(seed_value)

    # If you are using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        # Additional settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    from SuTraN.SuTraN import SuTraN
    model = SuTraN(num_activities=num_activities, 
                        d_model=d_model, 
                        cardinality_categoricals_pref=cardinality_list_prefix, 
                        num_numericals_pref=num_numericals_pref, 
                        num_prefix_encoder_layers=num_prefix_encoder_layers, 
                        num_decoder_layers=num_decoder_layers, 
                        num_heads= num_heads, 
                        d_ff = 4*d_model, 
                        dropout=dropout, 
                        remaining_runtime_head=True, # Always included. 
                        layernorm_embeds=layernorm_embeds, 
                        outcome_bool=outcome_bool)

    # Assign to GPU 
    model.to(device)

    # Initializing optimizer and learning rate scheduler 
    decay_factor = 0.96
    lr = 0.0002
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_factor)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_factor)

    
    # Training procedure 
    from SuTraN.train_procedure import train_model
    start_epoch = 0
    num_epochs = 200 
    num_classes = num_activities 
    batch_interval = 800
    train_model(model, 
                optimizer, 
                train_dataset, 
                val_dataset, 
                start_epoch, 
                num_epochs, 
                remaining_runtime_head,
                outcome_bool,
                num_classes, 
                batch_interval, 
                backup_path, 
                num_categoricals_pref, 
                mean_std_ttne, 
                mean_std_tsp, 
                mean_std_tss, 
                mean_std_rrt, 
                batch_size, 
                patience=24,
                lr_scheduler_present=True, 
                lr_scheduler=lr_scheduler)
    
    # Re-initializing new model after training to load best callback
    model = SuTraN(num_activities=num_activities, 
                        d_model=d_model, 
                        cardinality_categoricals_pref=cardinality_list_prefix, 
                        num_numericals_pref=num_numericals_pref, 
                        num_prefix_encoder_layers=num_prefix_encoder_layers, 
                        num_decoder_layers=num_decoder_layers, 
                        num_heads= num_heads, 
                        d_ff = 4*d_model, 
                        dropout=dropout, 
                        remaining_runtime_head=True, # Always included. 
                        layernorm_embeds=layernorm_embeds, 
                        outcome_bool=outcome_bool)

    # Assign to GPU 
    model.to(device)

    # Specifying path of csv in which the training and validation results 
    # of every epoch are stored. 
    final_results_path = os.path.join(backup_path, 'backup_results.csv')

    # Determining best epoch based on the validation 
    # scores for RRT and Activity Suffix prediction
    df = pd.read_csv(final_results_path)
    dl_col = 'Activity suffix: 1-DL (validation)'
    rrt_col = 'RRT - mintues MAE validation'
    df['rrt_rank_val'] = df[rrt_col].rank(method='min').astype(int)
    df['dl_rank_val'] = df[dl_col].rank(method='min', ascending=False).astype(int)
    df['summed_rank_val'] = df['rrt_rank_val'] + df['dl_rank_val']

    # Retrieving the row with the best general performance 
    row_with_lowest_loss = df.loc[df['summed_rank_val'].idxmin()]
    # Retrieve the value of the 'epoch' column for that row
    epoch_value = row_with_lowest_loss['epoch']

    # The models are stored with the string underneath
    best_epoch_string = 'model_epoch_{}.pt'.format(int(epoch_value))
    best_epoch_path = os.path.join(backup_path, best_epoch_string)

    # Loadd best model into memory again 
    model, _, _, _ = load_checkpoint(model, path_to_checkpoint=best_epoch_path, train_or_eval='eval', lr=0.002)
    model.to(device)
    model.eval()

    # Running final inference on test set 
    from SuTraN.inference_procedure import inference_loop

    # Initializing directory for final test set results 
    results_path = os.path.join(backup_path, "TEST_SET_RESULTS")
    os.makedirs(results_path, exist_ok=True)

    inf_results = inference_loop(model, 
                                 test_dataset, 
                                 remaining_runtime_head, 
                                 outcome_bool, 
                                 num_categoricals_pref, 
                                 mean_std_ttne, 
                                 mean_std_tsp, 
                                 mean_std_tss, 
                                 mean_std_rrt, 
                                 results_path=results_path, 
                                 val_batch_size=2048)
    

    # Retrieving the different metrics 
    # TTNE MAE metrics
    # avg_MAE1_stand, avg_MAE1_seconds, avg_MAE1_minutes, avg_MAE2_stand, avg_MAE2_seconds, avg_MAE2_minutes = inf_results[:6]
    avg_MAE_ttne_stand, avg_MAE_ttne_minutes = inf_results[:2]
    # # Inference Cross Entropy Activity Suffix prediction
    # avg_inference_CE = inf_results[6]
    # Average Normalized Damerau-Levenshtein distance Activity Suffix 
    # prediction
    avg_dam_lev = inf_results[2]


    # Percentage of validation instances for which the END token was 
    # predicted too early. 
    perc_too_early = inf_results[3]
    # Percentage of validation instances for which the END token was 
    # predicted too late. 
    perc_too_late = inf_results[4]
    # Percentage of validation instances for which the END token was 
    # predicted at the right moment. 
    perc_correct = inf_results[5]
    # Mean absolute lenght difference between predicted and actual 
    # suffix. 
    mean_absolute_length_diff = inf_results[6]
    # Avg num events that END token was predicted too early, averaged 
    # over all instances for which END was predicted too early. 
    mean_too_early = inf_results[7]
    # Avg num events that END token was predicted too late, averaged 
    # over all instances for which END was predicted too late. 
    mean_too_late = inf_results[8]

    
    if only_rrt:
        # MAE standardized RRT predictions 
        avg_MAE_stand_RRT = inf_results[9]
        # MAE RRT converted to minutes
        avg_MAE_minutes_RRT = inf_results[10]
    
    elif only_out:
        # Binary Cross Entropy outcome prediction
        avg_BCE_out = inf_results[9]
        # AUC-ROC outcome prediction
        auc_roc = inf_results[10]
        # AUC-PR outcome prediction
        auc_pr = inf_results[11]
    
    elif both:
        # MAE standardized RRT predictions 
        avg_MAE_stand_RRT = inf_results[9]
        # MAE RRT converted to minutes
        avg_MAE_minutes_RRT = inf_results[10]
        # Binary Cross Entropy outcome prediction
        avg_BCE_out = inf_results[11]
        # AUC-ROC outcome prediction
        auc_roc = inf_results[12]
        # AUC-PR outcome prediction
        auc_pr = inf_results[13]

    # Printing averaged results 
    print("Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand, avg_MAE_ttne_minutes))
    # print("Avg MAE Type 2 TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE2_stand, avg_MAE2_minutes))
    # print("Avg Cross Entropy acitivty suffix prediction validation set: {}".format(avg_inference_CE))
    print("Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev))
    print("Percentage of suffixes predicted to END: too early - {} ; right moment - {} ; too late - {}".format(perc_too_early, perc_correct, perc_too_late))
    print("Too early instances - avg amount of events too early: {}".format(mean_too_early))
    print("Too late instances - avg amount of events too late: {}".format(mean_too_late))
    print("Avg absolute amount of events predicted too early / too late: {}".format(mean_absolute_length_diff))
    if remaining_runtime_head: 
        print("Avg MAE RRT prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_stand_RRT, avg_MAE_minutes_RRT))
    if outcome_bool:
        print("Avg BCE outcome prediction validation set: {}".format(avg_BCE_out))
        print("AUC-ROC outcome prediction validation set: {}".format(auc_roc))
        print("AUC-PR outcome prediction validation set: {}".format(auc_pr))

    # Retrieving and storing dictionary of the metrics averaged over all 
    # test set instances (prefix-suffix pairs)
    avg_results_dict = {"MAE TTNE minutes" : avg_MAE_ttne_minutes, 
                        "DL sim" : avg_dam_lev, 
                        "MAE RRT minutes" : avg_MAE_minutes_RRT}
    path_name_average_results = os.path.join(results_path, 'averaged_results.pkl')

    
    # Retrieving and storing the dictionaries with the 
    # averaged results per prefix and suffix length
    results_dict_pref = inf_results[-2]
    results_dict_suf = inf_results[-1]

    path_name_prefix = os.path.join(results_path, 'prefix_length_results_dict.pkl')
    path_name_suffix = os.path.join(results_path, 'suffix_length_results_dict.pkl')
    with open(path_name_prefix, 'wb') as file:
        pickle.dump(results_dict_pref, file)
    with open(path_name_suffix, 'wb') as file:
        pickle.dump(results_dict_suf, file)
    with open(path_name_average_results, 'wb') as file:
        pickle.dump(avg_results_dict, file)



