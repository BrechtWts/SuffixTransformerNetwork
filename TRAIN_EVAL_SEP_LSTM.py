"""Module containing the entire pipeline to train and evaluate the 
SEP-LSTM benchmark. 
"""
import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle 



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
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
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

def train_eval(log_name, 
               tss_index):
    """Training and automatically evaluating the SEP-LSTM benchmark 
    model with the parameters used in the SuTraN paper. 

    Parameters
    ----------
    log_name : str
        Name of the event log on which the model is trained. Should be 
        the same string as the one specified for the `log_name` parameter 
        of the `log_to_tensors()` function in the 
        `Preprocessing\from_log_to_tensors.py` module. 
    tss_index : int 
        Zero-based index at which the time since start (tss) prefix 
        event feature was stored in the original, fully data-aware 
        datasets. The time since previous event (tsp) prefix event 
        feature is stored at index `tss_index+1`. These indices 
        are needed for the SEP LSTMs to retrieve the training mean 
        and standard deviation used for standardization of these two 
        features, since they need to be updated during final test set 
        suffix generation. 
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
    mean_std_tss_pref = [train_means_dict['prefix_df'][tss_index], train_std_dict['prefix_df'][tss_index]]
    mean_std_tsp_pref = [train_means_dict['prefix_df'][tss_index+1], train_std_dict['prefix_df'][tss_index+1]]
    # mean_std_tss_pref = [train_means_dict['prefix_df'][5], train_std_dict['prefix_df'][5]]
    # mean_std_tsp_pref = [train_means_dict['prefix_df'][6], train_std_dict['prefix_df'][6]]
    mean_std_rrt = [train_means_dict['timeLabel_df'][1], train_std_dict['timeLabel_df'][1]]
    num_numericals_pref = len(num_cols_dict['prefix_df'])
    num_numericals_suf = len(num_cols_dict['suffix_df'])

    num_categoricals_pref, num_categoricals_suf = len(cat_cols_dict['prefix_df']), len(cat_cols_dict['suffix_df'])


    dropout = 0.2
    batch_size = 128


    # specifying path results and callbacks 
    backup_path = os.path.join(log_name, "SEP_LSTM_results")
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



    # Benchmark model needs left-padded prefix events instead of right padded 
    from OneStepAheadBenchmarks.convert_tensordata import convert_to_SEP_data

    train_dataset = convert_to_SEP_data(data=train_dataset, 
                                        outcome_bool=False,
                                        num_categoricals_pref=num_categoricals_pref, 
                                        act_input_index=num_categoricals_pref-1, 
                                        tss_index=tss_index)
    val_dataset = convert_to_SEP_data(data=val_dataset, 
                                        outcome_bool=False,
                                        num_categoricals_pref=num_categoricals_pref, 
                                        act_input_index=num_categoricals_pref-1, 
                                        tss_index=tss_index)

    test_dataset = convert_to_SEP_data(data=test_dataset, 
                                        outcome_bool=False,
                                        num_categoricals_pref=num_categoricals_pref, 
                                        act_input_index=num_categoricals_pref-1, 
                                        tss_index=tss_index)

    # Creating TensorDataset for the training set 
    train_dataset = TensorDataset(*train_dataset)
    val_dataset = TensorDataset(*val_dataset)


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

    # Import Benchmark model 
    from OneStepAheadBenchmarks.models import SEP_Benchmarks_time

    model = SEP_Benchmarks_time(d_model=100,
                                num_activities=num_activities, 
                                dropout=0.2, 
                                num_shared_layers=1, 
                                num_specialized_layers=1)



    # Assign to GPU 
    model.to(device)

    # Initializing optimizer and learning rate scheduler 
    lr = 0.002

    # Optimizer and scheduler used by benchmark
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, 
                                                            threshold=0.0001, threshold_mode='rel', 
                                                            cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    
    # Training procedure 
    start_epoch = 0
    num_epochs = 500 
    num_classes = num_activities 
    batch_interval = 1600

    from OneStepAheadBenchmarks.train_procedure import train_model

    train_model(model=model, 
                optimizer=optimizer, 
                train_dataset=train_dataset, 
                val_dataset=val_dataset, 
                start_epoch=0, 
                num_epochs=num_epochs, 
                num_classes=num_activities, 
                batch_interval=1600, 
                path_name=backup_path, 
                mean_std_ttne=mean_std_ttne, 
                batch_size=128, 
                lr_scheduler_present=True, 
                lstm_scheduler=True,
                lr_scheduler=lr_scheduler)


    
    # Re-initializing new model after training to load best callback
    model = SEP_Benchmarks_time(d_model=100,
                                num_activities=num_activities, 
                                dropout=0.2, 
                                num_shared_layers=1, 
                                num_specialized_layers=1)
    # Assign to GPU 
    model.to(device)

    # Specifying path of csv in which the training and validation results 
    # of every epoch are stored. 
    final_results_path = os.path.join(backup_path, 'backup_results.csv')

    # Reading in the final results csv 
    import pandas as pd 
    df = pd.read_csv(final_results_path)

    # Retrieving the row with the best validation loss 
    row_with_lowest_loss = df.loc[df['composite validation loss'].idxmin()]

    # Retrieve the value of the 'epoch' column for that row
    epoch_value = row_with_lowest_loss['epoch']

    # The models are stored with the string underneath (but not an LSTM)
    best_epoch_string = 'model_epoch_{}.pt'.format(int(epoch_value))
    best_epoch_path = os.path.join(backup_path, best_epoch_string)

    # Loadd best model into memory again 
    model, _, _, _ = load_checkpoint(model, path_to_checkpoint=best_epoch_path, train_or_eval='eval', lr=0.002)
    model.to(device)
    model.eval()

    # Running final inference on test set 
    from OneStepAheadBenchmarks.inference_procedure_suffix_gen import inference_loop

    # Initializing directory for final test set results 
    results_path = os.path.join(backup_path, "TEST_SET_RESULTS")
    os.makedirs(results_path, exist_ok=True)

    inf_results = inference_loop(model, 
                                test_dataset, 
                                mean_std_ttne, 
                                mean_std_tss_pref, 
                                mean_std_tsp_pref,
                                mean_std_rrt, 
                                1, # num_categoricals_pref
                                2, # num_numericals_pref
                                results_path=results_path, 
                                val_batch_size=2048)
    
    avg_MAE1_stand, avg_MAE1_seconds, avg_MAE1_minutes, avg_MAE2_stand = inf_results[:4]
    avg_MAE2_seconds, avg_MAE2_minutes, avg_dam_lev, percentage_too_early = inf_results[4:8] 
    percentage_too_late, percentage_correct, mean_absolute_length_diff, mean_too_early =  inf_results[8:12]
    mean_too_late, avg_MAE_rrt_sum_minutes = inf_results[12:14]
    print("Avg MAE Type 1 RRT suffix prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE1_stand, avg_MAE1_minutes))
    print("Avg MAE Type 2 RRT suffix prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE2_stand, avg_MAE2_minutes))
    # print("Avg Cross Entropy acitivty suffix prediction validation set: {}".format(avg_inference_CE))
    print("Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev))
    print("Percentage of suffixes predicted to END: too early - {} ; right moment - {} ; too late - {}".format(percentage_too_early, percentage_correct, percentage_too_late))
    print("Too early instances - avg amount of events too early: {}".format(mean_too_early))
    print("Too late instances - avg amount of events too late: {}".format(mean_too_late))
    print("Avg absolute amount of events predicted too early / too late: {}".format(mean_absolute_length_diff))
    print("Avg MAE RRT prediction validation set: {} (minutes)'".format(avg_MAE_rrt_sum_minutes))


    # Retrieving and storing dictionary of the metrics averaged over all 
    # test set instances (prefix-suffix pairs)
    avg_results_dict = {"MAE TTNE minutes" : avg_MAE2_minutes, 
                        "DL sim" : avg_dam_lev, 
                        "MAE RRT minutes" : avg_MAE_rrt_sum_minutes}
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




