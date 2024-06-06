import pandas as pd 
import numpy as np
from Preprocessing.from_log_to_tensors import log_to_tensors
import os 
import torch

def preprocess_bpic19(log):
    """Preprocess the bpic19 event log. 

    Parameters
    ----------
    log : pandas.DataFrame 
        Event log.

    Returns
    -------
    log : pandas.DataFrame
        Preprocessed event log.
    """
    # Convert timestamp column to datetime64[ns, UTC] dtype 
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], format='%Y-%m-%d %H:%M:%S%z').dt.tz_convert('UTC')

    # Convert 2 cols to object dtype. 
    log['case:Item'] = log['case:Item'].astype('str')
    log['case:Purchasing Document'] = log['case:Purchasing Document'].astype('str')

    # Converting two booleans to string dtype 
    log['case:GR-Based Inv. Verif.'] = log['case:GR-Based Inv. Verif.'].astype('str')
    log['case:Goods Receipt'] = log['case:Goods Receipt'].astype('str')

    
    return log 




def construct_BPIC19_datasets():
    df = pd.read_csv(r'BPIC19.csv')
    df = preprocess_bpic19(df)
    categorical_casefeatures=[ 'case:Spend area text', 'case:Company', 'case:Document Type', 
                            'case:Sub spend area text', 'case:Item',
                            'case:Vendor', 'case:Item Type', 
                            'case:Item Category', 'case:Spend classification text', 
                            'case:GR-Based Inv. Verif.', 'case:Goods Receipt']
    numeric_eventfeatures= ['Cumulative net worth (EUR)']
    categorical_eventfeatures = ['org:resource']
    num_casefts = []
    case_id = 'case:concept:name'
    timestamp = 'time:timestamp'
    act_label = 'concept:name'


    start_date = "2018-01"
    end_date = "2019-02"
    max_days = 143.33
    window_size = 17
    log_name = 'BPIC_19'
    start_before_date = "2018-09"
    test_len_share = 0.25
    val_len_share = 0.2
    mode = 'workaround'
    outcome = None
    result = log_to_tensors(df, 
                            log_name=log_name, 
                            start_date=start_date, 
                            start_before_date=start_before_date,
                            end_date=end_date, 
                            max_days=max_days, 
                            test_len_share=test_len_share, 
                            val_len_share=val_len_share, 
                            window_size=window_size, 
                            mode=mode,
                            case_id=case_id, 
                            act_label=act_label, 
                            timestamp=timestamp, 
                            cat_casefts=categorical_casefeatures, 
                            num_casefts=num_casefts, 
                            cat_eventfts=categorical_eventfeatures, 
                            num_eventfts=numeric_eventfeatures, 
                            outcome=outcome)
    
    train_data, val_data, test_data = result

    # Create the log_name subfolder in the root directory of the repository
    # (Should already be created when having executed the `log_to_tensors()`
    # function.)
    output_directory = log_name
    os.makedirs(output_directory, exist_ok=True)

    # Save training tuples
    train_tensors_path = os.path.join(output_directory, 'train_tensordataset.pt')
    torch.save(train_data, train_tensors_path)

    # Save validation tuples
    val_tensors_path = os.path.join(output_directory, 'val_tensordataset.pt')
    torch.save(val_data, val_tensors_path)

    # Save test tuples
    test_tensors_path = os.path.join(output_directory, 'test_tensordataset.pt')
    torch.save(test_data, test_tensors_path)

