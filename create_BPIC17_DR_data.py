import pandas as pd 
import numpy as np
from Preprocessing.from_log_to_tensors import log_to_tensors
import os 
import torch

def preprocess_bpic17(log):
    """Preprocess the bpic17 event log. 

    Inspired by the code accompanying the following paper:

    "Irene Teinemaa, Marlon Dumas, Marcello La Rosa, and 
    Fabrizio Maria Maggi. 2019. Outcome-Oriented Predictive Process 
    Monitoring: Review and Benchmark. ACM Trans. Knowl. Discov. Data 13, 
    2, Article 17 (April 2019), 
    57 pages. https://doi.org/10.1145/3301300"

    Parameters
    ----------
    log : pandas.DataFrame 
        Event log.

    Returns
    -------
    log : pandas.DataFrame
        Preprocessed event log.
    """

    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], format = 'mixed').dt.tz_convert('UTC')

    # Specifying which activities (if occurring) last indicate whether a case is ...
    relevant_offer_events = ["O_Accepted", "O_Refused", "O_Cancelled"]

    # Retaining only the Offer events 
    log_offer_events = log[log['EventOrigin'] == "Offer"]

    # Getting a dataframe that gives the last Offer activity for each case. 
    last_Offer_Activities = log_offer_events.groupby('case:concept:name', sort=False).last().reset_index()[['case:concept:name','concept:name']]
    last_Offer_Activities.columns = ['case:concept:name', 'last_o_act']

    # Adding that column as a case feature to the main log by merging on case:concept:name: 
    log = log.merge(last_Offer_Activities, on = 'case:concept:name', how = 'left')

    # Subsetting last_Offer_Activities dataframe for only the invalid cases. 
    last_Offer_Activities_invalid = last_Offer_Activities[~last_Offer_Activities['last_o_act'].isin(relevant_offer_events)]

    invalid_cases_list = list(last_Offer_Activities_invalid['case:concept:name'])

    # Dropping all invalid cases (and their events) from the main event log 
    log = log[~log['case:concept:name'].isin(invalid_cases_list)]

    # Adding the three 1-0 target columns 'case accepted', 'case refused', 'case canceled'

    # and adding another categorical case feature that contains 3 levels, indicating whether 
    # a case is 'Accepted', 'Refused' or 'Canceled':
    log['case:outcome'] = log['last_o_act'].copy()
    categorical_outcome_labels = ['Accepted', 'Refused', 'Canceled']
    binary_outcome_colnames = ['case accepted', 'case refused', 'case canceled']
    for idx in range(3):
        offer_event = relevant_offer_events[idx]
        out_label = categorical_outcome_labels[idx]
        out_colname = binary_outcome_colnames[idx]
        log['case:outcome'] = np.where(log['last_o_act'] == offer_event, out_label, log['case:outcome'])
        log[out_colname] = np.where(log['last_o_act'] == offer_event, 1, 0)
    
    log = log.drop(['last_o_act'], axis = 1)

    return log 




def construct_BPIC17_DR_datasets():
    df = pd.read_csv(r'BPIC17_no_loop.csv', header = 0)
    df = preprocess_bpic17(df)
    categorical_casefeatures = ['case:LoanGoal', 'case:ApplicationType']
    numeric_eventfeatures = ['FirstWithdrawalAmount', 'NumberOfTerms', 'MonthlyCost', 'OfferedAmount']
    categorical_eventfeatures = ['org:resource', 'Accepted', 'Selected']
    numeric_casefeatures = ['case:RequestedAmount']
    case_id = 'case:concept:name'
    timestamp = 'time:timestamp'
    act_label = 'concept:name'


    start_date = None
    end_date = "2017-01"
    max_days = 47.81
    window_size = 46
    log_name = 'BPIC_17_DR'
    start_before_date = None
    test_len_share = 0.25
    val_len_share = 0.2
    mode = 'preferred'
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
                            num_casefts=numeric_casefeatures, 
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

