"""Module containing functionality for preprocessing categorical 
features, including the activity labels. 
"""
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import os
import pickle 


def missing_val_cat_mapping(train_df, 
                            test_df, 
                            case_id, 
                            act_label, 
                            timestamp, 
                            cat_casefts, 
                            num_casefts, 
                            cat_eventfts, 
                            num_eventfts, 
                            outcome, 
                            log_name, 
                            window_size):
    """Treat the categoricals, discard all cases with case length > 
    `window_size` (both in train and test set, and slice 
    out only the needed columns for further processing. 


    For each passed on categorical feature (in either `cat_casefts` or 
    `cat_eventfts`, and the activity label), the levels are mapped 
    to unique integers. If a categorical contains any missing values 
    (across the whole df), all missing values are mapped to index 0. If 
    it does not contain any missing values, index 0 is just used for a 
    regular level. Additionally, if the test set contains levels that are 
    not encountered in the training set, an Out-Of-Vocabulary (OOV) token 
    is included, and all of these previously unseen levels in the test 
    set are mapped to that same OOV token. (The OOV token is mapped to 
    an additional index, the last one.)

    Furthermore, all redundant columns (not specified within any of the 
    four lists, `cat_casefts`, `num_casefts`, `cat_eventfts` or `num_eventfts`, 
    or as `case_id`, `act_label` or `timestamp`) will be discarded. 

    This function also writes the following variables to disk: 
    
    * two dictionaries, cardinality_dict and categorical_mapping_dict, 
      as .pkl files. 

    * two dataframes, the resulting train_df and test_df, both with 
      mappings applied, as .csv files. 

    Parameters
    ----------
    train_df : pd.DataFrame
        Cases assigned to the training set. Still contains complete 
        cases only. I.e. the cases should still be parsed into 
        training instances (prefix-suffix pairs). 
    test_df : pd.DataFrame
        Cases assigned to the test set. Still contains complete 
        cases only. I.e. the cases should still be parsed into 
        test instances (prefix-suffix pairs). 
    case_id : str 
        Column name of column containing case IDs.
    act_label : str 
        Column name of column containing activity labels.
    timestamp : str
        Column name of column containing timestamps. Column Should be of 
        the datetime64 dtype. 
    cat_casefts : list of str 
        List of column names of the categorical case features. 
    num_casefts : list of str 
        List of column names of the numeric case features. 
    cat_eventfts : list of str 
        List of column names of the categorical event features. 
    num_eventfts : list of str 
        List of column names of the numeric event features. 
    outcome : str or None
        String representing the column name containing the binary 
        outcome. If `outcome=None`, it will be ignored.
    log_name : str
        Name of the event log to be processed. Will be contained 
        in file names written to disk throughout the entire preprocessing 
        procedure. 
    window_size : int
        The max sequence length. Traces with case length (in number of 
        events) larger than `window_size` will be discarded. For the 
        SuTraN paper, `window_size` is set to percentile 98.5 of the 
        case length distribution, i.e. the 1.5% most extreme outliers 
        in terms of case length were discarded. 
    """
    cat_cols = cat_casefts + cat_eventfts
    num_cols = num_casefts + num_eventfts
    needed_cols = [case_id, act_label, timestamp] + cat_cols + num_cols 
    if outcome:
        needed_cols.append(outcome)
    # Dropping cases with number of events larger than 'window_size'
    train_df['case_length'] = train_df.groupby(case_id, sort = False)[act_label].transform(len)
    test_df['case_length'] = test_df.groupby(case_id, sort = False)[act_label].transform(len)
    train_df = train_df[train_df['case_length'] <= window_size].copy()
    test_df = test_df[test_df['case_length'] <= window_size].copy()

    # Only the wanted / needed columns are retained 
    train_df = train_df[needed_cols].copy()
    test_df = test_df[needed_cols].copy()

    # Temporarily merging train and test df again to 
    # detect any missing values. 
    df = pd.concat([train_df, test_df], axis = 0)

    cat_cols_ext = cat_cols + [act_label]

    missing_value_catcols = []
    # Transform every cat col in category dtype and impute MVs with dedicated level
    for cat_col in tqdm(cat_cols_ext):
        train_df[cat_col] = train_df[cat_col].astype('category') # (retains MVs)
        test_df[cat_col] = test_df[cat_col].astype('category')
        if df[cat_col].isna().any():
            missing_value_catcols.append(cat_col)
            train_df[cat_col] = np.where(train_df[cat_col].isna(), 'MISSINGVL', train_df[cat_col])
            test_df[cat_col] = np.where(test_df[cat_col].isna(), 'MISSINGVL', test_df[cat_col])



    # Dictionary for retrieving the final cardinalities for each categorical, 
    # including potential missing values and OOV tokens. 
    cardinality_dict = {}
    categorical_mapping_dict = {}
    for cat_col in tqdm(cat_cols_ext): 
        cat_to_int = {}
        uni_level_train = list(train_df[cat_col].unique())
        uni_level_test = list(test_df[cat_col].unique())
        if cat_col in missing_value_catcols:
            if 'MISSINGVL' in uni_level_train: 
                uni_level_train.remove('MISSINGVL')
            if 'MISSINGVL' in uni_level_test:
                uni_level_test.remove('MISSINGVL')
            int_mapping = [i for i in range(1, len(uni_level_train)+1)]
            cat_to_int = dict(zip(uni_level_train, int_mapping))
            # Zero for missing values (if any)
            cat_to_int['MISSINGVL'] = 0
            # Every level occurring in test but not train should be 
            # mapped to the same out of value token. (Last level)
            unseen_index = len(int_mapping) + 1 
        else: 
            # If no missing values for that categorical, no MV level of 0 created. 
            int_mapping = [i for i in range(len(uni_level_train))]
            cat_to_int = dict(zip(uni_level_train, int_mapping))
            # Every level occurring in test but not train should be 
            # mapped to the same out of value token. (Last level)
            unseen_index = len(int_mapping)
        
        for test_level in uni_level_test:
            if test_level not in uni_level_train:
                cat_to_int[test_level] = unseen_index
        train_df[cat_col] = train_df[cat_col].map(cat_to_int)
        test_df[cat_col] = test_df[cat_col].map(cat_to_int)

        train_df[cat_col] = train_df[cat_col].astype('int')
        test_df[cat_col] = test_df[cat_col].astype('int')

        # Storing the final cardinalities for each categorical. 
        cardinality_dict[cat_col] = len(set(cat_to_int.values()))
        categorical_mapping_dict[cat_col] = cat_to_int

    # Writing the results to disk: 
    output_directory = log_name
    os.makedirs(output_directory, exist_ok=True)

    train_path = os.path.join(output_directory, log_name+ '_train_mapped.csv')
    test_path = os.path.join(output_directory, log_name + '_test_mapped.csv')
    # Path for saving a dictionary of the final cardinalities 
    cardinality_path = os.path.join(output_directory, log_name + '_cardin_dict.pkl')
    # Path for saving a dictionary of the categorical to int mappings
    # NOTE: May 2024 - One hot encoded tensors for categoricals replaced by 
    #       integer encoded tensors. Additional padding integer (idx 0) included
    #       to cater for the nn.Embedding() modules. Hence, the integer mappings 
    #       for every level should be incremented with one. (shifted 1 pos to the right)
    categorical_mapping_path = os.path.join(output_directory, log_name + '_categ_mapping.pkl')

    with open(cardinality_path, 'wb') as file:
        pickle.dump(cardinality_dict, file)

    with open(categorical_mapping_path, 'wb') as file:
        pickle.dump(categorical_mapping_dict, file)


    # Save within local repo for backup 
    train_df.to_csv(train_path, index = False, header = True)
    test_df.to_csv(test_path, index = False, header = True)

    return train_df, test_df, cardinality_dict
        

def load_cardinality_dict(path_name):
    with open(path_name, 'rb') as file:
        loaded_dict = pickle.load(file)
    
    return loaded_dict