"""This module takes care of constructing prefix and suffix dataframes, 
given a log (as a pandas dataframe). It first creates unbiased train 
and test sets, encodes categoricals to integers, takes care of missing 
values, constructs the prefix dataframe, suffix dataframe, and 
suffix dataframes pertaining to the labels, and finally takes care of 
numeric features by standardizing them, including the targets. 
"""
import pandas as pd 
import numpy as np 
from Preprocessing.create_benchmarks import remainTimeOrClassifBenchmark
from Preprocessing.categ_mapping_mv import missing_val_cat_mapping
from Preprocessing.prefix_suffix_creation import construct_PrefSuff_dfs_pipeline
from Preprocessing.treat_numericals import preprocess_numericals
import os
import pickle 





def sort_log(df, case_id = 'case:concept:name', timestamp = 'time:timestamp', act_label = 'concept:name'):
    """Sort events in event log such that cases that occur first are stored 
    first, and such that events within the same case are stored based on timestamp. 

    Parameters
    ----------
    df: pd.DataFrame 
        Event log to be preprocessed. 
    case_id : str, optional
        Column name of column containing case IDs. By default 
        'case:concept:name'. 
    timestamp : str, optional
        Column name of column containing timestamps. Column Should be of 
        the datetime64 dtype. By default 'time:timestamp'. 
    act_label : str, optional
        Column name of column containing activity labels. By default 
        'concept:name'. 
    """
    df_help = df.sort_values([case_id, timestamp], ascending = [True, True], kind='mergesort').copy()
    # Now take first row of every case_id: this contains first stamp 
    df_first = df_help.drop_duplicates(subset = case_id)[[case_id, timestamp]].copy()
    df_first = df_first.sort_values(timestamp, ascending = True, kind='mergesort')
    # Include integer index to sort on. 
    df_first['case_id_int'] = [i for i in range(len(df_first))]
    df_first = df_first.drop(timestamp, axis = 1)
    df = df.merge(df_first, on = case_id, how = 'left')
    df = df.sort_values(['case_id_int', timestamp], ascending = [True, True], kind='mergesort')
    df = df.drop('case_id_int', axis = 1)
    return df.reset_index(drop=True)


def create_numeric_timeCols(df, 
                            case_id = 'case:concept:name', 
                            timestamp = 'time:timestamp', 
                            act_label = 'concept:name'):
    """Adds the following columns to both ``train_df`` and ``test_df``: 

    - 'case_length' : the number of events for each case. Constant for 
      all events of the same case. 
    - 'tt_next' : for each event, the time (in seconds) until the next 
      event occurs. Zero for the last event of each case. 
    - 'ts_prev' : for each event, time (in sec) from previous event. 
      Zero for the first event of each case. 
    - 'ts_start' : for each event, time (in sec) from the first event 
      of that case. Zero for first event of each case. 
    - 'rtime' : for each event, time (in sec) until last event of its 
      case. Zero for last event of each case. 

    Parameters
    ----------
    df: pd.DataFrame 
        Event log to be preprocessed. 
    case_id : str, optional
        Column name of column containing case IDs. By default 
        'case:concept:name'. 
    timestamp : str, optional
        Column name of column containing timestamps. Column Should be of 
        the datetime64 dtype. By default 'time:timestamp'. 
    act_label : str, optional
        Column name of column containing activity labels. By default 
        'concept:name'. 
    """

    # Compute for each case the case length ('num_events'). Needed for later steps. 
    df['case_length'] = df.groupby(case_id, sort = False)[act_label].transform(len)

    # Sorting the cases and events (in case this did not happen yet)
    df = sort_log(df, case_id = case_id, timestamp = timestamp, act_label = act_label)

    # Create a df with only one event per case, with that single event solely containing the 
    # case id and timestamp of the FIRST EVNET (so start time) of every case. 
    case_df = df.drop_duplicates(subset = case_id).copy()
    case_df = case_df[[case_id, timestamp]]
    case_df.columns = [case_id, 'first_stamp']
    
    # Create a df with only one event per case, with that single event solely containing the 
    # case id and timestamp of the LAST EVENT (so end time) of every case. 
    last_stamp_df =  df[[case_id,timestamp]].groupby(case_id, sort = False).last().reset_index()
    last_stamp_df.columns = [case_id, 'last_stamp']

    # Adding the case-constant 'last_stamp' and 'first_stamp' column to the train or test df. 
    df = df.merge(last_stamp_df, on = case_id, how = 'left') # adding 'last_stamp' "case feature"
    df = df.merge(case_df, on = case_id, how = 'left') # adding 'first_stamp' "case feature"

    # Creating the 'next_stamp' column, which contains, for each event of the train or test df, 
    # the timestamp of the subsequent event. Needed for computing the ttne target column for each 
    # event. (For each last event, a NaN value is provided and that will later be filled with O.)
    df['next_stamp']= df.groupby([case_id], sort = False)[timestamp].shift(-1)

    # Time till next event (in seconds)
    df['tt_next'] = (df['next_stamp'] - df[timestamp]) / pd.Timedelta(seconds=1) 
    
    # Exactly same thing as with 'next_stamp', but then 'previous_stamp'. Hence every first event 
    # of every case wil also first get a NaN value, and will later be assigned a 0. In contrast to 
    # the 'next_stamp' column, this 'previous_stamp' column is not needed for computing a prediction 
    # (time label) target, but for the suffixes and prefixes I believe. 
    df['previous_stamp'] = df.groupby([case_id])[timestamp].shift(1)

    # Time since previous event (in seconds)
    df['ts_prev'] = (df[timestamp] - df['previous_stamp']) / pd.Timedelta(seconds = 1)

    # Time since start case (in seconds)
    df['ts_start'] = (df[timestamp] - df['first_stamp']) / pd.Timedelta(seconds = 1)

    # Remaining runtime (in seconds)
    df['rtime'] = (df['last_stamp'] - df[timestamp]) / pd.Timedelta(seconds = 1)

    df.drop(['next_stamp', 'previous_stamp', 'first_stamp', 'last_stamp'], axis = 1, inplace=True)
    # Filling the NaN's of 'ts_prev' (first event of each case) and 'tt_next' (last event of each case) 
    # correctly with 0. 
    values = {'ts_prev': 0, 'tt_next': 0}
    df = df.fillna(value = values)

    return df

def split_train_val(df, 
                    val_len_share, 
                    case_id, 
                    timestamp):
    """Further split up the training set in the ultimate training set and 
    the validation set. This split is carried out after already having 
    separated the event log in a final test set, and the training set. 
    The fraction of cases, indicated by `val_len_share`, hence pertains 
    to the fraction of cases not in the test set, that will be assigned 
    to the validation set. 

    Just as the training-test split, also the train-validation split is 
    an out-of-time split. 
    
    The `val_len_share*100` percent of cases in the current training set 
    (`df`) that start the latest, are assigned to the validation set. 

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the current training set, after having 
        already put aside an out-of-time test set. 
    val_len_share : float 
        Percentage of cases in the training set that is assigned to the 
        validation set. The validation split is performed only after 
        having performed the train-test split, and the percentage hence 
        indicates the percentage of the training set cases after having 
        split the full event log in a training and test set. 
    case_id : str
        Column name of the case id. 
    timestamp: str 
        Column name of the timestamp column.
    """
    # Getting df with 2 cols, case id and the start timestamp for each 
    # case
    case_start_df = df.pivot_table(values=timestamp, index=case_id, aggfunc='min').reset_index().sort_values(by=timestamp, ascending=True).reset_index(drop=True)
    ordered_id_list = list(case_start_df[case_id])

    # Get first validation case index
    first_val_case_id = int(len(ordered_id_list)*(1-val_len_share))

    # Get lists of case ids to be assigned to val and train set
    val_case_ids = ordered_id_list[first_val_case_id:]
    train_case_ids = ordered_id_list[:first_val_case_id]

    # Final train-val split 
    train_set = df[df[case_id].isin(train_case_ids)].copy().reset_index(drop=True)
    val_set = df[df[case_id].isin(val_case_ids)].copy().reset_index(drop=True)

    return train_set, val_set

def remove_overlapping_trainPrefs(train_pref_suff, 
                                  outcome, 
                                  case_id, 
                                  last_prefix_dict):
    prefix_df_train = train_pref_suff[0]
    suffix_df_train = train_pref_suff[1]
    timeLabel_df_train = train_pref_suff[2]
    actLabel_df_train = train_pref_suff[3]
    if outcome:
        outcomeLabel_df_train = train_pref_suff[4]
    # Getting a dictionary of valid overlapping prefix ids 
    valid_prefix_dict = {}
    for cid in last_prefix_dict.keys():
        valid_prefix_dict[cid] = [idx for idx in range(2, last_prefix_dict[cid]+2)]

    # Create list of all valid prefix id strings for the overlapping train cases 
    valid_prefix_ids = list(last_prefix_dict.keys())
    for cid in list(valid_prefix_dict.keys()):
        for prefix_id_retain in valid_prefix_dict[cid]:
            # Get prefix ids that should still be retained in there. 
            prefix_id_str = cid + '_{}'.format(prefix_id_retain)
            valid_prefix_ids.append(prefix_id_str)
    
    # Retrieve a list of invalid prefix ids for the overlapping train 
    # cases 
    overlap_train_ids = list(last_prefix_dict.keys())
    #   Subsetting all prefixes corresponding to overlapping train cases 
    prefix_df_overlap = prefix_df_train[prefix_df_train['orig_case_id'].isin(overlap_train_ids)].copy()
    if len(prefix_df_overlap)>0:
        #   Only retaining the invalid prefix ids 
        prefix_df_overlap = prefix_df_overlap[~prefix_df_overlap[case_id].isin(valid_prefix_ids)].copy()
        # Retrieving list of all invalid train prefix ids 
        invalid_prefix_ids = list(prefix_df_overlap[case_id].unique())

        prefix_df_train = prefix_df_train[~prefix_df_train[case_id].isin(invalid_prefix_ids)].copy()
        suffix_df_train = suffix_df_train[~suffix_df_train[case_id].isin(invalid_prefix_ids)].copy()
        timeLabel_df_train = timeLabel_df_train[~timeLabel_df_train[case_id].isin(invalid_prefix_ids)].copy()
        actLabel_df_train = actLabel_df_train[~actLabel_df_train[case_id].isin(invalid_prefix_ids)].copy()
        if outcome:
            outcomeLabel_df_train = outcomeLabel_df_train[~outcomeLabel_df_train[case_id].isin(invalid_prefix_ids)].copy()
    if outcome:
        return prefix_df_train, suffix_df_train, timeLabel_df_train, actLabel_df_train, outcomeLabel_df_train
    else: 
        return prefix_df_train, suffix_df_train, timeLabel_df_train, actLabel_df_train




def remove_overlapping_testPrefs(test_pref_suff, 
                                 outcome, 
                                 case_id, 
                                 first_prefix_dict):
    prefix_df_test = test_pref_suff[0]
    suffix_df_test = test_pref_suff[1]
    timeLabel_df_test = test_pref_suff[2]
    actLabel_df_test = test_pref_suff[3]
    if outcome:
        outcomeLabel_df_test = test_pref_suff[4]

    # Creating dictionary that for each overlapping test case id contains 
    # a list of (1-based) event indices that occurred before separation 
    # time. 
    invalid_prefix_dict = {}
    for cid in first_prefix_dict.keys():
        invalid_prefix_idx = [idx for idx in range(2, first_prefix_dict[cid]+1)]
        invalid_prefix_dict[cid] = invalid_prefix_idx
    
    # Based on `invalid_prefix_dict`, construct list of all 
    # prefix-suffix pair ids that need to be removed, as they do not 
    # contain a single prefix event occurring after separation time. 
    invalid_prefix_ids = list(first_prefix_dict.keys())
    for cid in invalid_prefix_dict.keys():
        for prefix_id_drop in invalid_prefix_dict[cid]:
            # Get prefix id string to drop 
            prefix_id_str = cid + '_{}'.format(prefix_id_drop)
            invalid_prefix_ids.append(prefix_id_str)
    
    # Remove those prefix-suffix pairs in the 4 or 5 dataframes 
    prefix_df_test = prefix_df_test[~prefix_df_test[case_id].isin(invalid_prefix_ids)]
    suffix_df_test = suffix_df_test[~suffix_df_test[case_id].isin(invalid_prefix_ids)]
    timeLabel_df_test = timeLabel_df_test[~timeLabel_df_test[case_id].isin(invalid_prefix_ids)]
    actLabel_df_test = actLabel_df_test[~actLabel_df_test[case_id].isin(invalid_prefix_ids)]
    if outcome:
        outcomeLabel_df_test = outcomeLabel_df_test[~outcomeLabel_df_test[case_id].isin(invalid_prefix_ids)]

        return prefix_df_test, suffix_df_test, timeLabel_df_test, actLabel_df_test, outcomeLabel_df_test
    else:
        return prefix_df_test, suffix_df_test, timeLabel_df_test, actLabel_df_test


def main_dataframe_pipeline(log,
                            log_name, 
                            start_date, 
                            start_before_date,
                            end_date, 
                            max_days,
                            test_len_share, 
                            val_len_share, 
                            window_size, 
                            log_transformed, 
                            mode,
                            case_id = 'case:concept:name', 
                            act_label = 'concept:name', 
                            timestamp = 'time:timestamp', 
                            cat_casefts = [], 
                            num_casefts = [], 
                            cat_eventfts = [], 
                            num_eventfts = [], 
                            outcome=None):
    """_summary_

    Parameters
    ----------
    log : pd.DataFrame 
        Event log to be preprocessed. 
    log_name : str
        Name of the event log to be processed. Will be contained 
        in file names written to disk throughout the entire preprocessing 
        procedure. 
    start_date : str or None
        "MM-YYYY" format. Only cases starting after or during the 
        specified month will be retained. If `None`, no cases will be 
        discarded based on `start_date`. 
    start_before_date : str or None
        "MM-YYYY" format. If not None, cases starting after that month 
        will be removed. 
    end_date : str or None 
        "MM-YYYY" format. Only cases ending before or during the 
        specified month will be retained. If `None`, no cases will be 
        discarded based on `end_date`. 
    max_days : float
        Max duration retained cases (in days).
    test_len_share : float
        Fraction of last occurring cases assigned to the test set in 
        the chronological train-test split. In the SuTraN paper, this 
        was set to 0.25 for all event logs. 
    val_len_share : float 
        Percentage of cases in the training set that is assigned to the 
        validation set. The validation split is performed only after 
        having performed the train-test split, and the percentage hence 
        indicates the percentage of the training set cases after having 
        split the full event log in a training and test set. In the 
        SuTraN paper, this was set to 0.20 for all event logs. 
    window_size : int
        The max sequence length. Traces with case length (in number of 
        events) larger than `window_size` will be discarded. For the 
        SuTraN paper, `window_size` is set to percentile 98.5 of the 
        case length distribution, i.e. the 1.5% most extreme outliers 
        in terms of case length were discarded. 
    log_transformed : bool
        If `True`, all the numeric features with solely positive values 
        will be log-transformed before being standardized. 
    mode : {'preferred', 'workaround'}
        Manner in which the out-of-time train-test split is performed. 
        The 'preferred' split adheres the the approach adopted from 
        Weytjens et al, and hence, for cases containing 
        events both before and after the split point, the prefix-suffix 
        pairs for which the prefix contains at least one event after the 
        split point are assigned to the test set, while the other pairs 
        pertaining to these cases are discarded. 
        If `mode='workaround'`, prefix-suffix pairs derived from those 
        overlapping cases, and for which all prefix events occur before 
        the split point, are assigned to the training set, while other 
        pairs pertaining to these cases (i.e. pairs for which one or more 
        prefix events are recorded after the plit point) are discarded. 

        Consequently, for `mode='preferred'`, the training set will only 
        contain cases having ended before the split point, while for 
        `mode='workaround'`, the test set will only contain cases having 
        started after the split point. 
    case_id : str, optional
        Column name of column containing case IDs. By default 
        'case:concept:name'. 
    act_label : str, optional
        Column name of column containing activity labels. By default 
        'concept:name'. 
    timestamp : str, optional
        Column name of column containing timestamps. Column Should be of 
        the datetime64 dtype. By default 'time:timestamp'. 
    cat_casefts : list of str 
        List of column names of the categorical case features. 
    num_casefts : list of str 
        List of column names of the numeric case features. 
    cat_eventfts : list of str 
        List of column names of the categorical event features. 
    num_eventfts : list of str 
        List of column names of the numeric event features. 
    outcome : str, optional 
        String representing the name containing the binary outcome 
        column. By default `None`. The default should be retained if the 
        event log does not contain an outcome label, or if no tensor 
        containing the binary outcome labels for each prefix-suffx pair 
        (aka instance) should be generated. 
    """
    if mode not in ['preferred', 'workaround']:
        raise ValueError("Error: mode must be 'preferred' or 'workaround'.")
    
    log = sort_log(df=log, 
                   case_id=case_id, 
                   timestamp=timestamp, 
                   act_label=act_label)

    # Splitting up the log in unbiased train and test sets
    train_df, test_df, prefix_dict = remainTimeOrClassifBenchmark(dataset=log, 
                                                                        file_name=log_name, 
                                                                        start_date=start_date, 
                                                                        start_before_date=start_before_date,
                                                                        end_date=end_date, 
                                                                        max_days=max_days, 
                                                                        test_len_share=test_len_share, 
                                                                        case_id=case_id, 
                                                                        timestamp=timestamp,
                                                                        mode=mode,
                                                                        output_type = 'csv')
    
    # Taking care of missing values, and mapping categoricals' levels to integers. 
    # This function also discards all cases with length (in number of events) 
    # larger than `window_size`, and additionally subsets the data, retaining 
    # only the needed columns 
    train_df, test_df, cardinality_dict = missing_val_cat_mapping(train_df, 
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
                                                                  window_size)
    
    # SPLIT TRAIN_DF INTO TRAIN_DF AND VAL_DF HERE: (proxy of valid out of time)
    # ........................
    train_df, val_df = split_train_val(train_df, 
                                       val_len_share, 
                                       case_id,
                                       timestamp)


    # Add 4 numeric time columns ( and 'case_length' to both the train and test df
    train_df = create_numeric_timeCols(train_df, case_id, timestamp, act_label)
    test_df = create_numeric_timeCols(test_df, case_id, timestamp, act_label)
    val_df = create_numeric_timeCols(val_df, case_id, timestamp, act_label)


    # Constructing prefix and suffix dataframes for both train and test set. 
    # Function outputs a tuple of 4 (or 5) dataframes: 
    #   prefix_df, suffix_df, timeLabel_df, actLabel_df (, outcomeLabel_df)
    train_pref_suff = construct_PrefSuff_dfs_pipeline(train_df, window_size, outcome, 
                                                      case_id = case_id, timestamp = timestamp, 
                                                      act_label = act_label, cat_casefts = cat_casefts, 
                                                      num_casefts = num_casefts, cat_eventfts = cat_eventfts, 
                                                      num_eventfts = num_eventfts)
    
    val_pref_suff = construct_PrefSuff_dfs_pipeline(val_df, window_size, outcome, 
                                                    case_id = case_id, timestamp = timestamp, 
                                                    act_label = act_label, cat_casefts = cat_casefts, 
                                                    num_casefts = num_casefts, cat_eventfts = cat_eventfts, 
                                                    num_eventfts = num_eventfts)
    
    test_pref_suff = construct_PrefSuff_dfs_pipeline(test_df, window_size, outcome, 
                                                     case_id = case_id, timestamp = timestamp, 
                                                     act_label = act_label, cat_casefts = cat_casefts, 
                                                     num_casefts = num_casefts, cat_eventfts = cat_eventfts, 
                                                     num_eventfts = num_eventfts)
    if mode == 'preferred':
        # Removing invalid prefixes from overlapping test cases. 
        test_pref_suff = remove_overlapping_testPrefs(test_pref_suff, outcome, 
                                                    case_id, prefix_dict)
    elif mode == 'workaround':
        # Removing invalid prefixes from overlapping train cases. 
        train_pref_suff = remove_overlapping_trainPrefs(train_pref_suff,
                                                        outcome, 
                                                        case_id, 
                                                        prefix_dict)
        val_pref_suff = remove_overlapping_trainPrefs(val_pref_suff, 
                                                      outcome, 
                                                      case_id, 
                                                      prefix_dict)
    
    # Specify numeric columns for each of the 4 created dfs
    pref_numcols = num_casefts + num_eventfts + ['ts_start', 'ts_prev']
    suff_numcols = ['ts_start', 'ts_prev']
    timelab_numcols = ['tt_next', 'rtime']

    # Specify categorical columns for each of the 4 created dfs
    pref_catcols = cat_casefts + cat_eventfts + [act_label] 
    suff_catcols = [act_label] 
    actlab_catcols = [act_label] 

    # Preprocessing numericals in each of the 4 (or 5) resulting dataframes:
    train_pref_suff, val_pref_suff, test_pref_suff, indicator_prefix_cols, train_means_dict, train_std_dict = preprocess_numericals(train_pref_suff=train_pref_suff, 
                                                                                                                                    val_pref_suff=val_pref_suff,
                                                                                                                                    test_pref_suff=test_pref_suff, 
                                                                                                                                    case_id=case_id,
                                                                                                                                    pref_numcols=pref_numcols, 
                                                                                                                                    suff_numcols=suff_numcols, 
                                                                                                                                    timelab_numcols=timelab_numcols,
                                                                                                                                    outcome=outcome, 
                                                                                                                                    log_transformed=log_transformed)

    # Add binary indicator cols for missing values to the ``pref_numcols`` list
    pref_numcols += indicator_prefix_cols

    # Create a dictionary containing the numeric cols needed for each of the 4 created dfs
    num_cols_dict = {'prefix_df': pref_numcols, 'suffix_df': suff_numcols, 'timeLabel_df': timelab_numcols}

    # Create a dictionary containing the categorical cols needed for each of the 4 created dfs
    cat_cols_dict = {'prefix_df': pref_catcols, 'suffix_df': suff_catcols, 'actLabel_df': actlab_catcols}

    # Writing the results to disk: 
    output_directory = log_name
    os.makedirs(output_directory, exist_ok=True)

    # Path for saving a dictionary 'num_cols_dict'
    num_cols_dict_path = os.path.join(output_directory, log_name + '_num_cols_dict.pkl')
    # Path for saving a dictionary 'cat_cols_dict'
    cat_cols_dict_path = os.path.join(output_directory, log_name + '_cat_cols_dict.pkl')
    # Path for saving dictionary 'train_means_dict'
    train_means_dict_path = os.path.join(output_directory, log_name + '_train_means_dict.pkl')
    # Path for saving dictionary 'train_std_dict'
    train_std_dict_path = os.path.join(output_directory, log_name + '_train_std_dict.pkl')

    with open(num_cols_dict_path, 'wb') as file:
        pickle.dump(num_cols_dict, file)

    with open(cat_cols_dict_path, 'wb') as file:
        pickle.dump(cat_cols_dict, file)

    with open(train_means_dict_path, 'wb') as file:
        pickle.dump(train_means_dict, file)

    with open(train_std_dict_path, 'wb') as file:
        pickle.dump(train_std_dict, file)
    
    return train_pref_suff, val_pref_suff, test_pref_suff, cardinality_dict, num_cols_dict, cat_cols_dict, train_means_dict, train_std_dict
    
