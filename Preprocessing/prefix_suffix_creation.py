
import pandas as pd 
import numpy as np 
from tqdm import tqdm



def sort_log(df, case_id = 'case:concept:name', timestamp = 'time:timestamp', act_label = 'concept:name'):
    """Sort events in event log such that cases that occur first are stored 
    first, and such that events within the same case are stored based on timestamp. 

    Parameters
    ----------
    df : _type_
        _description_
    case_id : str, optional
        _description_, by default 'case:concept:name'
    timestamp : str, optional
        _description_, by default 'time:timestamp'
    act_label : str, optional
        _description_, by default 'concept:name'
    """
    df_help = df.sort_values([case_id, timestamp], ascending = [True, True], kind='mergesort')
    # Now take first row of every case_id: this contains first stamp 
    df_first = df_help.drop_duplicates(subset = case_id)[[case_id, timestamp]]
    df_first = df_first.sort_values(timestamp, ascending = True, kind='mergesort')
    # Include integer index to sort on. 
    df_first['case_id_int'] = [i for i in range(len(df_first))]
    df_first = df_first.drop(timestamp, axis = 1)
    df = df.merge(df_first, on = case_id, how = 'left')
    df = df.sort_values(['case_id_int', timestamp], ascending = [True, True], kind='mergesort')
    df = df.drop('case_id_int', axis = 1)
    return df 

def create_prefix_suffixes(df, window_size, outcome, case_id = 'case:concept:name', 
                           timestamp = 'time:timestamp', act_label = 'concept:name', 
                           cat_casefts = [], num_casefts = [], 
                           cat_eventfts = [], num_eventfts = []):
    """Create dataframes for the prefixes (input encoder), decoder suffix 
    (input decoder), activity labels, time till next event labels and 
    remaining runtime labels.

    Parameters
    ----------
    df : _type_
        _description_
    window_size : _type_
        _description_
    outcome : {None, str}
        If a binary outcome column is contained within the event log and 
        outcome prediction labels are needed, outcome should be a string 
        representing the column name that contains the binary outcome. If 
        no binary outcome is present in the event log, or outcome 
        prediction is not needed, `outcome` is None. 
    case_id : str, optional
        _description_, by default 'case:concept:name'
    timestamp : str, optional
        _description_, by default 'time:timestamp'
    act_label : str, optional
        _description_, by default 'concept:name'
    cat_casefts : list, optional
        _description_, by default []
    num_casefts : list, optional
        _description_, by default []
    cat_eventfts : list, optional
        _description_, by default []
    num_eventfts : list, optional
        _description_, by default []
    """
    # Selecting needed columns for prefixes 
    # NOTE: actually timestamp not needed anymore 
    prefix_cols = [case_id, act_label, timestamp, 'ts_start', 'ts_prev', 'case_length'] + cat_casefts + num_casefts + cat_eventfts + num_eventfts
    prefix_subset = df[prefix_cols].copy()

    # Selecting needed columns for decoder suffixes 
    # NOTE: actually timestamp not needed anymore 
    suffix_cols = [case_id, act_label, timestamp, 'ts_start', 'ts_prev', 'case_length']
    suffix_subset = df[suffix_cols].copy()
    
    # Selecting needed columns for activity label suffixes 
    actLabel_cols = [case_id, act_label, 'case_length']
    actLabel_subset = df[actLabel_cols].reset_index(drop = True).copy()
    actLabel_subset['evt_idx'] = actLabel_subset.groupby(['case:concept:name']).cumcount() + 1

    # Adding an additional 'END' activity for each unique case_id. 
    case_data = df.drop_duplicates(subset = case_id).reset_index(drop = True)[[case_id, 'case_length']].copy()
    case_id_int = [i for i in range(len(case_data))]
    case_data['case_id_int'] = case_id_int
    case_id_list = list(case_data[case_id])
    case_length_list = list(case_data['case_length'])
    case_data = case_data.drop('case_length', axis = 1)
    end_act_list = ['END_TOKEN' for _ in range(len(case_length_list))]
    evt_idx_list = [caselen+1 for caselen in case_length_list]


    # DF already sorted properly. Adding int case id to be able to sort on it again 
    # further down the line for adding END activity. 
    actLabel_subset = actLabel_subset.merge(case_data, on = case_id, how = 'left')
    end_dict = {case_id : case_id_list, 
                act_label : end_act_list, 
                'case_length' : case_length_list, 
                'evt_idx' : evt_idx_list,
                'case_id_int' : case_id_int}
    end_df = pd.DataFrame(end_dict)
    actLabel_subset = pd.concat([actLabel_subset, end_df], axis = 0)
    # Making sure that each case's end event is concatenated after its last event. 
    actLabel_subset = actLabel_subset.sort_values(['case_id_int', 'evt_idx'], ascending = [True, True], kind='mergesort') 
    actLabel_subset = actLabel_subset.drop('evt_idx', axis = 1).reset_index(drop=True)

    # Selecting needed columns for tt_next and rtime labels 
    timeLabel_cols = [case_id, timestamp, 'case_length', 'tt_next', 'rtime']
    timeLabel_subset = df[timeLabel_cols].copy()

    # Selecting needed columns for outcome labels if needed
    if outcome:
        outcomeLabel_cols = [case_id, 'case_length', outcome]
        outcomeLabel_subset = df[outcomeLabel_cols].copy()
    # Making the prefixes and suffixes 
    
    # Initializing the prefix and suffix dataframes for prefix_nr = 1
    # prefixes 
    prefix_df =  prefix_subset[prefix_subset['case_length'] >= 1].groupby(case_id, sort = False).head(1)
    prefix_df['prefix_nr'] = 1
    prefix_df['orig_case_id'] = prefix_df[case_id].copy()
    # decoder suffixes
    suffix_df = suffix_subset.copy()
    suffix_df['prefix_nr'] = 1
    suffix_df['orig_case_id'] = suffix_df[case_id].copy()
    # time label suffixes (both time labels) already as is for first prefix of each case. 
    timeLabel_df = timeLabel_subset.copy()
    timeLabel_df['prefix_nr'] = 1
    timeLabel_df['orig_case_id'] = timeLabel_df[case_id].copy()
    # act labels (shifted)
    actLabel_df = actLabel_subset[actLabel_subset['case_length'] >= 1].groupby(case_id, sort = False).tail(-1)
    actLabel_df['prefix_nr'] = 1
    actLabel_df['orig_case_id'] = actLabel_df[case_id].copy()
    # outcome labels (if needed)
    # HERE YOU ARE: just take a first element or something each time. Perhaps indeed drop dups. Or .first() or something. 
    if outcome:
        # outcomeLabel_df = outcomeLabel_subset[outcomeLabel_subset['case_length'] >= 1].groupby(case_id, sort=False)...
        outcomeLabel_df = outcomeLabel_subset[outcomeLabel_subset['case_length'] >= 1].groupby(case_id, sort=False).head(1)
        outcomeLabel_df['prefix_nr'] = 1
        outcomeLabel_df['orig_case_id'] = outcomeLabel_df[case_id].copy()

    for nr_events in tqdm(range(2, window_size+1)):
        # Prefix DF:
        tmp_pref = prefix_subset[prefix_subset['case_length'] >= nr_events].groupby(case_id, sort = False).head(nr_events)
        tmp_pref['orig_case_id'] = tmp_pref[case_id].copy()
        tmp_pref[case_id] = tmp_pref[case_id] + '_{}'.format(nr_events)
        tmp_pref['prefix_nr'] = nr_events
        prefix_df = pd.concat([prefix_df, tmp_pref], axis = 0)

        # Decoder Suffixes:
        tmp_suff = suffix_subset[suffix_subset['case_length'] >= nr_events].groupby(case_id, sort = False).tail(-(nr_events-1))
        tmp_suff['orig_case_id'] = tmp_suff[case_id].copy()
        tmp_suff[case_id] = tmp_suff[case_id] + '_{}'.format(nr_events)
        tmp_suff['prefix_nr'] = nr_events
        suffix_df = pd.concat([suffix_df, tmp_suff], axis = 0)

        # Time Label suffixes:
        tmp_timelab = timeLabel_subset[timeLabel_subset['case_length'] >= nr_events].groupby(case_id, sort = False).tail(-(nr_events-1))
        tmp_timelab['orig_case_id'] = tmp_timelab[case_id].copy()
        tmp_timelab[case_id] = tmp_timelab[case_id] + '_{}'.format(nr_events)
        tmp_timelab['prefix_nr'] = nr_events
        timeLabel_df = pd.concat([timeLabel_df, tmp_timelab], axis = 0)

        # Act label suffixes: 
        tmp_actlab = actLabel_subset[actLabel_subset['case_length'] >= nr_events].groupby(case_id, sort = False).tail(-nr_events)
        tmp_actlab['orig_case_id'] = tmp_actlab[case_id].copy()
        tmp_actlab[case_id] = tmp_actlab[case_id] + '_{}'.format(nr_events)
        tmp_actlab['prefix_nr'] = nr_events
        actLabel_df = pd.concat([actLabel_df, tmp_actlab], axis = 0)

        # Outcome label (not a suffix, but 1 row for each 
        # prefix - suffix pair, and hence for each unique prefix id)
        if outcome:
            tmp_outLabel = outcomeLabel_subset[outcomeLabel_subset['case_length'] >= nr_events].groupby(case_id, sort = False).head(1)
            tmp_outLabel['orig_case_id'] = tmp_outLabel[case_id].copy()
            tmp_outLabel[case_id] = tmp_outLabel[case_id] + '_{}'.format(nr_events)
            tmp_outLabel['prefix_nr'] = nr_events
            outcomeLabel_df = pd.concat([outcomeLabel_df, tmp_outLabel], axis=0)
    
    actLabel_df.drop(['case_id_int'], axis = 1, inplace = True)
    
    if outcome:
        return prefix_df, suffix_df, timeLabel_df, actLabel_df, outcomeLabel_df
    else:
        return prefix_df, suffix_df, timeLabel_df, actLabel_df




    


    

def construct_PrefSuff_dfs_pipeline(df, window_size, outcome, case_id = 'case:concept:name', 
                                    timestamp = 'time:timestamp', act_label = 'concept:name', 
                                    cat_casefts = [], num_casefts = [], 
                                    cat_eventfts = [], num_eventfts = []):
    """_summary_

    Parameters
    ----------
    df : _type_
        _description_
    window_size : _type_
        _description_
    outcome : {None, str}
        If a binary outcome column is contained within the event log and 
        outcome prediction labels are needed, outcome should be a string 
        representing the column name that contains the binary outcome. If 
        no binary outcome is present in the event log, or outcome 
        prediction is not needed, `outcome` is None. 
    case_id : str, optional
        _description_, by default 'case:concept:name'
    timestamp : str, optional
        _description_, by default 'time:timestamp'
    act_label : str, optional
        _description_, by default 'concept:name'
    """


    # Create prefixes 
    pref_suf_dataframes = create_prefix_suffixes(df, window_size = window_size, 
                                                 outcome=outcome,
                                                 case_id = case_id, timestamp = timestamp, 
                                                 act_label = act_label, 
                                                 cat_casefts = cat_casefts, 
                                                 num_casefts = num_casefts, 
                                                 cat_eventfts = cat_eventfts, 
                                                 num_eventfts = num_eventfts)

    return pref_suf_dataframes 



def handle_missing_data(df, case_id = 'case:concept:name', timestamp = 'time:timestamp', act_label = 'concept:name', numerical_cols = [], cat_cols = []):
    """Handles missing data. Imputes NANs of numerical columns with 0, and 
    creates an indicator variable = 1 if corresponding column had a nan and 
    0 otherwise. We do not create indicator columns for numerical features 
    not containing any NANs. For the categorical columns, we impute NANs simply 
    by filling the NANs with an additional level 'MISSINGV'. 

    Only the columns provided in the arguments are retained in the resulting df. 

    Parameters
    ----------
    df : 
        _description_
    case_id : str, optional
        _description_, by default 'case:concept:name'
    timestamp : str, optional
        _description_, by default 'time:timestamp'
    act_label : str, optional
        _description_, by default 'concept:name'
    numerical_cols : list, optional
        _description_, by default []
    cat_cols : list, optional
        _description_, by default []
    """


"""REMARKS END OF DAY: """
