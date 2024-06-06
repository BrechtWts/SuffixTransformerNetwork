"""Source: https://github.com/hansweytjens/predictive-process-monitoring-benchmarks"""


from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.conversion.log import converter
import pandas as pd
import numpy as np

def sort_log(df, case_id = 'case:concept:name', timestamp = 'time:timestamp'):
    """Sort events in event log such that cases that occur first are stored 
    first, and such that events within the same case are stored based on timestamp. 

    Parameters
    ----------
    df : pd.DataFrame 
        Event log to be preprocessed. 
    case_id : str 
        Column name of column containing case IDs.
    timestamp : str
        Column name of column containing timestamps. Column Should be of 
        the datetime64 dtype. 
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

def start_from_date(dataset, start_date, case_id, timestamp):
    """Remove outlier cases starting before `start_date`.

    Parameters
    ----------
    dataset : pd.DataFrame
        Event log to be preprocessed. 
    start_date : str
        "MM-YYYY" format. Only cases starting after or during the 
        specified month will be retained.
    case_id : str, optional
        Column name of column containing case IDs.
    timestamp : str, optional
        Column name of column containing timestamps. Column Should be of 
        the datetime64 dtype.

    Returns
    -------
    dataset : pd.DataFrame
        Updated event log. 
    """
    case_starts_df = pd.DataFrame(dataset.groupby(case_id)[timestamp].min().reset_index())
    case_starts_df['date'] = case_starts_df[timestamp].dt.to_period('M')
    cases_after = case_starts_df[case_starts_df['date'].astype('str') >= start_date][case_id].values
    dataset = dataset[dataset[case_id].isin(cases_after)]
    return dataset.reset_index(drop=True)

def end_before_date(dataset, end_date, case_id, timestamp):
    """Remove outlier cases ending after `end_date`.

    Parameters
    ----------
    dataset : pd.DataFrame
        Event log to be preprocessed. 
    end_date : str
        "MM-YYYY" format. Only cases ending before or during the 
        specified month will be retained.
    case_id : str
        Column name of column containing case IDs.
    timestamp : str
        Column name of column containing timestamps. Column Should be of 
        the datetime64 dtype.

    Returns
    -------
    dataset : pd.DataFrame
        Updated event log. 
    """
    case_stops_df = pd.DataFrame(dataset.groupby(case_id)[timestamp].max().reset_index())
    case_stops_df['date'] = case_stops_df[timestamp].dt.to_period('M')
    cases_before = case_stops_df[case_stops_df['date'].astype('str') <= end_date][case_id].values
    dataset = dataset[dataset[case_id].isin(cases_before)]
    return dataset.reset_index(drop=True)

def start_before_date_select(dataset, start_before_date, case_id, timestamp):
    """Remove cases starting after the date specified by 
    `start_before_date`.

    Only needed for the workaround procedure to mitigate extraction bias 
    in the test set. 

    Parameters
    ----------
    dataset : pd.DataFrame
        Intermediate event log. 
    start_before_date : str
        "MM-YYYY" format. Cases starting after that month 
        will be removed. 
    case_id : str 
        Name case id column.
    timestamp : str
        Name timestamp column. 

    Returns
    -------
    dataset : pd.DataFrame
        Updated event log. 
    """
    case_starts_df = pd.DataFrame(dataset.groupby(case_id)[timestamp].min().reset_index())
    case_starts_df['date'] = case_starts_df[timestamp].dt.to_period('M')
    cases_before = case_starts_df[case_starts_df['date'].astype('str') <= start_before_date][case_id].values
    dataset = dataset[dataset[case_id].isin(cases_before)]
    return dataset.reset_index(drop=True)

def limited_duration(dataset, max_duration, case_id, timestamp):
    """Modified version of the `limited_duration()` function created by 
    Weytjens et al. 
    (https://github.com/hansweytjens/predictive-process-monitoring-benchmarks). 

    Discard cases with a case duration / throughput time larger than 
    `max_duration` (in days). 
    Compared to the original implementation, the end of dataset debiasing 
    step (condition 2 in the original code) is omitted, since the logs 
    did not contain any incomplete cases. 

    Parameters
    ----------
    dataset : pd.DataFrame
        Event log to be preprocessed. 
    max_duration : float
        Max duration retained cases (in days).
    case_id : str
        Column name of column containing case IDs.
    timestamp : str
        Column name of column containing timestamps. Column Should be of 
        the datetime64 dtype.

    Returns
    -------
    dataset : pd.DataFrame
        Updated event log. 
    """
    # compute each case's duration
    agg_dict = {timestamp :['min', 'max']}
    duration_df = pd.DataFrame(dataset.groupby(case_id).agg(agg_dict)).reset_index()
    duration_df["duration"] = (duration_df[(timestamp,"max")] - duration_df[(timestamp,"min")]).dt.total_seconds() / (24 * 60 * 60)
    # condition 1: cases are shorter than max_duration
    condition_1 = duration_df["duration"] <= max_duration *1.00000000001
    cases_retained = duration_df[condition_1][case_id].values
    dataset = dataset[dataset[case_id].isin(cases_retained)].reset_index(drop=True)

    return dataset


def trainTestSplit(df, test_len, case_id, timestamp, mode):
    """Split the dataset in train and test set, applying strict temporal 
    splitting and debiasing the test set.

    Parameters
    ----------
    df : pd.DataFrame
        Event log to be preprocessed. 
    test_len : float
        Fraction of last occurring cases assigned to the test set in 
        the chronological train-test split. In the SuTraN paper, this 
        was set to 0.25 for all event logs. 
    case_id : str, optional
        Column name of column containing case IDs.
    timestamp : str
        Column name of column containing timestamps. Column Should be of 
        the datetime64 dtype.
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

    Returns
    -------
    df_train : pd.DataFrame
        Cases assigned to the training set. Still contains complete 
        cases only. I.e. the cases should still be parsed into 
        training instances (prefix-suffix pairs). 
    df_test : pd.DataFrame
        Cases assigned to the test set. Still contains complete 
        cases only. I.e. the cases should still be parsed into 
        test instances (prefix-suffix pairs). 
    prefix_dict : dict 
        `first_prefix_dict` if `mode='preferred'`, `last_prefix_dict` if 
        `mode='workaround'`. 

        * `first_prefix_dict`: dictionary having case IDs of cases 
          intersecting with the split point / separation time, as keys, 
          and the (chronologically sorted) event index of the first event 
          occurring after separation time as values. This dictionary 
          is needed later on in the preprocessing pipeline, to discard 
          test set prefix-suffix pairs pertaining to prefixes containing 
          only events before separation time, and hence overlapping 
          with running training set cases. 

        * `last_prefix_dict`: dictionary having case IDs of cases 
          intersecting with the split point / separation time, as keys, 
          and the (chronologically sorted) event index of the last event 
          occurring before separation time as values. This dictionary 
          is needed later on in the preprocessing pipeline, to discard 
          training (and validation) set prefix-suffix pairs pertaining to 
          prefixes containing prefix events occurring after separation 
          time, and hence overlappign with test set case. 
    """

    case_starts_df = df.groupby(case_id)[timestamp].min()
    # Sort values puts the first starting case first, the last one last
    # .index.array gets the chronologically sorted list of cases, with 
    # since the case ids were the indices of the case_starts_df pd.series. 
    case_nr_list_start = case_starts_df.sort_values().index.array
    case_stops_df = df.groupby(case_id)[timestamp].max().to_frame()  

    ### TEST SET ###
    # case_nr_list_start chronologically ordered list of all cases. 
    first_test_case_nr = int(len(case_nr_list_start) * (1 - test_len))

    # Split point
    first_test_start_time = np.sort(case_starts_df.values)[first_test_case_nr]


    if mode=='preferred':
        # -----------------------------------------------------
        # List of all cases ending after separation time (`first_test_start_time`)
        test_case_ids_all = list(case_stops_df[case_stops_df[timestamp].values >= first_test_start_time].index)

        # List of all cases starting after separation time 
        test_case_ids_sa = list(case_nr_list_start[first_test_case_nr:])

        # List of cases that overlap. I.e. all cases ending after separation time 
        # but starting before it. For these cases in the test set, only 
        # the prefixes with at least one event after separation time can be 
        # contained within the final test set. 
        test_case_ids_overlap = list(set(test_case_ids_all)-set(test_case_ids_sa))

        # Retain in preliminary test set all cases that end after separation time. 
        df_test = df[df[case_id].isin(test_case_ids_all)].reset_index(drop=True).copy()

        train_case_ids = case_stops_df[case_stops_df[timestamp].values < first_test_start_time].index.array  # added values
        df_train = df[df[case_id].isin(train_case_ids)].reset_index(drop=True)

        # For overlapping test cases, derive dictionary of first prefix idx to be contained within 
        # the ultimate test set. 
        df_test_overlap = df_test[df_test[case_id].isin(test_case_ids_overlap)].copy().reset_index(drop=True)
        # Add event index column 
        df_test_overlap['evt_idx'] = df_test_overlap.groupby([case_id]).cumcount()
        # Get dataframe of only the events of those cases that occur after seperation time
        df_test_overlap_prefixes = df_test_overlap[df_test_overlap[timestamp].values>first_test_start_time].copy().reset_index(drop=True)
        # Only retain first row for each case 
        df_test_overlap_prefixes = df_test_overlap_prefixes.groupby(case_id, sort=False, as_index=False).first().reset_index(drop=True)
        # Deriving dictionary
        overlap_cases_list = list(df_test_overlap_prefixes[case_id])
        overlap_evt_ids = list(df_test_overlap_prefixes['evt_idx'])
        first_prefix_dict = dict(zip(overlap_cases_list, overlap_evt_ids))

        return df_train, df_test, first_prefix_dict
    elif mode=='workaround':
        # Dataframe containing for each case id the start timestamp 
        case_starts_df = df.groupby(case_id)[timestamp].min().to_frame()  
        # In test set: all cases that start at or after separation time 
        test_case_ids = list(case_starts_df[case_starts_df[timestamp].values >= first_test_start_time].index.array)
        df_test = df[df[case_id].isin(test_case_ids)].copy().reset_index(drop=True)

        # Training case ids: those cases that start before separation time
        train_case_ids_all = list(case_starts_df[case_starts_df[timestamp].values < first_test_start_time].index.array) # added values
        df_train = df[df[case_id].isin(train_case_ids_all)].copy().reset_index(drop=True)
        # -----------------------------------------------------
        # List of ids of all training cases that end before separation time
        train_case_ids_eb = list(case_stops_df[case_stops_df[timestamp].values < first_test_start_time].index.array)
        
        # List of ids of all training cases that start before, but end after, separation time
        train_case_ids_overlap = list(set(train_case_ids_all)-set(train_case_ids_eb))

        # Retaining only the training traces ending after separation time 
        df_train_overlap = df_train[df_train[case_id].isin(train_case_ids_overlap)].copy().reset_index(drop=True)
        # Adding zero-based event indices for each case 
        df_train_overlap['evt_idx'] = df_train_overlap.groupby([case_id]).cumcount()
        # Slicing out only those events that occur before separation time 
        df_train_overlap_prefixes = df_train_overlap[df_train_overlap[timestamp].values < first_test_start_time].copy().reset_index(drop=True)
        # Slicing out only the last event that occurred before separation 
        # time for each case
        df_train_overlap_prefixes = df_train_overlap_prefixes.groupby(case_id, sort=False, as_index=False).last().reset_index(drop=True)
        
        # Deriving dictionary
        last_prefix_dict = df_train_overlap_prefixes.set_index('case:concept:name')['evt_idx'].to_dict()


        return df_train, df_test, last_prefix_dict




def remainTimeOrClassifBenchmark(dataset, 
                                 file_name, 
                                 start_date, 
                                 start_before_date, 
                                 end_date, 
                                 max_days, 
                                 test_len_share, 
                                 case_id, 
                                 timestamp, 
                                 mode, 
                                 output_type="csv"):
    """Create an unbiased train and test set with data leakage prevention 
    according to a strict temporal out-of-time train-test split 
    procedure. Based on the methodlogy proposed by Weytjens et al. 

    Parameters
    ----------
    dataset : pd.DataFrame
        Event log to be preprocessed. 
    file_name : str
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
    case_id : str, optional
        Column name of column containing case IDs.
    timestamp : str
        Column name of column containing timestamps. Column Should be of 
        the datetime64 dtype.
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
    output_type : str, optional
        by default "csv"

    Returns
    -------
    dataset_train : pd.DataFrame
        Cases assigned to the training set. Still contains complete 
        cases only. I.e. the cases should still be parsed into 
        training instances (prefix-suffix pairs). 
    dataset_test : pd.DataFrame
        Cases assigned to the test set. Still contains complete 
        cases only. I.e. the cases should still be parsed into 
        test instances (prefix-suffix pairs). 
    prefix_dict : dict 
        `first_prefix_dict` if `mode='preferred'`, `last_prefix_dict` if 
        `mode='workaround'`. 

        * `first_prefix_dict`: dictionary having case IDs of cases 
          intersecting with the split point / separation time, as keys, 
          and the (chronologically sorted) event index of the first event 
          occurring after separation time as values. This dictionary 
          is needed later on in the preprocessing pipeline, to discard 
          test set prefix-suffix pairs pertaining to prefixes containing 
          only events before separation time, and hence overlapping 
          with running training set cases. 

        * `last_prefix_dict`: dictionary having case IDs of cases 
          intersecting with the split point / separation time, as keys, 
          and the (chronologically sorted) event index of the last event 
          occurring before separation time as values. This dictionary 
          is needed later on in the preprocessing pipeline, to discard 
          training (and validation) set prefix-suffix pairs pertaining to 
          prefixes containing prefix events occurring after separation 
          time, and hence overlappign with test set case. 
    """

    # Convert to datetime if needed
    dataset[timestamp] = pd.to_datetime(dataset[timestamp], utc=True)


    # remove chronological outliers and duplicates
    if start_date:
        # Remove cases starting strictly before the month of start_date
        dataset = start_from_date(dataset, start_date, case_id, timestamp)
    if end_date:
        # Remove cases ending strictly after the month of end_date
        dataset = end_before_date(dataset, end_date, case_id, timestamp)
    if start_before_date:
        dataset = start_before_date_select(dataset, start_before_date, case_id, timestamp)

    # Remove consecutive rows that are completely identical
    dataset.drop_duplicates(inplace=True, ignore_index = True)


    dataset_short = limited_duration(dataset, max_days, case_id, timestamp)
    


    # Split dataset in train and test set, applying strict temporal splitting and debiasing the test set
    dataset_train, dataset_test, prefix_dict = trainTestSplit(dataset_short, 
                                                                    test_len=test_len_share, 
                                                                    case_id=case_id, 
                                                                    timestamp=timestamp, 
                                                                    mode=mode)

    # record outputs
    path = ''
    if output_type == "xes":
        log_train = converter.apply(dataset_train, variant=converter.Variants.TO_EVENT_LOG)
        print("dataset_train converted to logs")
        xes_exporter.apply(log_train, path + "/" + file_name + "_train.xes")
        print("dataset_train exported as xes")
        log_test = converter.apply(dataset_test, variant=converter.Variants.TO_EVENT_LOG)
        print("dataset_test converted to logs")
        xes_exporter.apply(log_test, path + "/" + file_name + "_test.xes")
        print("dataset_test exported as xes")
    elif output_type == "pickle":
        dataset_train.to_pickle(path + "/" + file_name + "_train.pkl")
        dataset_test.to_pickle(path + "/" + file_name + "_test.pkl")
    elif output_type == "csv":
        # dataset_train.to_csv(path + "/" + file_name + "_train.pkl")
        # dataset_test.to_csv(path + "/" + file_name + "_test.pkl")
        dataset_train.to_csv(path + file_name + "_train_initial.csv")
        dataset_test.to_csv(path + file_name + "_test_initial.csv")
    else:
        print("output type unknown. Should be 'xes', 'pickle' or 'csv'")

    return dataset_train, dataset_test, prefix_dict