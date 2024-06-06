import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler


def standardizeTimeLabel_train_test_dfs(train, 
                                        val, 
                                        test, 
                                        case_id, 
                                        log_transformed):
    """Standardize ~N(0,1) the given numeric columns (of which the column 
    names are contained within the ``num_cols`` list) based on the mean 
    and standard deviation of the training df (``train``). Missing 
    values are disregarded in fit, and hence retained after transform. 
    If `log_transformed=True`, all numeric columns in num_cols that 
    only contain non-negative values (including missing values) are 
    log transformed before standardization to impose a more Gaussian 
    distribution. 

    Parameters
    ----------
    train : pd.DataFrame
        Training instances. 
    val : pd.DataFrame 
        Validation instances. 
    test : pd.DataFrame
        Test instances.
    case_id : str 
        Column name original ID for every prefix-suffix pair. 
        Needed for standardizing the remaining time targets. 
    num_cols : list of str
        The column names of the numeric features to be standardized. 
    log_transformed : bool
        If `True`, all the numeric features and targets with solely 
        positive values will be log-transformed before being standardized.
    
    Returns
    -------
    train : pd.DataFrame
        Training instances with standardized numericals. 
    val : pd.DataFrame 
        Validation instances with standardized numericals. 
    test : pd.DataFrame
        Test instances with standardized numericals. 
    train_mean : list of numpy.float64
        List of the training means used for standardization. Same order 
        as the num_cols list. Needed for de-standardization.
    train_std : list of numpy.float64
        List of the standard deviations used for standardization. Same 
        order as the num_cols list. Needed for de-standardization. 
    """
    if log_transformed:
        # Both time targets will only contain positive values. 
        train[['tt_next', 'rtime']] = np.log1p(train[['tt_next', 'rtime']])
        val[['tt_next', 'rtime']] = np.log1p(val[['tt_next', 'rtime']])
        test[['tt_next', 'rtime']] = np.log1p(test[['tt_next', 'rtime']])

    # Initialize StandardScaler
    scaler_ttne = StandardScaler()
    scaler_rrt = StandardScaler()

    # Fit the scaler on the training data, considering only non-missing values
    # train[num_cols] = scaler.fit_transform(train[num_cols])
    scaler_ttne.fit(train[['tt_next']])
    # train[['tt_next']] = scaler_ttne.fit_transform(train[num_cols])
    train[['tt_next']]= scaler_ttne.transform(train[['tt_next']])
    val[['tt_next']]= scaler_ttne.transform(val[['tt_next']])
    test[['tt_next']]= scaler_ttne.transform(test[['tt_next']])
    
    # Standardize rrt targets based on the first rrt value 
    # for each prefix-suffix pair only. 
    train_case = train.drop_duplicates(subset=case_id).copy()
    scaler_rrt.fit(train_case[['rtime']])
    train[['rtime']]= scaler_rrt.transform(train[['rtime']])
    val[['rtime']]= scaler_rrt.transform(val[['rtime']])
    test[['rtime']]= scaler_rrt.transform(test[['rtime']])
    # Transform the test data using the same scaler
    # test[num_cols] = scaler.transform(test[num_cols])

    # Get the mean values used for standardization
    train_mean = list(scaler_ttne.mean_)
    train_mean.append(list(scaler_rrt.mean_)[0])

    # Get the standard deviations used for standardization
    # train_std = list(np.sqrt(scaler.var_))
    train_std = list(np.sqrt(scaler_ttne.var_))
    # list(scaler_ttne.mean_)
    train_std.append(list(np.sqrt(scaler_rrt.var_))[0])

    return train, val, test, train_mean, train_std

def standardize_train_test_dfs(train, 
                               val,
                               test, 
                            #    case_id, 
                            #    timestamp, 
                            #    act_label, 
                               num_cols, 
                               log_transformed):
    """Standardize ~N(0,1) the given numeric columns (of which the column 
    names are contained within the ``num_cols`` list) based on the mean 
    and standard deviation of the training df (``train``). Missing 
    values are disregarded in fit, and hence retained after transform. 
    If `log_transformed=True`, all numeric columns in num_cols that 
    only contain non-negative values (including missing values) are 
    log transformed before standardization to impose a more Gaussian 
    distribution. 

    Parameters
    ----------
    train : pd.DataFrame
        Training instances. 
    val : pd.DataFrame
        Validation instances. 
    test : pd.DataFrame
        Test instances.
    num_cols : list of str
        The column names of the numeric features to be standardized. 
    log_transformed : bool
        If `True`, all the numeric features and targets with solely 
        positive values will be log-transformed before being standardized.
    
    Returns
    -------
    train : pd.DataFrame
        Training instances with standardized numericals. 
    val : pd.DataFrame
        Validation instances with standardized numericals. 
    test : pd.DataFrame
        Test instances with standardized numericals. 
    train_mean : list of numpy.float64
        List of the training means used for standardization. Same order 
        as the num_cols list. Needed for de-standardization.
    train_std : list of numpy.float64
        List of the standard deviations used for standardization. Same 
        order as the num_cols list. Needed for de-standardization. 
    """
    if log_transformed:
        # list cols in train only containing non-negatives
        train_num_df = train[num_cols]
        train_numCols_pos = list(train_num_df.columns[((train_num_df >= 0) | train_num_df.isnull()).all()])
        # list cols in val only containing non-negatives
        val_num_df = val[num_cols]
        val_numCols_pos = list(val_num_df.columns[((val_num_df >= 0) | val_num_df.isnull()).all()])
        # list cols in test only containing non-negatives
        test_num_df = test[num_cols]
        test_numCols_pos = list(test_num_df.columns[((test_num_df >= 0) | test_num_df.isnull()).all()])
        # Get list of num cols contained in all three lists 
        final_pos_cols = list(set(train_numCols_pos) & set(val_numCols_pos) & set(test_numCols_pos))
        # Log transforming those columns in both train and test 
        train[final_pos_cols] = np.log1p(train[final_pos_cols])
        val[final_pos_cols] = np.log1p(val[final_pos_cols])
        test[final_pos_cols] = np.log1p(test[final_pos_cols])
        

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data, considering only non-missing values
    train[num_cols] = scaler.fit_transform(train[num_cols])

    # Transform the validation data using the same scaler
    val[num_cols] = scaler.transform(val[num_cols])

    # Transform the test data using the same scaler
    test[num_cols] = scaler.transform(test[num_cols])

    # Get the mean values used for standardization
    train_mean = list(scaler.mean_) # same order 

    # Get the standard deviations used for standardization
    train_std = list(np.sqrt(scaler.var_))

    return train, val, test, train_mean, train_std


def preprocess_numericals(train_pref_suff, 
                          val_pref_suff, 
                          test_pref_suff, 
                          case_id, 
                        #   timestamp, 
                        #   act_label, 
                          pref_numcols, 
                          suff_numcols, 
                          timelab_numcols, 
                          outcome, 
                          log_transformed):
    """Preprocess the numericals in each of the generated prefix-suffix 
    dataframes, including the numerical targets. 

    All numerics are standardized ~N(0,1) based on the mean and std in 
    the training instances. 

    If `log_transformed=True`, all numerics that contain no negatives in 
    both the training and test dataframes, will also be log-transformed 
    prior to the standardization. The log-transformation entails taking 
    the natural logarithm of each value incremented with 1, to allow for 
    zeros being present in the data. 

    Only the numerics in ``prefix_df`` can still contain missing values. 
    If any missing values exist in the training and / or test instances, 
    an additional indicator column, containing either 0 or 1, is added. 
    The missing values are imputed with 0, and the corresponding cell in 
    the indicator column is set to 1, and 0 otherwise. Imputating missing 
    values with 0 is only done after standardization on only the 
    non-missing values. This imputation is only done after 
    standardization, based on the non-missing values only. 

    The binary indicator columns created will be treated as numeric columns, 
    even though they only contain zeros and ones. 

    Parameters
    ----------
    train_pref_suff : tuple of pd.DataFrame
        Contains the 4 or 5 prefix-suffx dfs: 'prefix_df', 'suffix_df', 
        'timeLabel_df' and 'actLabel_df'. Only the first three are needed.
    val_pref_suff : tuple of pd.DataFrame
        Contains the 4 or 5 prefix-suffx dfs: 'prefix_df', 'suffix_df', 
        'timeLabel_df' and 'actLabel_df'. Only the first three are needed.
    test_pref_suff : tuple of pd.DataFrame
        Contains the 4 or 5 prefix-suffx dfs: 'prefix_df', 'suffix_df', 
        'timeLabel_df' and 'actLabel_df'. Only the first three are needed.
    case_id : str 
        Column name of the case IDs (now unique for every prefix-suffix 
        pair, and not just for every case). Needed for standardizing the 
        remaining time targets. 
    pref_numcols : list of str
        Contains column names of numeric columns in the 'prefix_df'. 
    suff_numcols : list of str
        Contains column names of numeric columns in the 'suffix_df'. 
    timelab_numcols : list of str
        Contains column names of numeric columns in the 'timeLabel_df'. 
    outcome : {None, str}
        String representing the name containing the binary outcome 
        column. If `outcome=None`, it will be ignored. 
    log_transformed : bool
        If `True`, all the numeric features and targets with solely 
        positive values will be log-transformed before being standardized.

    Returns
    -------
    train_pref_suff : tuple of pd.DataFrame
        Contains the 4 or 5 prefix-suffx dfs: 'prefix_df', 'suffix_df', 
        'timeLabel_df' and 'actLabel_df'. Only the first three contain 
        numeric values, which are now standardized.
    val_pref_suff : tuple of pd.DataFrame
        Contains the 4 or 5 prefix-suffx dfs: 'prefix_df', 'suffix_df', 
        'timeLabel_df' and 'actLabel_df'. Only the first three contain 
        numeric values, which are now standardized.
    test_pref_suff : tuple of pd.DataFrame
        Contains the 4 or 5 prefix-suffx dfs: 'prefix_df', 'suffix_df', 
        'timeLabel_df' and 'actLabel_df'. Only the first three contain 
        numeric values, which are now standardized.
    indicator_prefix_cols : list of str
        List containing the column names of the added binary indicator 
        columns, if any are added. An empty string otherwise. 
    train_means_dict : dict 
        Dictionary containing 'prefix_df', 'suffix_df' and 'timeLabel_df' 
        as the keys, with the values for each being a list of floats 
        containing the training means used for standardization. The order 
        corresponds to the order of the columns in the ``pref_numcols``, 
        ``suff_numcols`` or ``timelab_numcols`` respectively. 
    train_std_dict : dict 
        Dictionary containing 'prefix_df', 'suffix_df' and 'timeLabel_df' 
        as the keys, with the values for each being a list of floats 
        containing the training standard deviations used for 
        standardization. The order corresponds to the order of the 
        columns in the ``pref_numcols``, ``suff_numcols`` or 
        ``timelab_numcols`` respectively. 
    
    """



    # Defining the needed train and test dataframes. 
    prefix_df_train = train_pref_suff[0]
    suffix_df_train = train_pref_suff[1]
    timeLabel_df_train = train_pref_suff[2]
    actLabel_df_train = train_pref_suff[3]

    prefix_df_val = val_pref_suff[0]
    suffix_df_val = val_pref_suff[1]
    timeLabel_df_val = val_pref_suff[2]
    actLabel_df_val = val_pref_suff[3]

    prefix_df_test = test_pref_suff[0]
    suffix_df_test = test_pref_suff[1]
    timeLabel_df_test = test_pref_suff[2]
    actLabel_df_test = test_pref_suff[3]

    if outcome: 
        outcomeLabel_df_train, outcomeLabel_df_val, outcomeLabel_df_test = train_pref_suff[4], val_pref_suff[4], test_pref_suff[4]

    # Standardizing prefix dfs 
    prefix_df_train, prefix_df_val, prefix_df_test, prefix_means, prefix_std = standardize_train_test_dfs(train=prefix_df_train, 
                                                                                                          val=prefix_df_val,
                                                                                                          test=prefix_df_test, 
                                                                                                          num_cols=pref_numcols, 
                                                                                                          log_transformed=log_transformed)

    # Standardizing suffix dfs 
    suffix_df_train, suffix_df_val, suffix_df_test, suffix_means, suffix_std = standardize_train_test_dfs(train=suffix_df_train, 
                                                                                                          val=suffix_df_val,
                                                                                                          test=suffix_df_test, 
                                                                                                          num_cols=suff_numcols, 
                                                                                                          log_transformed=log_transformed)

    # Standardizing timeLabel dfs 
    # Only the first remaining runtime value for each prefix will be used. Therefore, 
    # we should standardize only based on the mean and standarddeviation of that first 
    # value for each prefix id. 
    timeLabel_df_train, timeLabel_df_val, timeLabel_df_test, timeLabel_means, timeLabel_std = standardizeTimeLabel_train_test_dfs(train=timeLabel_df_train, 
                                                                                                                                  val = timeLabel_df_val,
                                                                                                                                  test=timeLabel_df_test, 
                                                                                                                                  case_id=case_id, 
                                                                                                                                  log_transformed=log_transformed)
    
    # Create a dictionary containing the training means used for standardization, for each of the (3) dataframes. 
    # Order is the same as order of numeric columns in their respective num_cols lists. 
    train_means_dict = {'prefix_df': prefix_means, 'suffix_df': suffix_means, 'timeLabel_df': timeLabel_means}
    # Create a dictionary containing the training std devs used for standardization, for each of the (3) dataframes. 
    train_std_dict = {'prefix_df': prefix_std, 'suffix_df': suffix_std, 'timeLabel_df': timeLabel_std}
    
    # If any numeric col, either in train or test df, contains missing values: 
    # impute with 0 and add binary indicator column. 

    indicator_prefix_cols = []
    for num_col in pref_numcols:
        if prefix_df_train[num_col].isna().any() or prefix_df_val[num_col].isna().any() or prefix_df_test[num_col].isna().any():
            # Make new binary indicator column
            col_name = num_col + '_missing'
            indicator_prefix_cols.append(col_name)
            prefix_df_train[col_name] = np.where(prefix_df_train[num_col].isna(), 1, 0)
            prefix_df_val[col_name] = np.where(prefix_df_val[num_col].isna(), 1, 0)
            prefix_df_test[col_name] = np.where(prefix_df_test[num_col].isna(), 1, 0)

            # Impute missing values original column with 0
            values = {num_col : 0}
            prefix_df_train = prefix_df_train.fillna(value = values)
            prefix_df_val = prefix_df_val.fillna(value = values)
            prefix_df_test = prefix_df_test.fillna(value = values)

    # Changes needed here too! Change the inputs, ask for the 3 dataframes separately, because actLabel also not needed. Make it a tuple again after this function. 
    if outcome:
        train_pref_suff = prefix_df_train, suffix_df_train, timeLabel_df_train, actLabel_df_train, outcomeLabel_df_train
        val_pref_suff = prefix_df_val, suffix_df_val, timeLabel_df_val, actLabel_df_val, outcomeLabel_df_val
        test_pref_suff = prefix_df_test, suffix_df_test, timeLabel_df_test, actLabel_df_test, outcomeLabel_df_test
    else:
        train_pref_suff = prefix_df_train, suffix_df_train, timeLabel_df_train, actLabel_df_train
        val_pref_suff = prefix_df_val, suffix_df_val, timeLabel_df_val, actLabel_df_val
        test_pref_suff = prefix_df_test, suffix_df_test, timeLabel_df_test, actLabel_df_test
    
    return train_pref_suff, val_pref_suff, test_pref_suff, indicator_prefix_cols, train_means_dict, train_std_dict

