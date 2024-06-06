"""Preprocessing package. Takes an event log as a 
pandas dataframe and returns the train and test sets, including 
labels etc, as tuples of tensors. You only need the log_to_tensors() 
function inside the from_log_to_tensors.py module. 
"""

from Preprocessing.from_log_to_tensors import log_to_tensors