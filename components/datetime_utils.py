"""
Enhanced DataFrame preparation in components/datetime_utils.py
This handles currency strings and other problematic data types.
"""

import pandas as pd
import re
from datetime import datetime
from typing import Any, Dict, List, Union

def prepare_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame for Streamlit display by converting problematic data types.
    
    Args:
        df: DataFrame to prepare
        
    Returns:
        DataFrame with all columns converted to Streamlit-compatible formats
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Handle datetime columns
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
        # Handle object columns (strings, mixed types)
        elif df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].apply(lambda x: _clean_cell_value(x))
    
    return df_copy


def _clean_cell_value(value: Any) -> Any:
    """
    Clean individual cell values for Streamlit compatibility.
    
    Args:
        value: Cell value to clean
        
    Returns:
        Cleaned value suitable for Streamlit display
    """
    if pd.isna(value):
        return value
    
    if isinstance(value, str):
        # Handle currency strings (e.g., '$0.01', '$1,234.56')
        if value.startswith('$'):
            try:
                # Remove $ and commas, convert to float
                numeric_str = value[1:].replace(',', '')
                return float(numeric_str)
            except ValueError:
                # If conversion fails, return as string
                return value
        
        # Handle datetime strings
        if 'T' in value and ':' in value:
            try:
                return format_datetime_for_display(value)
            except:
                return value
        
        # Handle percentage strings (e.g., '25%', '100.5%')
        if value.endswith('%'):
            try:
                # Remove % and convert to float
                numeric_str = value[:-1]
                return float(numeric_str)
            except ValueError:
                return value
        
        # Handle numeric strings with commas (e.g., '1,234', '1,234.56')
        if re.match(r'^[\d,]+\.?\d*$', value):
            try:
                return float(value.replace(',', ''))
            except ValueError:
                return value
    
    # Handle datetime objects
    elif isinstance(value, datetime):
        return format_datetime_for_display(value)
    
    # Handle lists/dicts by converting to string
    elif isinstance(value, (list, dict)):
        return str(value)
    
    return value


def format_datetime_for_display(dt: Union[str, datetime]) -> str:
    """
    Format datetime for display in Streamlit.
    
    Args:
        dt: Datetime object or ISO string
        
    Returns:
        Formatted datetime string
    """
    if isinstance(dt, str):
        try:
            # Handle ISO format with or without timezone
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except:
            return dt
    
    if isinstance(dt, datetime):
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return str(dt)


def sanitize_dict_for_display(data: Union[Dict, List]) -> Union[Dict, List]:
    """
    Recursively sanitize a dictionary or list for display by converting datetime objects.
    
    Args:
        data: Dictionary or list to sanitize
        
    Returns:
        Sanitized data structure
    """
    if isinstance(data, dict):
        return {k: sanitize_dict_for_display(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_dict_for_display(item) for item in data]
    elif isinstance(data, datetime):
        return format_datetime_for_display(data)
    elif isinstance(data, str):
        # Handle currency and percentage strings
        if data.startswith('$') or data.endswith('%'):
            return _clean_cell_value(data)
        # Check if it's an ISO datetime string
        if 'T' in data and ':' in data:
            try:
                return format_datetime_for_display(data)
            except:
                return data
        return data
    else:
        return data