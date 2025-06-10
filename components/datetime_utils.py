import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Union

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


def prepare_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame for Streamlit display by converting datetime columns.
    
    Args:
        df: DataFrame to prepare
        
    Returns:
        DataFrame with datetime columns converted to strings
    """
    df_copy = df.copy()
    
    # Convert datetime columns to strings
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        elif df_copy[col].dtype == 'object':
            # Check if column contains datetime strings
            try:
                sample = df_copy[col].dropna().iloc[0] if len(df_copy[col].dropna()) > 0 else None
                if sample and isinstance(sample, str) and 'T' in sample and ':' in sample:
                    df_copy[col] = df_copy[col].apply(lambda x: format_datetime_for_display(x) if pd.notna(x) else x)
            except:
                pass
    
    return df_copy


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
        # Check if it's an ISO datetime string
        if 'T' in data and ':' in data:
            try:
                return format_datetime_for_display(data)
            except:
                return data
        return data
    else:
        return data