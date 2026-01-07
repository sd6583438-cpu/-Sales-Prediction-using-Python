import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Checks for missing values and handles them.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values found:\n", missing_values)
        df = df.dropna() # Simple drop for now, can be imputed if needed
        print("Dropped rows with missing values.")
    else:
        print("No missing values found.")
        
    return df

def preprocess_features(df, target_column='Sales'):
    """
    Scales numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        
    Returns:
        pd.DataFrame: Dataframe with scaled features.
        scaler: The fitted StandardScaler object.
    """
    features = df.drop(columns=[target_column])
    target = df[target_column]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns, index=df.index)
    scaled_df[target_column] = target
    
    print("Features scaled using StandardScaler.")
    return scaled_df, scaler
