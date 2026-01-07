import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_model(df, target_column='Sales'):
    """
    Trains a Linear Regression model.
    
    Args:
        df (pd.DataFrame): Dataframe with features and target.
        target_column (str): Name of the target column.
        
    Returns:
        model: Trained model.
        metrics (dict): Dictionary containing MSE, RMSE, and R2 score.
        X_test, y_test, y_pred: Test data and predictions for analysis.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
    
    print("Model Training Completed.")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return model, metrics, X_test, y_test, y_pred

def analyze_coefficients(model, feature_names):
    """
    Returns the coefficients of the model.
    
    Args:
        model: Trained Linear Regression model.
        feature_names (list): List of feature names.
        
    Returns:
        pd.DataFrame: Dataframe containing feature names and their coefficients.
    """
    coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    coefs = coefs.sort_values(by='Coefficient', ascending=False)
    print("\nFeature Coefficients:")
    print(coefs)
    return coefs
