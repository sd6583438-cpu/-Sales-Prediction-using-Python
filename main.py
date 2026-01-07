import os
import sys
import pandas as pd
from data_loader import load_data
from preprocessing import clean_data, preprocess_features
from eda import perform_eda
from model import train_model, analyze_coefficients

def main():
    print("Starting Sales Prediction Pipeline...")
    
    # 1. Load Data
    filepath = os.path.join(os.getcwd(), 'Advertising.csv')
    try:
        df = load_data(filepath)
    except FileNotFoundError as e:
        print(e)
        return

    # 2. EDA
    print("\n--- Performing Exploratory Data Analysis ---")
    perform_eda(df)

    # 3. Data Cleaning
    print("\n--- Cleaning Data ---")
    df = clean_data(df)
    
    # Keeps original for interpretation
    df_original = df.copy()

    # 4. Preprocessing (Scaling)
    # Note: For Linear Regression, scaling isn't strictly necessary for prediction accuracy 
    # but helps in comparing coefficients if features have different scales.
    # However, 'Sales' should generally not be scaled if we want interpretable RMSE in original units.
    # The current preprocess_features scales everything including target? 
    # Let's check preprocessing.py. 
    # It separates target, scales features, then re-attaches target. So Sales is NOT scaled. Good.
    
    print("\n--- Preprocessing Features ---")
    df_scaled, scaler = preprocess_features(df)
    
    # 5. Model Training & Evaluation
    print("\n--- Training Model ---")
    model, metrics, X_test, y_test, y_pred = train_model(df_scaled)
    
    # 6. Analysis
    print("\n--- Analyzing Impact ---")
    # Feature names are the columns of X_test
    analyze_coefficients(model, X_test.columns)
    
    # 7. Insights
    print("\n--- Key Insights ---")
    coefs = pd.DataFrame({'Feature': X_test.columns, 'Coefficient': model.coef_})
    max_impact = coefs.loc[coefs['Coefficient'].abs().idxmax()]
    print(f"The most influential factor is {max_impact['Feature']} with a coefficient of {max_impact['Coefficient']:.4f}.")
    print("Positive coefficients indicate a positive relationship with Sales.")
    
    # Save Report
    with open('report.txt', 'w') as f:
        f.write("Sales Prediction Report\n=======================\n\n")
        f.write(f"MSE: {metrics['MSE']:.4f}\n")
        f.write(f"RMSE: {metrics['RMSE']:.4f}\n")
        f.write(f"R2 Score: {metrics['R2']:.4f}\n\n")
        f.write("Feature Coefficients:\n")
        f.write(coefs.to_string())
        f.write(f"\n\nKey Insight: The most influential factor is {max_impact['Feature']} with a coefficient of {max_impact['Coefficient']:.4f}.")
    print("\nReport saved to report.txt")

    # Save Model and Scaler
    import joblib
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model saved to model.pkl")
    print("Scaler saved to scaler.pkl")

    print("\nPipeline Completed Successfully.")

if __name__ == "__main__":
    main()
