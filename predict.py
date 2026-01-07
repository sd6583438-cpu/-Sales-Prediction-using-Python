import joblib
import pandas as pd
import os
import sys

def predict_sales():
    print("--- Sales Prediction Tool ---")
    print("Enter the advertising budget for each channel to predict Sales.")
    
    # Load Model and Scaler
    if not os.path.exists('model.pkl') or not os.path.exists('scaler.pkl'):
        print("Error: Model or Scaler not found. Please run main.py first to train the model.")
        return

    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # User Input
    try:
        tv = float(input("TV Spend ($): "))
        radio = float(input("Radio Spend ($): "))
        newspaper = float(input("Newspaper Spend ($): "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return
        
    # Create DataFrame
    data = pd.DataFrame({
        'TV': [tv],
        'Radio': [radio],
        'Newspaper': [newspaper]
    })
    
    # Preprocess
    # IMPORTANT: Use transform(), not fit_transform()
    data_scaled_array = scaler.transform(data)
    
    # Predict
    prediction = model.predict(data_scaled_array)
    
    print("\n-----------------------------")
    print(f"Predicted Sales: {prediction[0]:.2f} units")
    print("-----------------------------\n")

if __name__ == "__main__":
    predict_sales()
