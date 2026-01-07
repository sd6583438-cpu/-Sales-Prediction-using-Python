import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def perform_eda(df, output_dir='plots'):
    """
    Generates EDA plots and saves them to the output directory.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        output_dir (str): Directory to save plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    print(f"Saved correlation_matrix.png to {output_dir}")
    
    # Pairplot
    sns.pairplot(df)
    plt.savefig(os.path.join(output_dir, 'pairplot.png'))
    plt.close()
    print(f"Saved pairplot.png to {output_dir}")
    
    # Feature vs Sales Scatter Plots
    features = [col for col in df.columns if col != 'Sales']
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[feature], y=df['Sales'])
        plt.title(f'{feature} vs Sales')
        plt.xlabel(feature)
        plt.ylabel('Sales')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature}_vs_sales.png'))
        plt.close()
        print(f"Saved {feature}_vs_sales.png to {output_dir}")
