import pandas as pd
import numpy as np
import seaborn as sns

# Load the Titanic dataset
df =  sns.load_dataset('titanic')
print("Dataset loaded successfully!")
print("shape of dataset: ", df.shape)
print(df.head())

df.info()
print(df.describe())

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print("Numeric columns: ", numeric_cols)
print("Categorical columns: ", categorical_cols)

#unique values in categorical columns
for col in categorical_cols:
    print(f"Unique values in column '{col}': {df[col].unique()}")

print("\n --- SUMMARY ---")
print(f"total rows: {df.shape[0]}")
print(f"total columns: {df.shape[1]}")
print(f"Numerical Columns: {numeric_cols}")
print(f"Categorical Columns: {categorical_cols}")
print(f"total missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")