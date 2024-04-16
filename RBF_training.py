# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from rbflib import RBFNet
import joblib

# Set seaborn style for plots
sns.set(style="whitegrid")

# Preprocess
project_folder = "/content/drive/Shareddrives/ML4/GRNN"
file_path = f"{project_folder}/water quality 1.xlsx"

# Load the dataset into a pandas DataFrame
dataset = pd.read_excel(file_path, header=0)

# Drop 'Date' column if exists
if "Date" in dataset.columns:
    dataset = dataset.drop(columns=["Date"])

# Select features and target
features = dataset.columns[1:-1]
X = dataset[features].copy()
y = dataset["PAC"].copy()

# Data normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training with RBFN
rbfn = RBFNet(
    lr=1e-2, k=10, inferStds=True
)  # Learning rate, number of RBF neurons, and infer standard deviations
rbfn.fit(X_train, y_train)

# Prediction and evaluation
y_pred = rbfn.predict(X_test)
mse_RBFN = np.mean((y_test - y_pred) ** 2)

# Save the model and scaler
model_path = f"{project_folder}/rbfn_model.pkl"
scaler_path = f"{project_folder}/scaler.pkl"

joblib.dump(rbfn, model_path)
joblib.dump(scaler, scaler_path)

print(f"MSE for RBFN: {mse_RBFN}")
