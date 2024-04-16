# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from hpelm import ELM
import joblib

# Set seaborn style for plots
sns.set(style="whitegrid")

# Preprocess
project_folder = "/content/drive/Shareddrives/ML4/ELM"
file_path = f"{project_folder}/preprocessed_.xlsx"

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

# Training with ELM
elm = ELM(X_train.shape[1], 1)  # Number of features and output dimension
elm.add_neurons(50, "sigm")  # Number of neurons and type of activation function
elm.train(X_train, y_train[:, np.newaxis], "r")  # 'r' indicates regression

# Prediction and evaluation
y_pred = elm.predict(X_test)
mse_ELM = np.mean((y_test[:, np.newaxis] - y_pred) ** 2)

# Save the model and scaler
model_path = f"{project_folder}/elm_model.pkl"
scaler_path = f"{project_folder}/scaler.pkl"

joblib.dump(elm, model_path)
joblib.dump(scaler, scaler_path)

print(f"MSE for ELM: {mse_ELM}")
