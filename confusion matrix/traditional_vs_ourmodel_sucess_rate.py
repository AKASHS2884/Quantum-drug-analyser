# -*- coding: utf-8 -*-
"""TRADITIONAL VS OURMODEL SUCESS RATE

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KsknXnQ5246DzCXgglrAzFJ6R6mqF6tc
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the dataset
file_path = "/content/drug_discovery_data.csv"
df = pd.read_csv(file_path)

# Extract actual and predicted labels
y_true = df["Actual_Label"]
y_pred_traditional = df["Traditional_Prediction"]
y_pred_ai_quantum = df["AI_Quantum_Prediction"]

# Compute confusion matrices
cm_traditional = confusion_matrix(y_true, y_pred_traditional)
cm_ai_quantum = confusion_matrix(y_true, y_pred_ai_quantum)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Failed Drug", "Successful Drug"], yticklabels=["Failed Drug", "Successful Drug"])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()

# Plot results
plot_confusion_matrix(cm_traditional, "Traditional Drug Identification")
plot_confusion_matrix(cm_ai_quantum, "AI-Quantum Model")

"""# New Section"""