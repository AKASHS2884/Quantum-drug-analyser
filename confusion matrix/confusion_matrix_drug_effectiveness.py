



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


file_path = "drug_effectiveness_data.csv"
df = pd.read_csv(file_path)


y_true = df["True_Class"]
y_pred = df["Predicted_Class"]


cm = confusion_matrix(y_true, y_pred, labels=["Effective", "Not Effective"])


print("Confusion Matrix:\n", cm)


print("\nAccuracy Score:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Effective", "Not Effective"], yticklabels=["Effective", "Not Effective"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()