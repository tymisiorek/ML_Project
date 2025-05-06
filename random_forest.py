import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

subset_path = Path.home() / "Documents/2025Spring/CS4774/Dataset/network_intrusion/data/full_data.csv"
df = pd.read_csv(subset_path)
print("Loaded data:", df.shape)

#Keep numeric features + Flow ID for grouping
groups = df["Flow ID"]
X = df.drop(columns=["Label"]).select_dtypes(include=[np.number])
y = df["Label"]

#clean infinities & missing
X.replace([np.inf, -np.inf], np.nan, inplace=True)
mask = X.notna().all(axis=1)
X, y, groups = X[mask], y[mask], groups[mask]

#drop tiny classes
vc = y.value_counts()
keep = vc[vc >= 2].index
mask = y.isin(keep)
X, y, groups = X[mask], y[mask], groups[mask]
print("After cleaning:", X.shape, "classes:", len(vc))

#Group‐aware split: 70% of FLOW IDs train, 30%  test
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("Train/test sizes:", X_train.shape, X_test.shape)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#train pruned rf
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

#eval
y_pred = rf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Results:\n", classification_report(y_test, y_pred))

# 9) Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test),
            cmap="Greens")
plt.title("RF Confusion Matrix (Group‐aware split, depth=10)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# True vs predicted: BENIGN or !BENIGN
def to_binary(lbl):
    return "BENIGN" if lbl == "BENIGN" else "!BENIGN"

y_test_bin = y_test.map(to_binary)
y_pred_bin = pd.Series(y_pred, index=y_test.index).map(to_binary)
labels = ["BENIGN", "!BENIGN"]
cm_bin = confusion_matrix(y_test_bin, y_pred_bin, labels=labels)
plt.figure(figsize=(5,4))
sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("RF Confusion Matrix: BENIGN vs !BENIGN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()