import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load Dataset
df = pd.read_csv("creditcard.csv")

# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Normalize Amount Column
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# Handle Imbalanced Data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))