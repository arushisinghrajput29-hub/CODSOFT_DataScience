
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Important Columns Select
df = df[['Survived','Pclass','Sex','Age','Fare']]

# Missing Values Fill
df['Age']=df['Age'].fillna(df['Age'].mean())

# Convert Male/Female into Numbers
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Features and Target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)