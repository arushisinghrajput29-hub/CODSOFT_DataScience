import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load Dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

# Important Columns
df = df[['Genre', 'Director', 'Actor 1', 'Rating']]

# Missing Values Remove
df = df.dropna()

# Convert Text to Numbers
le = LabelEncoder()

df['Genre'] = le.fit_transform(df['Genre'])
df['Director'] = le.fit_transform(df['Director'])
df['Actor 1'] = le.fit_transform(df['Actor 1'])

# Features and Target
X = df[['Genre', 'Director', 'Actor 1']]
y = df['Rating']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)