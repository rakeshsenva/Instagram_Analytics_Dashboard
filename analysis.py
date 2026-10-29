


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv("instagram_reach.csv")
df["date"] = pd.to_datetime(df["date"])


print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

numerical_columns = df.select_dtypes(include="number").columns
print("Numerical columns:", list(numerical_columns))


print("Mean reach:", np.mean(df["reach"]))
print("Median reach:", np.median(df["reach"]))
print("Reach standard deviation:", np.std(df["reach"]))

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="post_type", y="reach")
plt.title("Reach Distribution by Post Type")
plt.xticks(rotation=20)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["reach"])
plt.title("Instagram Reach Trend")
plt.xlabel("Date")
plt.ylabel("Reach")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


fig = px.scatter(
    df, x="followers", y="reach", color="post_type",
    size="likes", hover_data=["hour", "hashtags"]
)
fig.show()

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

features = [
    "post_type", "hour", "followers", "caption_length", "hashtags",
    "likes", "comments", "shares", "saves", "impressions", "engagement_rate"
]

X = df[features]
y = df["reach"]

categorical = ["post_type"]
numeric = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric)
])

model = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=250, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))
