
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Instagram Reach Analysis", page_icon="📊", layout="wide")

st.title("📊 Instagram Reach Analysis Dashboard")
st.caption("NumPy • Pandas • Seaborn • Matplotlib • Plotly • Machine Learning")

@st.cache_data
def load_data():
    data = pd.read_csv("instagram_reach.csv")
    data["date"] = pd.to_datetime(data["date"])
    return data

df = load_data()

st.sidebar.header("🔎 Filters")
types = st.sidebar.multiselect(
    "Post Type",
    options=sorted(df["post_type"].unique()),
    default=sorted(df["post_type"].unique())
)
min_followers, max_followers = int(df.followers.min()), int(df.followers.max())
follower_range = st.sidebar.slider(
    "Followers range", min_followers, max_followers,
    (min_followers, max_followers)
)

filtered = df[
    df["post_type"].isin(types) &
    df["followers"].between(follower_range[0], follower_range[1])
].copy()


c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Posts", f"{len(filtered):,}")
c2.metric("Total Reach", f"{filtered.reach.sum():,.0f}")
c3.metric("Average Reach", f"{filtered.reach.mean():,.0f}")
c4.metric("Avg Engagement", f"{filtered.engagement_rate.mean():.2f}%")

st.divider()

st.subheader("📈 Reach Trend")
daily = filtered.groupby("date", as_index=False)["reach"].mean()
fig = px.line(daily, x="date", y="reach", markers=True,
              title="Average Reach Over Time")
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📱 Reach by Post Type")
    type_df = filtered.groupby("post_type", as_index=False)["reach"].mean().sort_values("reach", ascending=False)
    fig = px.bar(type_df, x="post_type", y="reach", text_auto=".0f",
                 title="Average Reach by Post Type")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("⏰ Best Posting Hours")
    hour_df = filtered.groupby("hour", as_index=False)["reach"].mean()
    fig = px.bar(hour_df, x="hour", y="reach", title="Average Reach by Hour")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🔥 Correlation Analysis")
num_cols = [
    "followers", "caption_length", "hashtags", "likes", "comments",
    "shares", "saves", "impressions", "reach", "engagement_rate"
]
corr = filtered[num_cols].corr()

fig, ax = plt.subplots(figsize=(11, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig, use_container_width=True)


st.subheader("💡 Followers vs Reach")
fig = px.scatter(
    filtered,
    x="followers",
    y="reach",
    color="post_type",
    size="likes",
    hover_data=["date", "hour", "hashtags", "engagement_rate"],
    title="Followers vs Reach"
)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.header("🤖 Machine Learning: Predict Instagram Reach")

features = [
    "post_type", "hour", "followers", "caption_length", "hashtags",
    "likes", "comments", "shares", "saves", "impressions", "engagement_rate"
]
target = "reach"

X = df[features]
y = df[target]

categorical = ["post_type"]
numeric = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric)
    ]
)

model = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=250, random_state=42, max_depth=12, min_samples_leaf=2
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

m1, m2 = st.columns(2)
m1.metric("MAE", f"{mean_absolute_error(y_test, pred):,.0f}")
m2.metric("R² Score", f"{r2_score(y_test, pred):.3f}")

pred_df = pd.DataFrame({"Actual Reach": y_test.values, "Predicted Reach": pred})
fig = px.scatter(
    pred_df, x="Actual Reach", y="Predicted Reach",
    title="Actual vs Predicted Reach",
    trendline="ols"
)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "Note: The included CSV is sample/synthetic data for learning. "
    "For a real project, replace it with your own Instagram Insights export."
)

with st.expander("📄 View filtered data"):
    st.dataframe(filtered, use_container_width=True)
