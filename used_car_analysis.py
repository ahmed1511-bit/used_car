
import streamlit as st
import plotly.express as px
import pandas as pd

st.title("🚗 Vehicle Data - Multivariate Analysis")

# Load Data
df = pd.read_csv("cleaned vehicles.csv")

# Sidebar
st.sidebar.header("Filters")
manufacturers = st.sidebar.multiselect(
    "Manufacturer",
    options=df["manufacturer"].unique(),
    default=df["manufacturer"].unique()
)
conditions = st.sidebar.multiselect(
    "Condition",
    options=df["condition"].unique(),
    default=df["condition"].unique()
)

df = df[df["manufacturer"].isin(manufacturers) & df["condition"].isin(conditions)]

# 1. Correlation Heatmap
st.subheader("1. Correlation Heatmap")
numerical_cols = df.select_dtypes(include="number").columns
corr = df[numerical_cols].corr()
fig1 = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Correlation Heatmap"
)
st.plotly_chart(fig1, use_container_width=True)

# 2. Scatter Matrix
st.subheader("2. Scatter Matrix")
fig2 = px.scatter_matrix(
    df,
    dimensions=["price", "year", "odometer", "cylinders"],
    color="condition",
    title="Scatter Matrix"
)
st.plotly_chart(fig2, use_container_width=True)

# 3. Price vs Year by Condition
st.subheader("3. Price vs Year by Condition")
fig3 = px.scatter(
    df,
    x="year",
    y="price",
    color="condition",
    title="Price vs Year by Condition",
    hover_data=["manufacturer", "type"]
)
st.plotly_chart(fig3, use_container_width=True)

# 4. Price vs Odometer by Fuel
st.subheader("4. Price vs Odometer by Fuel Type")
fig4 = px.scatter(
    df,
    x="odometer",
    y="price",
    color="fuel",
    title="Price vs Odometer by Fuel Type",
    hover_data=["manufacturer", "year"]
)
st.plotly_chart(fig4, use_container_width=True)

# 5. Price by Manufacturer and Type
st.subheader("5. Price by Manufacturer and Vehicle Type")
fig5 = px.box(
    df,
    x="manufacturer",
    y="price",
    color="type",
    title="Price by Manufacturer and Vehicle Type"
)
fig5.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig5, use_container_width=True)

# 6. Price by Condition and Transmission
st.subheader("6. Price by Condition and Transmission")
fig6 = px.box(
    df,
    x="condition",
    y="price",
    color="transmission",
    title="Price by Condition and Transmission"
)
st.plotly_chart(fig6, use_container_width=True)

# 7. Price by Title Status
st.subheader("7. Price by Title Status")
fig7 = px.box(
    df,
    x="title_status",
    y="price",
    color="title_status",
    title="Price by Title Status",
    points="outliers"
)
st.plotly_chart(fig7, use_container_width=True)

# 8. Price by Drive and Type
st.subheader("8. Price by Vehicle Type and Drive")
fig8 = px.box(
    df,
    x="type",
    y="price",
    color="drive",
    title="Price by Vehicle Type and Drive"
)
fig8.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig8, use_container_width=True)
