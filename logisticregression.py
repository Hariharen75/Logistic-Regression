import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.title("🚢 Titanic Survival Prediction – Logistic Regression")

# Upload dataset
uploaded_file = st.file_uploader("Upload Titanic-Dataset.csv", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Drop unwanted columns
    columns_to_drop = ['Name', 'Ticket', 'Cabin']
    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # Handle missing values
    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)

    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    if 'Fare' in df.columns:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    for col in ['Sex', 'Embarked']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    st.subheader("🧹 Cleaned Dataset")
    st.dataframe(df.head())

    # Features & target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    st.subheader("✅ Model Accuracy")
    st.write(f"**Accuracy:** {acc:.2f}")

    # Classification report
    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("📉 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=['Not Survived', 'Survived'],
        yticklabels=['Not Survived', 'Survived'],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix – Titanic Survival")

    st.pyplot(fig)

else:
    st.info("👆 Please upload the Titanic-Dataset.csv file to continue.")
