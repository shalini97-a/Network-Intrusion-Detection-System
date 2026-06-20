import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("models/ids_pipeline.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

st.set_page_config(page_title="Network Intrusion Detection", layout="wide")

st.title("Network Intrusion Detection System")
st.write("Machine Learning based minor project using Network Intrusion Detection Dataset")

uploaded_file = st.file_uploader("Upload Test_data.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "class" in df.columns:
        df_input = df.drop("class", axis=1)
    else:
        df_input = df.copy()

    predictions = model.predict(df_input)
    predicted_labels = label_encoder.inverse_transform(predictions)

    result_df = df.copy()
    result_df["Predicted_Class"] = predicted_labels

    st.subheader("Prediction Results")
    st.dataframe(result_df.head(50))

    normal_count = list(predicted_labels).count("normal")
    attack_count = len(predicted_labels) - normal_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(predicted_labels))
    col2.metric("Normal Traffic", normal_count)
    col3.metric("Attack Traffic", attack_count)

    st.subheader("Normal vs Attack Distribution")

    counts = pd.Series(predicted_labels).value_counts()

    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Intrusion Detection Result")
    st.pyplot(fig)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Prediction CSV",
        csv,
        "network_intrusion_predictions.csv",
        "text/csv"
    )

else:
    st.info("Upload Test_data.csv to get predictions.")