import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("models", exist_ok=True)

train_df = pd.read_csv("dataset/Train_data.csv")
test_df = pd.read_csv("dataset/Test_data.csv")

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

target_col = "class"

X = train_df.drop(target_col, axis=1)
y = train_df[target_col]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]
)

print("Training model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print("\nAccuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

test_predictions = model.predict(test_df)
test_labels = label_encoder.inverse_transform(test_predictions)

test_output = test_df.copy()
test_output["Predicted_Class"] = test_labels
test_output.to_csv("test_predictions.csv", index=False)

joblib.dump(model, "models/ids_pipeline.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("\nModel saved successfully.")
print("Test predictions saved as test_predictions.csv")