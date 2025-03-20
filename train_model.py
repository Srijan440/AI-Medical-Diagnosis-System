# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, f1_score
import numpy as np
import joblib

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(url, names=column_names)

# Data Preprocessing: Replace 0 values in certain columns with NaN and fill missing values with median
columns_to_check = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[columns_to_check] = data[columns_to_check].replace(0, pd.NA)
data.dropna(subset=["Glucose"], inplace=True)
data.fillna(data.median(), inplace=True)

# Feature Engineering: Add new meaningful features
data["Glucose_BMI_Ratio"] = data["Glucose"] / data["BMI"]
data["Age_Glucose_Ratio"] = data["Age"] / data["Glucose"]
data["BloodPressure_BMI_Ratio"] = data["BloodPressure"] / data["BMI"]
data["Insulin_Glucose_Ratio"] = data["Insulin"] / data["Glucose"]

# Split data into features (X) and target (y)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Apply BorderlineSMOTE to balance the dataset
smote = BorderlineSMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use in Flask app
joblib.dump(scaler, "scaler.pkl")

# Build the neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,  # Adjusted batch size
    callbacks=[early_stopping],
    class_weight={0: 1, 1: 2}  # Adjust class weights for imbalance
)

# Evaluate the model
y_pred_prob = model.predict(X_test)

# Compute ROC curve and find optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold:", optimal_threshold)

# Use the optimal threshold for predictions
y_pred = (y_pred_prob > optimal_threshold).astype(int)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and threshold for the Flask app
model.save("diabetes_model.h5")
joblib.dump(optimal_threshold, "optimal_threshold.pkl")