import joblib
from data_preprocessing import preprocess_data
from model_training import train_and_evaluate

# Load the saved model
model = joblib.load('fraud_detection_model.pkl')

# Use the preprocess_data function from data_preprocessing.py
X_train, X_test, y_train, y_test = preprocess_data('creditcard.csv', sample_fraction=0.2)

# Use the model for predictions
predictions = model.predict(X_test)

# train and evaluate the model
train_and_evaluate(X_train, X_test, y_train, y_test)

# Now X_train, X_test, y_train, and y_test are ready for model training
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
